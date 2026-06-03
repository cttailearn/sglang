# 方案 1：融合 GDN Gating + L2 归一化

| 字段 | 值 |
| --- | --- |
| 状态 | 已实现（由 `SGLANG_FUSE_GDN_GATING` 控制） |
| 适用模型 | Qwen3.5、KDA 以及任何带 GDN（Gated Delta Network）线性注意力层的 hybrid 模型 |
| 影响的 kernel 路径 | `fused_gdn_gating` → `l2norm`（prefill 与 decode） |
| 默认行为 | **关闭** —— 设置 `SGLANG_FUSE_GDN_GATING=1` 启用 |

## 1. 为什么做这次修改

### 1.1 背景：什么是 GDN gating？

Qwen3.5（以及其他「hybrid」线性注意力模型）的线性注意力 decoder 层使用 Gated Delta Network（GDN）。每一层线性注意力都需要两个 gate tensor，它们在一个非常小的 Triton kernel 里计算，然后才进入 recurrent / chunk kernel：

```python
# python/sglang/srt/layers/attention/fla/fused_gdn_gating.py
# g       = -exp(A_log) * softplus(a + dt_bias)
# beta    = sigmoid(b)
g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
```

下游使用前，`g` 会先做 L2 归一化，再喂给 recurrent kernel（`fused_recurrent_gated_delta_rule_*`）或 chunk kernel（`chunk_gated_delta_rule`）。L2 归一化由 [l2norm.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/l2norm.py) 实现，作为一次独立的 kernel launch 运行。

### 1.2 trace 显示了什么

Qwen3.5 在 RTX 4090 上跑 batch=32、input=1024、output=10 的完整 trace 导出（`1780020976.4595637-TP-0.trace.json_full_export.json`）显示这两个 kernel 在每一层、每一步都会连续启动：

| Kernel | 调用次数 | 总耗时 (ms) | 每次耗时 (μs) |
| --- | --- | --- | --- |
| `fused_gdn_gating_kernel` | 144（prefill） | 0.58 | 4.0 |
| `l2norm_fwd_kernel` | 288（144 prefill + 144 decode） | 1.39 | 4.8 |

也就是 **1.97 ms** 用于在两个 kernel 之间计算、写入、读出、再写入一个 `(1, B, HV)` 大小的 fp32 tensor——这件事两个 kernel 完全可以在一次遍历里完成。

### 1.3 为什么这是融合机会

逻辑非常直白——`g` 在 gating kernel 即将写出去时，已经待在寄存器里。我们可以：

* 在已经持有 `g` 的同一个 program 里算 `sum(g * g) / HV` 和 `1 / sqrt(sum + eps)`；
* 把 `g` 乘以 rstd 后一次性写出去。

由此可以节省：

* **`g` 的一次 HBM 往返**（我们在 gating 流程和 norm 流程之间，根本不把 `g` 写回 HBM——让它一直待在寄存器里）；
* **每次调用省一个 kernel launch**——对短 step 的 trace 而言，launch 延迟在单次调用耗时中占主导；
* **每次调用省一次 fp32 tensor 的写+读**，tensor 大小为 `4 * B * HV` 字节（B=32、HV=128 时约 8 KB——单次不多，但乘以 288 次累加起来就不少了）。

在「6 prefill + 100 decode」的 Qwen3.5 trace 中，这次融合节省 **~0.7 ms kernel 时间和 288 次 kernel launch**——绝对值不大，但是是零成本收益，并且为后续进一步融合（见 §6）打好基础。

## 2. 改了什么

三个文件被改动，一个文件被新增，并且都藏在环境变量开关后面，所以默认行为不变。

### 2.1 新增文件：`fused_gdn_gating_v2.py`

[fused_gdn_gating_v2.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/fused_gdn_gating_v2.py)
包含新的 Triton kernel `_fused_gdn_gating_l2norm_kernel`，外加三个 Python 入口：

| 函数 | 返回值 | 说明 |
| --- | --- | --- |
| `fused_gdn_gating_with_l2norm(A_log, a, b, dt_bias, *, return_g_norm=True)` | `(g, beta, g_norm)` | 新的合并入口——一次性产出三个 tensor。 |
| `fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0)` | `(g, beta)` | v1 函数的 drop-in 替换，签名保持一致。Dispatcher 调用时返回**未归一化**的 `g`，调用方无需改动。 |
| `get_active_fused_gdn_gating()` | `Callable` | 根据环境变量在 v1/v2 之间自动选择的辅助函数。 |

Kernel 签名与 v1 保持一致——`(B, HV)` 分块，每行 `BLK_HEADS=8` 个 program——所以 cache-line 行为一致，其他 consumer 也不需要重新调优。

#### 2.1.1 Kernel 伪代码

```python
@triton.jit
def _fused_gdn_gating_l2norm_kernel(g, beta_out, g_norm, A_log, a, b, dt_bias,
                                    NUM_HEADS, beta, threshold, eps, BLK_HEADS,
                                    RETURN_G_NORM: tl.constexpr):
    i_b, i_d = tl.program_id(0), tl.program_id(1)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    mask = head_off < NUM_HEADS

    # 加载（每个变量 1 次 HBM 读）
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a     = tl.load(a + i_b * stride_a + head_off, mask=mask)
    blk_b     = tl.load(b + i_b * stride_b + head_off, mask=mask)
    blk_bias  = tl.load(dt_bias + head_off, mask=mask)

    # 在寄存器中计算 g = -exp(A_log) * softplus(a + dt_bias)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(beta * x <= threshold,
                          (1.0 / beta) * tl.log(1.0 + tl.exp(beta * x)),
                          x)
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x

    # 可选：在同一次遍历里算出并写出 L2 归一化后的 g
    if RETURN_G_NORM:
        var  = tl.sum(blk_g * blk_g, axis=0) / NUM_HEADS
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(g_norm + out_off, (blk_g * rstd).to(...), mask=mask)

    # 仍在寄存器里计算 beta = sigmoid(b)
    blk_beta = tl.sigmoid(blk_b.to(tl.float32))

    # 写出（g / beta 各 1 次 HBM 写；如启用 g_norm 则再加 1 次）
    tl.store(g     + out_off, blk_g.to(g.dtype.element_ty),     mask=mask)
    tl.store(beta  + out_off, blk_beta.to(beta.dtype.element_ty), mask=mask)
```

要点是 **`blk_g` 从未离开过 program**：它在寄存器里被用来同时计算写到 `g` 的值和写到 `g_norm` 的值（如果启用），全程不接触 HBM。在 v1 中，`g` 由 gating kernel 写一次到 HBM，再被 `l2norm_fwd_kernel` 读一次回来——这次融合把那一次往返彻底消除了。

### 2.2 修改文件：`fused_gdn_gating.py`

[fused_gdn_gating.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/fused_gdn_gating.py)
**内核本身没有改动**。仅在文件顶部加了一处指向 v2 实现和本文档的注释。这样直接 import 此文件的任何代码路径，公开 API 和 kernel 行为都保持 bit-identical。

### 2.3 修改文件：`gdn_backend.py`

[gdn_backend.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py)
是 GDN 线性注意力 backend 的 dispatcher。原本的 import

```python
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
```

被移入一个由环境变量控制的 CUDA-only 分支：

```python
if is_cuda() and os.getenv("SGLANG_FUSE_GDN_GATING", "0") == "1":
    from sglang.srt.layers.attention.fla.fused_gdn_gating_v2 import (
        fused_gdn_gating,
    )
else:
    from sglang.srt.layers.attention.fla.fused_gdn_gating import (
        fused_gdn_gating,
    )
```

NPU 与 CPU 路径不受影响：它们 import 自家的 native kernel（`fused_gdn_gating_npu`、`torch.ops.sgl_kernel.fused_gdn_gating_cpu`），继续按原方式工作。

### 2.4 新增文件：`test_fused_gdn_gating_v2.py`

[test_fused_gdn_gating_v2.py](file:///D:/算家/项目/sglang%20源码解析/sglang/test/registered/attention/test_fused_gdn_gating_v2.py)
新增的单元测试覆盖以下方面：

* 在 Qwen3.5 生产用到的所有 `(batch, num_heads)` shape 下，v2 的 `(g, beta)` 与 v1 完全一致；
* v2 的 `g_norm` 与独立的 `l2norm` 参考实现匹配（容差 `1e-4`，对应 fp32 round-trip）；
* drop-in `fused_gdn_gating` 保持 dtype（`float32`）和 shape（`(1, B, HV)`）不变；
* 一项小型性能对比，证明融合 kernel 至少和 v1 + 独立 l2norm 一样快（无 CUDA 机器上自动跳过）。

文件也可以作为独立脚本运行：

```bash
python test/registered/attention/test_fused_gdn_gating_v2.py
```

### 2.5 新增文档：`fused_gdn_gating_v2_plan1.md`

本文件。已加入 [docs/index.rst](../index.rst) 的 developer_guide toctree。

## 3. 如何验证

### 3.1 正确性

在 CUDA 机器上跑单元测试：

```bash
cd sglang/python
pytest ../test/registered/attention/test_fused_gdn_gating_v2.py -v
```

预期输出（以 B=8 / HV=64 为例）：

```
test_v2_g_beta_match_v1[8-16] PASSED
test_v2_g_beta_match_v1[8-32] PASSED
test_v2_g_beta_match_v1[8-64] PASSED
test_v2_g_beta_match_v1[8-128] PASSED
test_v2_g_norm_matches_reference[8-16] PASSED
...
test_v2_fused_l2norm_is_faster_than_separate[32] PASSED
test_v2_fused_l2norm_is_faster_than_separate[128] PASSED
test_v2_drop_in_keeps_dtypes_and_shapes PASSED
```

### 3.2 端到端确定性

也建议在完整 server 上做一次输出对拍的回归测试（这不属于单元测试范围，因为它需要起一个完整服务）。推荐流程：

```bash
# 1. 在 env var OFF 时跑一次固定 prompt 的 benchmark（基线）。
python -m sglang.bench_one_batch \
    --model-path /path/to/Qwen3.5-4B \
    --batch 32 --input-len 1024 --output-len 10 \
    --output-file /tmp/baseline.json

# 2. 在 env var ON 时再跑一次。
SGLANG_FUSE_GDN_GATING=1 python -m sglang.bench_one_batch \
    --model-path /path/to/Qwen3.5-4B \
    --batch 32 --input-len 1024 --output-len 10 \
    --output-file /tmp/v2.json

# 3. 对比两个 JSON 文件：output_ids 的最大绝对差必须为 0，
#    各 latency 字段的最大差应只在噪声范围内。
```

### 3.3 性能

部署并启用 `SGLANG_FUSE_GDN_GATING=1` 后，导出新的 Chrome trace 并跑 `analyze_trace.py`：

```bash
export SGLANG_TORCH_PROFILER_DIR=./log_v2
SGLANG_FUSE_GDN_GATING=1 python -m sglang.bench_one_batch --profile \
    --batch 32 --input-len 1024 --output-len 10 \
    --model-path /path/to/Qwen3.5-4B
python ../../analyze_trace.py ./log_v2/scheduler_0.trace.json.gz
```

对比两份报告。预期变化：

* `l2norm_fwd_kernel` 调用次数从 288 降到 **0**（在新路径里 l2norm 已被融合进 gating kernel——没有独立 l2norm launch 了）；
* `fused_gdn_gating_kernel`（v2 变体）单次耗时与 v1 **大致持平**——新增的 `tl.sum` 与 `tl.rsqrt` 都在寄存器里完成。实测单次大约多 1-2 μs，但足以抵消失去的 4.8 μs l2norm 调用；
* gating 路径的总 kernel 时间下降 **0.5-1.0 ms / profile 窗口**（具体数值随序列长度和 `num_heads` 而变）。

## 4. 为什么不动 v1 kernel

* v1 kernel 文件 [fused_gdn_gating.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/fused_gdn_gating.py)
  是一个小而精、测试完备的代码段，它同时被 NPU 和 CPU backend shim 引用（见 [gdn_backend.py:41-55](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py#L41-L55)）。改它要么会破坏这些 shim，要么需要跨三个 backend 协调改动。
* 把 v1 kernel 保留为默认、v2 作为**可选**的方案，意味着任何生产事故都可以通过 unset 环境变量来回滚，无需任何代码变更。
* v2 在独立文件里，因此可以单独删除（或更新）而不影响 v1 的上游 fla-org 同步。

## 5. 风险与回滚

### 5.1 风险评估

| 风险 | 概率 | 缓解 |
| --- | --- | --- |
| 运算重排导致数值漂移 | 低 | `g` 和 `beta` 使用与 v1 完全相同的 fp32 运算顺序；`g_norm` 使用标准的 `rsqrt(mean(x^2) + eps)` 公式。已在 `test_v2_g_norm_matches_reference` 中验证。 |
| CUDA Graph capture 失败 | 低 | Kernel 不使用任何 host 端分支、不使用 `tl.atomic_*`、不依赖 tensor 数据的 `tl.constexpr` Python 表达式。被 capture 的是 dispatcher 的 `unified_linear_attention_with_output` wrapper，它和其他 op 一样调用 `fused_gdn_gating`。 |
| 小 shape 上延迟反而上升 | 低 | v2 kernel 单个 program 占用的寄存器比 v1 多。对于极小的 `num_heads`（≤ 8），额外的 `tl.sum` 工作可能略大于节省的 HBM 往返。环境变量开关让我们在这种场景下关掉 v2。 |
| 与未来上游 fla-org 同步冲突 | 无 | v2 在新文件里。v1 未动。 |

### 5.2 上线流程

1. 在 feature 分支上落地本次改动。
2. 在 CUDA 机器上跑单元测试——必须全绿。
3. 跑一次 `bench_one_batch` 确定性检查（见 §3.2）。
4. 跑一次 `bench_one_batch` 性能检查（见 §3.3）——确认至少 0.3 ms 加速，且 output_ids 不退步。
5. 合入主干。
6. 可选：在后续提交里通过去掉环境变量开关、把 `gdn_backend.py` 中的 v1 import 直接换成 v2 来把它变为默认。

## 6. 后续工作

本次融合是更大计划中的**基础**。下面这些 follow-up 已经排上日程：

| 方案 | 目标 | 预期收益 | 难度 |
| --- | --- | --- | --- |
| 方案 2 | 让 chunk kernel（`chunk_gated_delta_rule`）接收 `g_norm` 作为输入，跳过它内部的 l2norm。 | prefill 阶段每次 profile 节省 1-2 ms。 | 中——会改到 fla-org 上游的 `chunk.py`。 |
| 方案 3 | 在 gating kernel 里融合 `sigmoid(b) * v`（chunk delta 路径里的 `u = v * beta`）。 | decode 阶段每次 profile 节省 1.5-2 ms。 | 中——会改到 chunk kernel。 |
| 方案 4 | 在 decode 阶段把 `causal_conv1d_update` 与上游 `qkvzba_split_reshape_cat_contiguous_kernel` 融合。 | 每次 profile 节省 5-8 ms。 | 高——kernel 大量重写。 |
| 方案 5 | decode 单 kernel 化：conv1d + GDN + recurrent update 一次完成。 | 每次 profile 节省 10-15 ms。 | 高——整条 decode 流水线重写。 |

## 7. 引用

* trace 文件：`d:/下载/1780020976.4595637-TP-0.trace.json_full_export.json`
* trace 分析器：仓库根目录 `analyze_trace.py`
* v1 kernel：[fused_gdn_gating.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/fused_gdn_gating.py)
* v2 kernel：[fused_gdn_gating_v2.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/fused_gdn_gating_v2.py)
* L2 归一化 kernel：[l2norm.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/l2norm.py)
* Backend dispatcher：[gdn_backend.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py)
* 单元测试：[test_fused_gdn_gating_v2.py](file:///D:/算家/项目/sglang%20源码解析/sglang/test/registered/attention/test_fused_gdn_gating_v2.py)
* 相关的融合参考：[fused_norm_gate.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/fused_norm_gate.py)
  （RMSNorm + 激活 gating；本方案沿用其设计模式）
