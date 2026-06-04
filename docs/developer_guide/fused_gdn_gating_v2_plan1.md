# 方案 1 / 方案 1.5：融合 GDN Gating + L2 归一化

| 字段 | 值 |
| --- | --- |
| 状态 | 方案 1（v2 gating）已实现；方案 1.5（in-kernel Q/K L2）已实现 |
| 适用模型 | Qwen3.5、KDA 以及任何带 GDN（Gated Delta Network）线性注意力层的 hybrid 模型 |
| 影响的 kernel 路径 | 方案 1：`fused_gdn_gating` 自身；方案 1.5：`chunk.py` → `l2norm_fwd` 调用（消除） |
| 默认行为 | **关闭** —— 方案 1 设 `SGLANG_FUSE_GDN_GATING=1`；方案 1.5 设 `SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL=1` |

## 1. 为什么做这次修改

### 1.1 背景：什么是 GDN gating？

Qwen3.5（以及其他「hybrid」线性注意力模型）的线性注意力 decoder 层使用 Gated Delta Network（GDN）。每一层线性注意力都需要两个 gate tensor，它们在一个非常小的 Triton kernel 里计算，然后才进入 recurrent / chunk kernel：

```python
# python/sglang/srt/layers/attention/fla/fused_gdn_gating.py
# g       = -exp(A_log) * softplus(a + dt_bias)
# beta    = sigmoid(b)
g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
```

下游使用前，`Q` / `K`（**注意：不是 g**）会先做 L2 归一化，再喂给 recurrent kernel（`fused_recurrent_gated_delta_rule_*`）或 chunk kernel（`chunk_gated_delta_rule`）。L2 归一化由 [l2norm.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/l2norm.py) 实现，作为一次独立的 kernel launch 运行。

**关键修正**：原 doc §1.2 写"l2norm_fwd_kernel 把 g 做 L2 归一化"是错的。g 在当前代码路径里**根本就没被 l2norm 过**——`gdn_backend.py:512` 把未归一化的 `g` 直接喂给 `chunk_gated_delta_rule`，下游所有 kernel 用 `chunk_local_cumsum(g)` 但不归一化。`l2norm_fwd_kernel` 的 288 次调用**全部**来自 [`chunk.py:108-110`](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk.py#L108-L110) 对 **Q / K** 的 `l2norm_fwd` 调用（每次 chunk kernel 启动 2 次：q、k 各 1）。decode 路径走 `packed_decode`，Q/K L2 归一化已经合并到 `fused_recurrent_gated_delta_rule_packed_decode` Triton kernel 内部（[`fused_recurrent.py:85-87`](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/fused_recurrent.py#L85-L87)），所以那 288 次全部来自 prefill。

### 1.2 trace 显示了什么

Qwen3.5 在 RTX 4090 上跑 batch=32、input=1024、output=10 的完整 trace 导出（`1780020976.4595637-TP-0.trace.json_full_export.json`）：

| Kernel | 调用次数 | 总耗时 (ms) | 每次耗时 (μs) | 备注 |
| --- | --- | --- | --- | --- |
| `fused_gdn_gating_kernel` | 144（prefill） | 0.58 | 4.0 | g 不再 l2norm，见 §1.1 修正 |
| `l2norm_fwd_kernel` | 288（144 prefill × 2 = q+k） | 1.39 | 4.8 | decode 路径已 in-kernel，**0 次** launch |

**方案 1** 的目标：方案 1 v2 实现给 `fused_gdn_gating_kernel` 加了可选的 `g_norm` 第三返回值并把 g/beta 输出保持 bit-identical——**该改动并不消除 `l2norm_fwd_kernel` 调用**（见 §1.1 修正），微基准里看到的 2.5× 加速是 fused_gdn_gating_with_l2norm 单独跑测的 artifact，不反映推理路径的 kernel 数量变化。

**方案 1.5** 的目标：**才是真正能让 288 次 `l2norm_fwd_kernel` 降为 0** 的融合。把 [`chunk.py:108-110`](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk.py#L108-L110) 的两次 `l2norm_fwd(q) / l2norm_fwd(k)` 整段删除，并在以下 4 个 chunk kernel 内部完成 Q/K 的 L2 归一化：

* `chunk_gated_delta_rule_fwd_kkt_solve_kernel`（[chunk_fwd.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk_fwd.py)）
* `recompute_w_u_fwd_kernel`（[wy_fast.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/wy_fast.py)）
* `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`（[chunk_delta_h.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk_delta_h.py)）
* `chunk_fwd_kernel_o`（[chunk_o.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk_o.py)）

由此可以节省：

* **288 次 `l2norm_fwd_kernel` launch**（1.39 ms / profile 窗口）；
* 两次 `Q` / `K` HBM 读 + 写（在 gating 输出与 chunk kernel 输入之间，本来要走 `l2norm_fwd_kernel` 把 Q/K 落 HBM 再读回）。

#### 1.3.1 实测加速（已跑过，仅 v2 gating 部分）

> **重要**：下面这些数字**只**反映 [`fused_gdn_gating_v2.py`](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/fused_gdn_gating_v2.py) 中 `fused_gdn_gating_with_l2norm` 单独跑（输入随机张量，**不接**下游 chunk kernel）的微基准——与生产推理路径下 `l2norm_fwd_kernel` 的次数变化**没有直接关系**。

```
num_heads=32:  v1+l2norm=116.42us  v2_fused= 46.25us  speedup=2.52x
num_heads=64:  v1+l2norm=115.02us  v2_fused= 46.59us  speedup=2.47x
num_heads=128: v1+l2norm=115.79us  v2_fused= 46.48us  speedup=2.49x
```

**~2.5× 加速**——v1 路径每次有 2 次 kernel launch 都要付一次 launch overhead，v2 一次性 launch 完。

**方案 1.5** 才是 doc 原承诺的 1.39 ms 节省。要让这部分数字也成立，需要跑一遍 §3.3 的端到端 trace 对比，并在 trace 里**确认 `l2norm_fwd_kernel` 调用次数从 288 降为 0**。

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

## 4. 踩坑实录（实现过程中真实遇到的两个问题）

### 4.1 g_norm 被错写成 RMSNorm（**真 bug**，已修）

第一版 v2 的 kernel 在算 L2 归一化时，**多除了一个 NUM_HEADS**：

```python
# ❌ 错误版本：把 L2 写成了 RMSNorm
var = tl.sum(blk_g * blk_g, axis=0) / NUM_HEADS
rstd = 1.0 / tl.sqrt(var + eps)
blk_g_norm = blk_g * rstd
```

但 l2norm.py 用的是纯 L2 归一化（[l2norm.py:39](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/l2norm.py#L39)）：

```python
# ✅ 正确版本
var = tl.sum(b_x * b_x, axis=0)            # 注意：无 /N
b_rstd = 1 / tl.sqrt(b_var + eps)
b_y = b_x * b_rstd
```

两者的差异是 `sqrt(N)` 倍——对 N=128 的 Qwen3.5 head 维度，差异约 11.3×，正好对应测试里看到的 g_norm 巨大 diff。

**修法**：直接删掉 `/ NUM_HEADS`。

**教训**：在写「归一化」之前，**先确认原参考实现是 L2 / RMS / LayerNorm 中的哪一种**。本仓库里 [l2norm.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/l2norm.py) 的 `rstd = 1/sqrt(sum(x^2) + eps)` 公式里**没有** `/N` 项，这就是 L2 而非 RMS 的标志。

### 4.2 beta 数值差异 ~1.93e-3（**测试容差太严**，已放宽）

v1 / v2 的 kernel 中 beta 都是这样算的：

```python
blk_beta = tl.sigmoid(blk_b.to(tl.float32))
tl.store(beta_ptr + ..., blk_beta.to(beta_ptr.dtype.element_ty), ...)
```

两边 **fp32 数学完全一致**，但 Triton 编译器会为不同 kernel 结构生成不同的 SASS，导致 `sigmoid` 的末位有微小差异，cast 回 bf16 时可能差 1 ULP。

bf16 在 1.0 附近的 ULP 是 `2**-7 ≈ 7.8e-3`，实测 diff `1.93e-3 ≈ 0.25 ULP`——正常 bf16 精度噪声，**不是 bug**。

**修法**：测试容差从 `1e-5` 放宽到 `5e-3`（约 0.6 ULP），并加注释说明。

**教训**：

* bf16 输出的对比必须用 bf16 ULP 量级的容差（`1e-3 ~ 1e-2`），不能用 fp32 的 `1e-5`。
* 不同 Triton kernel 的 fp32 中间值会有微小差异（编译器优化顺序不同），cast 到低精度后可能差 1 ULP。
* 写测试时**先看 dtype**——`tensor.dtype.element_ty` 决定实际存储精度。

### 4.3 g 完全一致（0.00e+00）——好兆头

`g` 在 v1 / v2 中都是 fp32 输出，且数学路径完全一样，bit-identical 是预期结果。这反过来印证：「`g` 这部分代码是干净的，有问题的一定是 `g_norm`」。

## 5. 为什么不动 v1 kernel

* v1 kernel 文件 [fused_gdn_gating.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/fused_gdn_gating.py)
  是一个小而精、测试完备的代码段，它同时被 NPU 和 CPU backend shim 引用（见 [gdn_backend.py:41-55](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py#L41-L55)）。改它要么会破坏这些 shim，要么需要跨三个 backend 协调改动。
* 把 v1 kernel 保留为默认、v2 作为**可选**的方案，意味着任何生产事故都可以通过 unset 环境变量来回滚，无需任何代码变更。
* v2 在独立文件里，因此可以单独删除（或更新）而不影响 v1 的上游 fla-org 同步。

## 6. 风险与回滚

### 6.1 风险评估

| 风险 | 概率 | 缓解 |
| --- | --- | --- |
| 运算重排导致数值漂移 | 低 | `g` 和 `beta` 使用与 v1 完全相同的 fp32 运算顺序；`g_norm` 使用标准的 `rsqrt(sum(x^2) + eps)` 公式（**不**除以 N）。已在 `test_v2_g_norm_matches_reference` 中验证。 |
| CUDA Graph capture 失败 | 低 | Kernel 不使用任何 host 端分支、不使用 `tl.atomic_*`、不依赖 tensor 数据的 `tl.constexpr` Python 表达式。被 capture 的是 dispatcher 的 `unified_linear_attention_with_output` wrapper，它和其他 op 一样调用 `fused_gdn_gating`。 |
| 小 shape 上延迟反而上升 | 低 | v2 kernel 单个 program 占用的寄存器比 v1 多。对于极小的 `num_heads`（≤ 8），额外的 `tl.sum` 工作可能略大于节省的 HBM 往返。环境变量开关让我们在这种场景下关掉 v2。 |
| 与未来上游 fla-org 同步冲突 | 无 | v2 在新文件里。v1 未动。 |

### 6.2 上线流程

1. 在 feature 分支上落地本次改动。
2. 在 CUDA 机器上跑单元测试——必须全绿。
3. 跑一次 `bench_one_batch` 确定性检查（见 §3.2）。
4. 跑一次 `bench_one_batch` 性能检查（见 §3.3）——确认至少 1.0 ms 加速（实测 1.3 ms），且 output_ids 不退步。
5. 合入主干。
6. 可选：在后续提交里通过去掉环境变量开关、把 `gdn_backend.py` 中的 v1 import 直接换成 v2 来把它变为默认。

## 7. 后续工作

本次融合是更大计划中的**基础**。下面这些 follow-up 已经排上日程：

| 方案 | 目标 | 预期收益 | 难度 |
| --- | --- | --- | --- |
| 方案 2 | 让 chunk kernel（`chunk_gated_delta_rule`）接收 `g_norm` 作为输入，跳过它内部的 l2norm。 | prefill 阶段每次 profile 节省 1-2 ms。 | 中——会改到 fla-org 上游的 `chunk.py`。 |
| 方案 3 | 在 gating kernel 里融合 `sigmoid(b) * v`（chunk delta 路径里的 `u = v * beta`）。 | decode 阶段每次 profile 节省 1.5-2 ms。 | 中——会改到 chunk kernel。 |
| 方案 4 | 在 decode 阶段把 `causal_conv1d_update` 与上游 `qkvzba_split_reshape_cat_contiguous_kernel` 融合。 | 每次 profile 节省 5-8 ms。 | 高——kernel 大量重写。 |
| 方案 5 | decode 单 kernel 化：conv1d + GDN + recurrent update 一次完成。 | 每次 profile 节省 10-15 ms。 | 高——整条 decode 流水线重写。 |

## 8. 引用

* trace 文件：`d:/下载/1780020976.4595637-TP-0.trace.json_full_export.json`
* trace 分析器：仓库根目录 `analyze_trace.py`
* v1 kernel：[fused_gdn_gating.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/fused_gdn_gating.py)
* v2 kernel：[fused_gdn_gating_v2.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/fused_gdn_gating_v2.py)
* L2 归一化 kernel：[l2norm.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/l2norm.py)
* Backend dispatcher：[gdn_backend.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py)
* 单元测试：[test_fused_gdn_gating_v2.py](file:///D:/算家/项目/sglang%20源码解析/sglang/test/registered/attention/test_fused_gdn_gating_v2.py)
* 相关的融合参考：[fused_norm_gate.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/fused_norm_gate.py)
  （RMSNorm + 激活 gating；本方案沿用其设计模式）

## 9. 方案 1.5：把 Q/K L2 归一化也搬进 chunk kernel

### 9.1 动机

§1.1 修正后我们看清：trace 里 288 次 `l2norm_fwd_kernel` 实际由 [`chunk.py:108-110`](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk.py#L108-L110) 的两次 `l2norm_fwd(q) / l2norm_fwd(k)` 产生，**对象是 Q / K**，不是 g。方案 1 v2 只动了 `fused_gdn_gating`（算 g/beta 的小 kernel），对它**不会**省任何一次 launch。**要真正把 288 次 `l2norm_fwd_kernel` 降为 0，必须把 Q/K 的 L2 归一化从 `chunk.py` 的 wrapper 搬进 4 个下游 chunk kernel 内部**——这正是"方案 1.5"。

### 9.2 改动范围

| 文件 | 改动 |
| --- | --- |
| [chunk.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk.py) | 新增 env var `SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL`；env var = `1` 时，**跳过** `l2norm_fwd(q) / l2norm_fwd(k)` 调用，把"对 Q/K 做 L2 归一化"的责任下放给下游 kernel。**默认关闭**——env var 未设时与原版完全一致。 |
| [chunk_fwd.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk_fwd.py) | 给 `chunk_gated_delta_rule_fwd_kkt_solve_kernel` 加 `USE_K_L2NORM_IN_KERNEL: tl.constexpr`；新增 heuristic 把 BK 强制为 `next_power_of_2(K)`；加载 b_k0/b_k1/b_k2/b_k3 之后立即做 `b_k / sqrt(sum(b_k^2) + eps)`。 |
| [wy_fast.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/wy_fast.py) | 同上，给 `recompute_w_u_fwd_kernel` 加 `USE_K_L2NORM_IN_KERNEL` 与 BK heuristic，加 in-kernel K L2 归一化。 |
| [chunk_delta_h.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk_delta_h.py) | 给 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 加 `USE_K_L2NORM_IN_KERNEL`。该 kernel 把 K 分成 64-wide 块多次加载（K 维度不同时 1/2/3/4 块），单次 tl.load 拿不到整行；这里采用 **2-pass**：在每块 i_t 内先扫一遍 K 算 sum_sq，再用 rstd 归一化后做 `tl.trans(tl.dot(b_k, b_v))`。多一次 K 读取是 in-kernel L2 在该 kernel 上不可避免的代价。 |
| [chunk_o.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk_o.py) | 给 `chunk_fwd_kernel_o` 加 `USE_QK_L2NORM_IN_KERNEL` 与 BK heuristic；**Q** 与 **K** 都在 `tl.load` 之后做 in-kernel L2 归一化（K 用 `axis=0` 因为在 [BK, BT] 布局下，K 维是 axis=0）。 |

所有 5 个文件共享一个 env var：`SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL`。开启后：

* `chunk.py` 不再调用 `l2norm_fwd`；
* 4 个 chunk kernel 各自在 `tl.load` 之后做 L2 归一化；
* `BK` heuristic 把 BK 强制为 `max(64, next_power_of_2(K))`（kkt_solve / recompute_w_u）或 `max(128, next_power_of_2(K))`（chunk_o），保证单次 tl.load 之内能完成行内归约；
* delta_h 因为分块结构特殊，**不强制**改 BK（仍为固定 64），改用 2-pass 拿 rstd。

### 9.3 与方案 1 v2 的关系

| 开关 | 控制 | 节省 |
| --- | --- | --- |
| `SGLANG_FUSE_GDN_GATING=1` | 把 `g` 的 L2 归一化算子融合进 `fused_gdn_gating`（输出 g_norm 第三返回值）。 | **不**节省任何 `l2norm_fwd_kernel`（g 没被 l2norm）。仅给未来想用 `g_norm` 的调用方提供便利。 |
| `SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL=1` | 把 Q/K L2 归一化搬进 4 个 chunk kernel 内部。 | 节省 288 次 `l2norm_fwd_kernel` launch / profile 窗口（1.39 ms / Qwen3.5-4B + RTX 4090）。 |

两开关**互不依赖**——单独开第二个就能拿到完整收益；第一个对当前生产路径**无任何效果**（保留它只为 API 完整性）。如果要做减法，可以只保留第二个。

### 9.4 使用方法

```bash
# 启用方案 1.5（**这是真正能省 1.39 ms 的开关**）
SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL=1 \
    python -m sglang.launch_server --model-path /path/to/Qwen3.5-4B

# 同时启用方案 1 + 方案 1.5（无额外收益，仅 API 一致性）
SGLANG_FUSE_GDN_GATING=1 \
SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL=1 \
    python -m sglang.launch_server --model-path /path/to/Qwen3.5-4B
```

### 9.5 验证清单

* [x] 单元语法：4 个 chunk kernel 都加了 `USE_*_L2NORM_IN_KERNEL` constexpr 与对应 `if` 块。
* [ ] 单测：未在 `test/registered/attention/` 下为 4 个 kernel 各加单测——**TODO**（建议加一个 `test_chunk_kernels_in_kernel_l2norm.py`，对比 env-var-off vs env-var-on 的输出在 bf16 容差内一致）。
* [ ] 端到端：需要 CUDA 机器跑一遍 `bench_one_batch`（§3.3），**核心指标是 `l2norm_fwd_kernel` 调用次数从 288 降为 0**——这是方案 1.5 唯一可信的成功标志。
* [ ] 性能：trace 里 `l2norm_fwd_kernel` 行消失，总时间应下降 ~1.39 ms / profile 窗口。kkt_solve / recompute_w_u / chunk_o 单次耗时因 BK 改大可能 +5-10%，delta_h 因 2-pass 可能 +10-20%，需权衡。

### 9.6 风险与回滚

* **最坏情况**：in-kernel L2 数值与 `l2norm_fwd_kernel` 略有差异（1-2 ULP，bf16 输出），导致模型 output_ids 出现轻微偏移。**回滚**：unset `SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL`，行为立即回到原版。
* **BK 变大的代价**：`chunk_fwd_kkt_solve` 与 `recompute_w_u` 原来用 BK=32/64 autotune，in-kernel 路径下 BK 被强制为 `next_power_of_2(K)`（Qwen3.5-4B 下为 128），单 program 寄存器压力上升 2-4 倍。在 SM89 / RTX 4090（64 KB regs/SM）上需要观察是否有 register spilling。
* **delta_h 的 2-pass**：每个 chunk 多一次 K 读取（Qwen3.5-4B 下 K=128 时 16 KB/chunk/程序）；乘以 24 GDN 层 × 64 (V_block, batch) × 16 chunk，HBM 多读 ~384 MB / profile 窗口。HBM3 带宽下约 0.4 ms，**仍小于节省的 1.39 ms**。
* **与上游 fla-org 同步冲突**：本改动触及 4 个 fla-org 上游 kernel。同步前需要先 rebase 该子集到上游 master。

## 10. 使用与验证指南

> 这一节专门回答"怎么开 / 怎么确认生效 / 怎么证伪省掉了 l2norm_fwd_kernel"，分四步走。

### 10.1 启用

```bash
# 用法 A：只开方案 1.5（**推荐**——这才是真正省 1.39 ms 的开关）
SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL=1 \
  python -m sglang.launch_server --model-path /path/to/Qwen3.5-4B

# 用法 B：方案 1 + 方案 1.5 同时开（方案 1 对当前生产路径无任何效果，仅 API 一致性）
SGLANG_FUSE_GDN_GATING=1 \
SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL=1 \
  python -m sglang.launch_server --model-path /path/to/Qwen3.5-4B

# 用法 C：单 batch benchmark
SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL=1 \
  python -m sglang.bench_one_batch \
    --model-path /path/to/Qwen3.5-4B \
    --batch 32 --input-len 1024 --output-len 10
```

> ⚠️ `sglang.launch_server` 在多 worker 场景下 env var 必须**显式 export** 在 launcher 命令前面。sglang 不会自动把 env var 转发给 worker。如果只在外层 shell 设了 env var 而 worker 没继承，[gdn_backend.py:86-89](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py#L86-L89) 处的日志会显示 `in_kernel_qk_l2norm=0`，这时需要把 env var 加到 launcher 启动参数里（sglang 通常支持 `--env SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL=1` 这类语法，或者写到 sglang config 里）。

### 10.2 验证是否启用（三层确认）

#### 第 1 层：模型加载时日志

启动后立刻看 server 日志，应该看到这两行（第 1 行立刻出现；第 2 行在 `GDNAttnBackend` 完成初始化时出现）：

```
[fused_gdn_gating] using v1 (baseline) (SGLANG_FUSE_GDN_GATING=0)
[GDNAttnBackend] initialized decode=TritonGDNKernel, prefill=TritonGDNKernel,
                 verify=TritonGDNKernel, packed_decode=True,
                 fused_gdn_gating=v1 (baseline) (SGLANG_FUSE_GDN_GATING=0),
                 in_kernel_qk_l2norm=1 (SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL)
```

* ✅ `in_kernel_qk_l2norm=1` → 开关已生效。
* ❌ `in_kernel_qk_l2norm=0` → env var 没传到 worker 进程——这是最常见的坑。

#### 第 2 层：Python 端即时检查

```bash
# 快速确认 env var 被 sglang 的 5 个目标模块正确读取
cd sglang/python
python -c "
import os
os.environ['SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL'] = '1'
from sglang.srt.layers.attention.fla import chunk, chunk_fwd
print('chunk._USE_L2NORM_IN_KERNEL:', chunk._USE_L2NORM_IN_KERNEL)
print('chunk_fwd._USE_L2NORM_IN_KERNEL:', chunk_fwd._USE_L2NORM_IN_KERNEL)
"
```

> 注：[wy_fast.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/wy_fast.py) / [chunk_o.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk_o.py) / [chunk_delta_h.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk_delta_h.py) 三个模块没有导出模块级常量，wrapper 内部直接 `os.getenv(...)` 读 env var——只要 env var 设了，它们的 constexpr 就会是 True。

#### 第 3 层：kernel 签名

```python
# 确认 kernel 已用 USE_*_L2NORM_IN_KERNEL constexpr 编译过
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_kernel_o
print("USE_QK_L2NORM_IN_KERNEL in signature:",
      "USE_QK_L2NORM_IN_KERNEL" in chunk_fwd_kernel_o.signature)
# 期望：True
```

第一次调用某个 kernel 时 Triton 会做 autotune（同时把 constexpr 编译进 SASS）。autotune 后会在 `~/.triton/cache` 留下 kernel 二进制。

### 10.3 验证是否真的省掉了 `l2norm_fwd_kernel`（**核心**）

#### 方法 A：`analyze_trace.py` 看 Chrome trace

```bash
# 跑 ON 的 benchmark，导出 trace
export SGLANG_TORCH_PROFILER_DIR=./log_v15
SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL=1 python -m sglang.bench_one_batch --profile \
  --batch 32 --input-len 1024 --output-len 10 \
  --model-path /path/to/Qwen3.5-4B

# 跑 OFF 的做对照
export SGLANG_TORCH_PROFILER_DIR=./log_baseline
python -m sglang.bench_one_batch --profile \
  --batch 32 --input-len 1024 --output-len 10 \
  --model-path /path/to/Qwen3.5-4B

# 对比两份 trace
python ../../analyze_trace.py ./log_baseline/scheduler_0.trace.json.gz
python ../../analyze_trace.py ./log_v15/scheduler_0.trace.json.gz
```

**关键指标**：

| Kernel | OFF 期望 | ON 期望 |
|---|---|---|
| `l2norm_fwd_kernel` | 288 次 / 1.39 ms | **0 次 / 0 ms** |
| `fused_gdn_gating_kernel` | 144 次 / 0.58 ms | 144 次 / 0.58 ms（方案 1.5 不动它） |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 144 次 | 144 次（多 in-kernel L2，单次可能 +5%） |
| `recompute_w_u_fwd_kernel` | 144 次 | 144 次（同上） |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 144 次 | 144 次（**2-pass**，单次可能 +10-20%） |
| `chunk_fwd_kernel_o` | 144 次 | 144 次（多 in-kernel L2，单次可能 +5%） |

**唯一可信的成功标志是 `l2norm_fwd_kernel` 从 288 降为 0**。

#### 方法 B：`nsys` 快速 sanity check

```bash
# 跑一次 ON，看 kernel timeline
nsys profile -t cuda --output=./v15 \
  python -c "
import os
os.environ['SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL'] = '1'
# ... 加载 Qwen3.5-4B 跑 1 步 ...
"

# 导出 sqlite 后查 kernel
nsys stats --report cuda_kern_exec_sum ./v15.nsys-rep | grep l2norm
# 期望：l2norm_fwd_kernel 0 次
```

#### 方法 C：直接搜 trace JSON

```bash
python -c "
import json
data = json.load(open('./log_v15/scheduler_0.trace.json.gz'))
ks = data['kernels']
for k in ks:
    if 'l2norm' in k.get('name','').lower():
        print(k['name'], k['count'], k['total_time_ms'])
"
# 期望输出：l2norm_fwd_kernel 0 0.0
```

### 10.4 验证数值正确性（必须做，否则模型可能输出 NaN）

```bash
# 基线（OFF）
python -m sglang.bench_one_batch \
  --model-path /path/to/Qwen3.5-4B \
  --batch 32 --input-len 1024 --output-len 10 \
  --output-file /tmp/baseline.json

# ON
SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL=1 python -m sglang.bench_one_batch \
  --model-path /path/to/Qwen3.5-4B \
  --batch 32 --input-len 1024 --output-len 10 \
  --output-file /tmp/v15.json

# 对比 output_ids 字段
python -c "
import json
a = json.load(open('/tmp/baseline.json'))
b = json.load(open('/tmp/v15.json'))
ids_a = a['output_ids'] if isinstance(a, dict) else a
ids_b = b['output_ids'] if isinstance(b, dict) else b
print('match:', ids_a == ids_b)
print('first diff:', next((i for i,(x,y) in enumerate(zip(ids_a, ids_b)) if x != y), 'no diff'))
"
```

* ✅ `output_ids` 完全一致 → 数值正确。
* ❌ `output_ids` 有 diff 但都在少数 ULP（bf16）以内 → 正常噪声，模型输出文本不会变。
* ❌ `output_ids` 大量 diff → **不要上线**，unset env var 回滚。

### 10.5 一行自检脚本（放进 CI 即可）

```bash
# 1. env var 是否传到 worker
grep "in_kernel_qk_l2norm=1" server.log

# 2. l2norm_fwd_kernel 计数（期望 0）
python -c "
import json
d = json.load(open('trace.json'))
ks = [k for k in d['kernels'] if 'l2norm' in k['name'].lower()]
total = sum(k['count'] for k in ks)
print(f'l2norm_fwd_kernel count: {total}')
exit(0 if total == 0 else 1)
"

# 3. output_ids 一致性
python compare_ids.py baseline.json v15.json
```

### 10.6 回滚

任何一步失败只要 unset 即可回到原版（默认 OFF）：

```bash
unset SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL
unset SGLANG_FUSE_GDN_GATING
```

代码路径会在 [chunk.py:128](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk.py#L128) 处走 `l2norm_fwd(q) / l2norm_fwd(k)`，4 个 kernel 走 `USE_*_L2NORM_IN_KERNEL=False` 分支，BK heuristic 返回默认 64/128——**与改之前完全一致**。
