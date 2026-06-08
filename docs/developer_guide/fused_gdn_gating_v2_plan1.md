# 方案 1：在 chunk kernel 内做 Q/K L2 归一化

| 字段 | 值 |
| --- | --- |
| 状态 | 已实现（由 `SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL` 控制） |
| 适用模型 | Qwen3.5、KDA 以及任何带 GDN（Gated Delta Network）线性注意力层的 hybrid 模型 |
| 影响的 kernel 路径 | `chunk.py` → `l2norm_fwd` 调用（消除）→ 4 个 chunk kernel 内部 |
| 默认行为 | **关闭** —— 设置 `SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL=1` 启用 |

## 1. 动机

### 1.1 背景：什么是 Q/K L2 归一化

Qwen3.5（以及其他「hybrid」线性注意力模型）的线性注意力 decoder 层使用 Gated Delta Network（GDN）。prefill 阶段 chunk 路径会调 [`chunk.py:108-110`](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk.py#L108-L110)：

```python
if use_qk_l2norm_in_kernel:
    q = l2norm_fwd(q)   # ← 触发 1 次 l2norm_fwd_kernel
    k = l2norm_fwd(k)   # ← 触发 1 次 l2norm_fwd_kernel
```

把 Q / K 做 L2 归一化后，再喂给 4 个 chunk kernel（`chunk_gated_delta_rule_fwd_kkt_solve_kernel` / `recompute_w_u_fwd_kernel` / `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` / `chunk_fwd_kernel_o`）。decode 路径走 `packed_decode`，Q/K L2 归一化已经合并到 `fused_recurrent_gated_delta_rule_packed_decode` Triton kernel 内部（[`fused_recurrent.py:85-87`](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/fused_recurrent.py#L85-L87)），所以不 launch 额外的 kernel。

### 1.2 trace 显示了什么

Qwen3.5 在 RTX 4090 上跑 batch=32、input=1024、output=10 的完整 trace 导出（`d:/下载/1780020976.4595637-TP-0.trace.json_full_export.json`）：

| Kernel | 调用次数 | 总耗时 (ms) | 每次耗时 (μs) | 备注 |
| --- | --- | --- | --- | --- |
| `fused_gdn_gating_kernel` | 144（prefill） | 0.58 | 4.0 | 算 g、beta |
| `l2norm_fwd_kernel` | 288（144 prefill × 2 = q+k） | 1.39 | 4.8 | **本方案要消除的就是这个** |

**本方案的目标**：把 288 次 `l2norm_fwd_kernel` launch 降为 0，省 1.39 ms / profile 窗口；省 2 次 Q / K HBM 读 + 写（本来要走 `l2norm_fwd_kernel` 把 Q/K 落 HBM 再读回）。

## 2. 改了什么

5 个文件改动，都藏在 env var 后面，默认关闭，线上出问题 unset 即可回滚。

| 文件 | 改动 |
| --- | --- |
| [chunk.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk.py) | `ChunkGatedDeltaRuleFunction.forward` 里把 `l2norm_fwd(q) / l2norm_fwd(k)` 包了一层 `if not _USE_L2NORM_IN_KERNEL`。同时计算 `do_in_kernel_l2norm = use_qk_l2norm_in_kernel and _USE_L2NORM_IN_KERNEL` 沿调用链传下去。env var 在这里**唯一**生效一次。 |
| [chunk_fwd.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk_fwd.py) | `chunk_gated_delta_rule_fwd_intra` 加 `do_in_kernel_l2norm` 入参，传给 `chunk_gated_delta_rule_fwd_kkt_solve_kernel` 与 `recompute_w_u_fwd`。kkt_solve kernel 加 `USE_K_L2NORM_IN_KERNEL: tl.constexpr` 与 BK heuristic；加载 b_k0/b_k1/b_k2/b_k3 后立即 `b_k / sqrt(sum(b_k^2) + eps)`。 |
| [wy_fast.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/wy_fast.py) | `recompute_w_u_fwd` 加 `do_in_kernel_l2norm` 入参；kernel 加 `USE_K_L2NORM_IN_KERNEL` 与 BK heuristic，加 in-kernel K L2 归一化。 |
| [chunk_delta_h.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk_delta_h.py) | `chunk_gated_delta_rule_fwd_h` 加 `do_in_kernel_l2norm` 入参；kernel 加 `USE_K_L2NORM_IN_KERNEL`。该 kernel 把 K 分成 64-wide 块多次加载（K 维度不同时 1/2/3/4 块），单次 tl.load 拿不到整行；这里采用 **2-pass**：在每块 i_t 内先扫一遍 K 算 sum_sq，再用 rstd 归一化后做 `tl.trans(tl.dot(b_k, b_v))`。多一次 K 读取是 in-kernel L2 在该 kernel 上不可避免的代价。 |
| [chunk_o.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk_o.py) | `chunk_fwd_o` 加 `do_in_kernel_l2norm` 入参；`chunk_fwd_kernel_o` 加 `USE_QK_L2NORM_IN_KERNEL` 与 BK heuristic；**Q** 与 **K** 都在 `tl.load` 之后做 in-kernel L2 归一化（K 在 [BK, BT] 布局下 axis=0 是 K 维）。 |

### 2.1 设计要点

* **env var 只在 `chunk.py` 决定"wrapper 做还是 in-kernel 做"**——`chunk.py` 是 source of truth。4 个 kernel wrapper 接受 `do_in_kernel_l2norm` 参数，**不**读 env var。这样保证 env var 不会"越权"让 caller 明确说不要 l2norm 的模型被强制注入归一化。
* **`BK` heuristic 强制为 `next_power_of_2(K)`**——保证单次 `tl.load` 之内能完成行内归约。对 Qwen3.5-4B（K=128）来说 BK=128，原 BK=32/64 的 autotune 会被这条路径跳过。
* **数值精度**：所有 in-kernel L2 都用 `b_k.to(tl.float32)` 转 fp32 算 rstd，再 cast 回原 dtype，**与 `l2norm_fwd_kernel` 完全一致**。
* **eps 一致**：`+ 1e-6` 跟 [`l2norm.py:39`](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/l2norm.py#L39) 一致。

### 2.2 Kernel 伪代码（chunk_o / kkt_solve / recompute_w_u 通用模式）

```python
@triton.jit
def kernel(b_k_ptr, ...):
    b_k = tl.load(p_k, boundary_check=(0, 1))     # shape [BC, K]
    if USE_K_L2NORM_IN_KERNEL:                     # caller + env var 合并后为 True
        b_k_f32 = b_k.to(tl.float32)
        b_k = (b_k_f32
               / tl.sqrt(tl.sum(b_k_f32 * b_k_f32,
                                axis=K_AXIS, keep_dims=True) + 1e-6)
              ).to(b_k.dtype)
    # 用归一化后的 b_k 继续做 dot product ...
```

delta_h 因为 K 是 64-wide 块多次加载，所以走 2-pass：

```python
if USE_K_L2NORM_IN_KERNEL:
    # Pass 1: 扫一遍 K 算 rstd
    b_k1 = tl.load(p_k1, ...)
    b_k_sq = tl.sum(b_k1 * b_k1, axis=0)
    if K > 64:
        b_k2 = tl.load(p_k2, ...)
        b_k_sq += tl.sum(b_k2 * b_k2, axis=0)
    b_k_rstd = 1.0 / tl.sqrt(b_k_sq + 1e-6)        # [BT]

# Pass 2: 加载 K、归一化、使用
b_k1 = tl.load(p_k1, ...)
b_k1 = (b_k1.to(tl.float32) * b_k_rstd[None, :]).to(b_k1.dtype)
b_h1 += tl.trans(tl.dot(b_k1, b_v))
# ...
```

## 3. 使用与验证指南

### 3.1 启用

```bash
# 推荐：只开这一个开关
SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL=1 \
  python -m sglang.launch_server --model-path /path/to/Qwen3.5-4B

# 单 batch benchmark
SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL=1 \
  python -m sglang.bench_one_batch \
    --model-path /path/to/Qwen3.5-4B \
    --batch 32 --input-len 1024 --output-len 10
```

> ⚠️ `sglang.launch_server` 在多 worker 场景下 env var 必须**显式 export** 在 launcher 命令前面。如果只在外层 shell 设了 env var 而 worker 没继承，[gdn_backend.py:298-308](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py#L298-L308) 处的日志会显示 `in_kernel_qk_l2norm=0`，这时需要把 env var 加到 launcher 启动参数里。

### 3.2 验证是否启用（三层确认）

#### 第 1 层：模型加载时日志

启动后立刻看 server 日志，应该看到这一行（`GDNAttnBackend` 完成初始化时出现）：

```
[GDNAttnBackend] initialized decode=TritonGDNKernel, prefill=TritonGDNKernel,
                 verify=TritonGDNKernel, packed_decode=True,
                 in_kernel_qk_l2norm=1 (SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL)
```

* ✅ `in_kernel_qk_l2norm=1` → 开关已生效。
* ❌ `in_kernel_qk_l2norm=0` → env var 没传到 worker 进程——这是最常见的坑。

#### 第 2 层：Python 端即时检查

```bash
# 快速确认 env var 被 chunk.py 正确读取
cd sglang/python
python -c "
import os
os.environ['SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL'] = '1'
from sglang.srt.layers.attention.fla import chunk
print('chunk._USE_L2NORM_IN_KERNEL:', chunk._USE_L2NORM_IN_KERNEL)
"
# 期望：chunk._USE_L2NORM_IN_KERNEL: True
```

#### 第 3 层：kernel 签名

```python
# 确认 kernel 已用 USE_*_L2NORM_IN_KERNEL constexpr 编译过
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_kernel_o
print("USE_QK_L2NORM_IN_KERNEL in signature:",
      "USE_QK_L2NORM_IN_KERNEL" in chunk_fwd_kernel_o.signature)
# 期望：True
```

### 3.3 验证是否真的省掉了 `l2norm_fwd_kernel`（**核心**）

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
| `fused_gdn_gating_kernel` | 144 次 / 0.58 ms | 144 次 / 0.58 ms（方案 1 不动它） |
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

### 3.4 验证数值正确性（必须做，否则模型可能输出 NaN）

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

### 3.5 一行自检脚本（放进 CI 即可）

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

### 3.6 回滚

任何一步失败只要 unset 即可回到原版（默认 OFF）：

```bash
unset SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL
```

代码路径会在 [`chunk.py:128`](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk.py#L128) 处走 `l2norm_fwd(q) / l2norm_fwd(k)`，4 个 kernel 走 `USE_*_L2NORM_IN_KERNEL=False` 分支，BK heuristic 返回默认 64/128——**与改之前完全一致**。

## 4. 验证清单

* [x] 单元语法：4 个 chunk kernel 都加了 `USE_*_L2NORM_IN_KERNEL` constexpr 与对应 `if` 块。
* [ ] 单测：未在 `test/registered/attention/` 下为 4 个 kernel 各加单测——**TODO**（建议加一个 `test_chunk_kernels_in_kernel_l2norm.py`，对比 env-var-off vs env-var-on 的输出在 bf16 容差内一致）。
* [ ] 端到端：需要 CUDA 机器跑一遍 `bench_one_batch`（§3.3），**核心指标是 `l2norm_fwd_kernel` 调用次数从 288 降为 0**——这是方案 1 唯一可信的成功标志。
* [ ] 性能：trace 里 `l2norm_fwd_kernel` 行消失，总时间应下降 ~1.39 ms / profile 窗口。kkt_solve / recompute_w_u / chunk_o 单次耗时因 BK 改大可能 +5-10%，delta_h 因 2-pass 可能 +10-20%，需权衡。

## 5. 风险与回滚

* **最坏情况**：in-kernel L2 数值与 `l2norm_fwd_kernel` 略有差异（1-2 ULP，bf16 输出），导致模型 output_ids 出现轻微偏移。**回滚**：unset `SGLANG_FUSE_L2NORM_INTO_CHUNK_KERNEL`，行为立即回到原版。
* **BK 变大的代价**：`chunk_fwd_kkt_solve` 与 `recompute_w_u` 原来用 BK=32/64 autotune，in-kernel 路径下 BK 被强制为 `next_power_of_2(K)`（Qwen3.5-4B 下为 128），单 program 寄存器压力上升 2-4 倍。在 SM89 / RTX 4090（64 KB regs/SM）上需要观察是否有 register spilling。
* **delta_h 的 2-pass**：每个 chunk 多一次 K 读取（Qwen3.5-4B 下 K=128 时 16 KB/chunk/程序）；乘以 24 GDN 层 × 64 (V_block, batch) × 16 chunk，HBM 多读 ~384 MB / profile 窗口。HBM3 带宽下约 0.4 ms，**仍小于节省的 1.39 ms**。
* **与上游 fla-org 同步冲突**：本改动触及 4 个 fla-org 上游 kernel。同步前需要先 rebase 该子集到上游 master。

## 6. 引用

* trace 文件：`d:/下载/1780020976.4595637-TP-0.trace.json_full_export.json`
* trace 分析器：仓库根目录 `analyze_trace.py`
* L2 归一化 kernel：[l2norm.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/l2norm.py)
* chunk.py（dispatcher）：[chunk.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk.py)
* 4 个被改的 Triton kernel：
  * [chunk_fwd.py:chunk_gated_delta_rule_fwd_kkt_solve_kernel](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk_fwd.py)
  * [wy_fast.py:recompute_w_u_fwd_kernel](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/wy_fast.py)
  * [chunk_delta_h.py:chunk_gated_delta_rule_fwd_kernel_h_blockdim64](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk_delta_h.py)
  * [chunk_o.py:chunk_fwd_kernel_o](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/chunk_o.py)
* Backend dispatcher：[gdn_backend.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py)
* 相关的融合参考：[fused_norm_gate.py](file:///D:/算家/项目/sglang%20源码解析/sglang/python/sglang/srt/layers/attention/fla/fused_norm_gate.py)
  （RMSNorm + 激活 gating；本方案沿用其设计模式）
