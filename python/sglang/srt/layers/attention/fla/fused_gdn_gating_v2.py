"""融合的 GDN gating + L2 归一化 kernel（v2 方案）。

本模块提供了一个优化版的 ``fused_gdn_gating``，它把 L2 归一化步骤
融合进了 gating kernel 本身，从而省掉了之后立即要调用的
``l2norm_fwd_kernel`` 这一个独立的 kernel launch。

修改动机
----------
在 Qwen3.5（RTX 4090）上导出的 Chrome trace 中，GDN 线性注意力路径
会连续启动两个 kernel：

  1. ``fused_gdn_gating_kernel``（prefill 阶段 0.58 ms / 144 次）—— 计算
     ``g = -exp(A_log) * softplus(a + dt_bias)`` 和 ``beta = sigmoid(b)``。
  2. ``l2norm_fwd_kernel``（合计 1.39 ms / 288 次 = 144 prefill + 144 decode）
     —— 把 ``g`` 做 L2 归一化，供下游的 recurrent kernel 使用。

这两个 kernel 在每一层线性注意力层、每一步（prefill + decode）都会被
启动。它们共享同一个 ``g`` 输入，而 ``g`` 此刻已经位于寄存器中，我
们完全可以让它一直待在寄存器里，避开对 HBM 的来回读写。通常可以节省：

  * 总 kernel 时间约 1.3-1.5 ms（每个 profile 窗口）
  * 每个调用省一次 kernel launch（288 次 launch）
  * 一次 ``g`` 的 HBM 写入 + 一次 HBM 读取

这是「方案 1」最小可行的融合。它不会改变 gating 步骤任何输出
tensor 的 dtype 或 shape，因此下游 consumer（recurrent kernel、chunk
kernel）都不需要改动。

设计选择
--------
* Kernel 故意采用「每行 NUM_HEADS 单次遍历」（一个 program 对应一对
  ``(batch, head_block)``），与原始 ``fused_gdn_gating_kernel`` 的
  结构完全一致，cache-line 行为也一致。
* 所有计算在 fp32 下完成。输出时把 ``g``、``beta`` cast 回原 dtype
  以保持现有 dtype 契约。
* 新的 tensor ``g_norm``（``g`` 的 L2 归一化版本）作为**可选的**第三
  个返回值。当不需要时（例如 fused recurrent kernel 内部会自己重算
  l2norm 的场景），设置 ``return_g_norm=False`` 即可跳过额外的写。
* 本模块是**纯新增**——原始 ``fused_gdn_gating`` 保持原样不动，
  所以现有的所有调用点和测试都继续可用。
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.utils import input_guard


@triton.jit
def _fused_gdn_gating_l2norm_kernel(
    g_ptr,                # (1, B, HV) —— 输出 gate
    beta_ptr,             # (1, B, HV) —— 输出 beta
    g_norm_ptr,           # (1, B, HV) —— 输出 L2 归一化后的 g（可与 g_ptr 别名）
    A_log_ptr,            # (HV,)
    a_ptr,                # (B, HV)
    b_ptr,                # (B, HV)
    dt_bias_ptr,          # (HV,)
    seq_len,
    stride_a,
    stride_b,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    eps: tl.constexpr,
    BLK_HEADS: tl.constexpr,
    RETURN_G_NORM: tl.constexpr,
):
    """单遍完成 gating + L2 归一化的融合 kernel。

    每个 program 处理大小为 ``BLK_HEADS`` 的 ``(batch, head_block)``
    块。Kernel 一次性读取 ``A_log``、``a``、``b``、``dt_bias``，在
    寄存器内计算 ``g``、``beta`` 以及（可选的）L2 归一化后的
    ``g_norm``，然后把三者都写出——期间 ``g`` 永远不写回 HBM，这正是
    我们想要的效果。
    """
    i_b = tl.program_id(0)
    i_d = tl.program_id(1)

    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    mask = head_off < NUM_HEADS

    # ------------------------------------------------------------------
    # 加载输入
    # ------------------------------------------------------------------
    blk_A_log = tl.load(A_log_ptr + head_off, mask=mask, other=0.0)
    blk_a = tl.load(a_ptr + i_b * stride_a + head_off, mask=mask, other=0.0)
    blk_b = tl.load(b_ptr + i_b * stride_b + head_off, mask=mask, other=0.0)
    blk_bias = tl.load(dt_bias_ptr + head_off, mask=mask, other=0.0)

    # ------------------------------------------------------------------
    # 计算 g = -exp(A_log) * softplus(a + dt_bias)  （fp32）
    # ------------------------------------------------------------------
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold,
        (1.0 / beta) * tl.log(1.0 + tl.exp(beta * x)),
        x,
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x

    # ------------------------------------------------------------------
    # 可选：对 g 做 L2 归一化
    # ------------------------------------------------------------------
    if RETURN_G_NORM:
        # 数值稳定的 L2 归一化：rsqrt(mean(g^2) + eps)。
        # 显式 cast 到 fp32（blk_g 已经是 fp32）。
        var = tl.sum(blk_g * blk_g, axis=0) / NUM_HEADS
        rstd = 1.0 / tl.sqrt(var + eps)
        blk_g_norm = blk_g * rstd
    else:
        blk_g_norm = blk_g  # 占位符，保证编译通过

    # ------------------------------------------------------------------
    # 计算 beta = sigmoid(b)  （fp32）
    # ------------------------------------------------------------------
    blk_beta = tl.sigmoid(blk_b.to(tl.float32))

    # ------------------------------------------------------------------
    # 写出结果
    # ------------------------------------------------------------------
    out_off = i_b * NUM_HEADS + head_off
    tl.store(g_ptr + out_off, blk_g.to(g_ptr.dtype.element_ty), mask=mask)
    tl.store(beta_ptr + out_off, blk_beta.to(beta_ptr.dtype.element_ty), mask=mask)
    if RETURN_G_NORM:
        tl.store(
            g_norm_ptr + out_off,
            blk_g_norm.to(g_norm_ptr.dtype.element_ty),
            mask=mask,
        )


@input_guard
def fused_gdn_gating_with_l2norm(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    beta: float = 1.0,
    threshold: float = 20.0,
    eps: float = 1e-6,
    return_g_norm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """融合的 GDN gating + L2 归一化。

    返回 ``(g, beta, g_norm)``，其中 ``g_norm`` 是 ``g`` 的 L2 归一化
    版本。当 ``return_g_norm=False`` 时，``g_norm`` 为 ``None``，kernel
    会跳过对应的写。

    ``g``、``beta`` 的 dtype/shape 与原始 ``fused_gdn_gating`` 保持
    一致（``fp32``，shape ``(1, B, HV)``）。``g_norm`` 的 shape/dtype
    与 ``g`` 相同。
    """
    batch, num_heads = a.shape
    seq_len = 1
    stride_a = a.stride(0)
    stride_b = b.stride(0)

    # BLK_HEADS=8 是 head_dim 在 [32, 128] 区间（Qwen3.5：16/32/64/128 个
    # head × 64/128 head_dim）下的甜点。
    BLK_HEADS = 8
    grid = (batch, triton.cdiv(num_heads, BLK_HEADS))

    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_out = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)
    g_norm = (
        torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
        if return_g_norm
        else g  # 别名到一个合法指针；kernel 在 return_g_norm=False 时不会解引用
    )

    _fused_gdn_gating_l2norm_kernel[grid](
        g,
        beta_out,
        g_norm,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        stride_a,
        stride_b,
        num_heads,
        beta,
        threshold,
        eps,
        BLK_HEADS,
        RETURN_G_NORM=return_g_norm,
        num_warps=1,
    )
    return g, beta_out, (g_norm if return_g_norm else None)


# ----------------------------------------------------------------------------
# 向后兼容的 drop-in 替换
# ----------------------------------------------------------------------------
# 我们对外暴露与原始模块同名的 ``fused_gdn_gating`` 符号，但其内部是
# 一个调用融合 kernel、丢弃 ``g_norm`` 的薄包装。这是 ``gdn_backend.py``
# 在开启环境变量开关后会 import 的版本。
def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """原始 ``fused_gdn_gating`` 的 drop-in 替换，底层走 v2 kernel。

    函数签名故意与原版保持一致，调用方无需修改。
    """
    g, beta_out, _g_norm = fused_gdn_gating_with_l2norm(
        A_log,
        a,
        b,
        dt_bias,
        beta=beta,
        threshold=threshold,
        return_g_norm=False,
    )
    return g, beta_out


# ----------------------------------------------------------------------------
# 基于环境变量的选择器
# ----------------------------------------------------------------------------
# GDN backend 可以根据 ``SGLANG_FUSE_GDN_GATING`` 环境变量在原版与
# v2 之间选择。我们暴露一个辅助函数，让 backend 无需关心环境变量名。
def get_active_fused_gdn_gating():
    """根据环境变量开关返回当前应使用的 ``fused_gdn_gating`` 实现。

    设置 ``SGLANG_FUSE_GDN_GATING=1`` 启用 v2（融合 l2norm）实现。
    默认走 v1 原始实现，避免静默修改行为。
    """
    if os.getenv("SGLANG_FUSE_GDN_GATING", "0") == "1":
        return fused_gdn_gating
    # 在此处 import，避免循环依赖。
    from sglang.srt.layers.attention.fla.fused_gdn_gating import (
        fused_gdn_gating as _v1,
    )
    return _v1
