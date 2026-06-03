"""v2 融合 GDN gating kernel 的单元测试（方案 1 融合）。

本测试验证两件事：

1. **数值正确性**：v2 kernel 产出的 ``g``、``beta`` tensor 必须与 v1
   kernel 严格一致；``g_norm`` 必须与一份手写的 L2 归一化参考实现一致。

2. **性能**：当 ``return_g_norm=True`` 时，v2 kernel 至少要与 v1 kernel
   + 独立 L2 归一化调用一样快，因为前者把 L2 归一化融合进了 gating 过程。

运行方式::

    pytest test/registered/attention/test_fused_gdn_gating_v2.py -v

或作为独立脚本::

    python test/registered/attention/test_fused_gdn_gating_v2.py
"""

from __future__ import annotations

import sys

import pytest
import torch

# v1 参考 kernel 始终可用；v2 kernel 紧邻其旁，是本测试的重点。
from sglang.srt.layers.attention.fla.fused_gdn_gating import (
    fused_gdn_gating as fused_gdn_gating_v1,
)
from sglang.srt.layers.attention.fla.fused_gdn_gating_v2 import (
    fused_gdn_gating as fused_gdn_gating_v2,
    fused_gdn_gating_with_l2norm,
)
from sglang.srt.layers.attention.fla.l2norm import l2norm


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _make_inputs(batch, num_heads, device="cuda", dtype=torch.bfloat16, seed=0):
    """构造四个 GDN gating 输入，shape 与生产环境一致。"""
    torch.manual_seed(seed)
    A_log = torch.randn(num_heads, dtype=torch.float32, device=device)
    a = torch.randn(batch, num_heads, dtype=dtype, device=device)
    b = torch.randn(batch, num_heads, dtype=dtype, device=device)
    dt_bias = torch.randn(num_heads, dtype=dtype, device=device)
    return A_log, a, b, dt_bias


# ---------------------------------------------------------------------------
# 数值正确性
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
@pytest.mark.parametrize("batch", [1, 8, 32])
@pytest.mark.parametrize("num_heads", [16, 32, 64, 128])
def test_v2_g_beta_match_v1(batch, num_heads):
    """v2 的 ``(g, beta)`` 输出必须与 v1 一致（kernel 计算逻辑相同）。

    容差说明
    --------
    * ``g`` 在 v1 / v2 中都以 fp32 写出，两份 kernel 计算逻辑相同，
      应当 bit-identical，因此用 fp32 级容差 ``1e-5``。
    * ``beta`` 在 v1 / v2 中都以 bf16 写出。Triton 编译器会为不同
      kernel 结构生成不同的 SASS，导致 ``sigmoid(blk_b.to(fp32))`` 的
      末位有微小差异，cast 回 bf16 时可能差 1 ULP。bf16 在 1.0 附近
      的 ULP 为 ``2**-7 ≈ 7.8e-3``，所以容差设为 ``5e-3``（约 0.6 ULP）。
    """
    A_log, a, b, dt_bias = _make_inputs(batch, num_heads)

    g1, beta1 = fused_gdn_gating_v1(A_log, a, b, dt_bias)
    g2, beta2 = fused_gdn_gating_v2(A_log, a, b, dt_bias)

    # g 是 fp32 输出，应 bit-identical。
    assert torch.allclose(g1, g2, atol=1e-5, rtol=1e-5), (
        f"g 不一致：最大差 = {(g1 - g2).abs().max().item()}"
    )
    # beta 是 bf16 输出，容差放宽到 bf16 ULP 量级。
    assert torch.allclose(beta1, beta2, atol=5e-3, rtol=5e-3), (
        f"beta 不一致：最大差 = {(beta1 - beta2).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
@pytest.mark.parametrize("batch", [1, 8, 32])
@pytest.mark.parametrize("num_heads", [16, 32, 64, 128])
def test_v2_g_norm_matches_reference(batch, num_heads):
    """v2 的 ``g_norm`` 必须与独立的 ``l2norm`` 参考实现一致。

    同时附带一个 L2 范数的健全性检查：``g_norm`` 的 L2 范数应该
    约等于 1（即确实是「归一化」过的，不是 RMSNorm）。
    """
    A_log, a, b, dt_bias = _make_inputs(batch, num_heads)

    g_v1, beta_v1 = fused_gdn_gating_v1(A_log, a, b, dt_bias)
    g_ref, beta_v2, g_norm_v2 = fused_gdn_gating_with_l2norm(A_log, a, b, dt_bias)

    # 先验证 g / beta 与 v1 参考一致。
    assert torch.allclose(g_v1, g_ref, atol=1e-5, rtol=1e-5)

    # 把 g_norm 与参考 l2norm 比较。
    g_norm_ref = l2norm(g_v1)
    diff = (g_norm_ref - g_norm_v2).abs().max().item()
    assert diff < 1e-4, f"g_norm 最大差 = {diff}"

    # 健全性检查：L2 归一化后，``sum(g_norm ** 2) ≈ 1``（每个行）。
    # 如果该值 ≈ 1/N，则是 RMSNorm 而不是 L2 norm（曾经的 bug 现象）。
    sumsq = (g_norm_v2 * g_norm_v2).sum(dim=-1)  # (B,)
    assert torch.allclose(sumsq, torch.ones_like(sumsq), atol=1e-3), (
        f"g_norm 不是 L2 归一化：sum(g_norm^2) 应 ≈ 1，实际 = {sumsq.tolist()}"
    )


# ---------------------------------------------------------------------------
# 性能
# ---------------------------------------------------------------------------
def _time_fn(fn, iters=100, warmup=20):
    """用 CUDA 事件对函数计时（自带 warmup）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms / 次


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
@pytest.mark.parametrize("num_heads", [32, 128])
def test_v2_fused_l2norm_is_faster_than_separate(num_heads):
    """v2（gating + l2norm 单 kernel）应比 v1 + 独立 l2norm 更快。"""
    A_log, a, b, dt_bias = _make_inputs(batch=32, num_heads=num_heads)

    # 参考路径：v1 gating + 独立 l2norm。
    def v1_with_l2norm():
        g, beta = fused_gdn_gating_v1(A_log, a, b, dt_bias)
        g_norm = l2norm(g)
        return g, beta, g_norm

    # 融合路径：单 v2 kernel。
    def v2_fused():
        return fused_gdn_gating_with_l2norm(A_log, a, b, dt_bias)

    t_v1 = _time_fn(v1_with_l2norm)
    t_v2 = _time_fn(v2_fused)
    print(
        f"num_heads={num_heads}: v1+l2norm={t_v1*1000:.2f}us, "
        f"v2_fused={t_v2*1000:.2f}us, speedup={t_v1/t_v2:.2f}x"
    )
    # 融合 kernel 至少应该一样快；实际上 v1+l2norm 路径有 2 次 kernel
    # launch + 2 次 ``g`` 的 HBM 往返，v2 一定赢。
    assert t_v2 < t_v1, (
        f"v2 融合 ({t_v2*1000:.2f}us) 不比 v1+l2norm "
        f"({t_v1*1000:.2f}us) 快"
    )


# ---------------------------------------------------------------------------
# 端到端烟雾测试（无 benchmark）：模拟 GDN 线性注意力消费 gating + l2norm
# 的链路。
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_v2_drop_in_keeps_dtypes_and_shapes():
    """drop-in ``fused_gdn_gating`` 必须保持与 v1 一致的 dtype/shape，
    这样现有调用点无需修改。"""
    A_log, a, b, dt_bias = _make_inputs(batch=4, num_heads=64)

    g1, beta1 = fused_gdn_gating_v1(A_log, a, b, dt_bias)
    g2, beta2 = fused_gdn_gating_v2(A_log, a, b, dt_bias)

    assert g1.shape == g2.shape == (1, 4, 64)
    assert beta1.shape == beta2.shape == (1, 4, 64)
    assert g1.dtype == g2.dtype == torch.float32
    assert beta1.dtype == beta2.dtype == torch.float32


# ---------------------------------------------------------------------------
# 独立入口（无 pytest 时手动跑 benchmark 用）
# ---------------------------------------------------------------------------
def _main():
    if not torch.cuda.is_available():
        print("CUDA 不可用，无事可做。")
        return 0

    print("=== v2 vs v1：数值正确性 ===")
    for nh in (16, 32, 64, 128):
        for bs in (1, 8, 32):
            A_log, a, b, dt_bias = _make_inputs(bs, nh)
            g1, beta1 = fused_gdn_gating_v1(A_log, a, b, dt_bias)
            g2, beta2 = fused_gdn_gating_v2(A_log, a, b, dt_bias)
            g_d = (g1 - g2).abs().max().item()
            b_d = (beta1 - beta2).abs().max().item()
            print(f"  bs={bs:3d} hv={nh:3d}  max|g1-g2|={g_d:.2e}  "
                  f"max|b1-b2|={b_d:.2e}")

    print("\n=== v2 vs v1+l2norm：性能 ===")
    for nh in (32, 64, 128):
        A_log, a, b, dt_bias = _make_inputs(batch=32, num_heads=nh)
        t_v1 = _time_fn(lambda: (
            l2norm(fused_gdn_gating_v1(A_log, a, b, dt_bias)[0])
        ))
        t_v2 = _time_fn(lambda: fused_gdn_gating_with_l2norm(A_log, a, b, dt_bias))
        print(
            f"  num_heads={nh}: v1+l2norm={t_v1*1000:6.2f}us  "
            f"v2_fused={t_v2*1000:6.2f}us  speedup={t_v1/t_v2:.2f}x"
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
