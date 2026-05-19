# pyright: reportAttributeAccessIssue=none
"""
HCS -- Hyena Cascade Short.

A from-scratch Triton depthwise causal 1D convolution for the short-filter
(fir_length < 128) gated branch of HyenaInferenceEngine.parallel_fir.

The convolution is the only time-mixing op in an HCS layer: a depthwise
filter of fir_length taps (7 in evo2_7b) applied per channel. This module
provides the @triton.jit kernel and a thin Python launcher; the hcs_conv
adapter that wires it behind the use_hcs_kernel config flag is added alongside.
"""

from typing import Callable

import torch
import triton
import triton.language as tl

# Autotuned search space for the conv kernel's register-tile sizes. Triton
# benchmarks these once per (D, L, FIR_LEN) and caches the winner, so no
# single tile size is hard-coded -- the GPU and shape pick it.
_AUTOTUNE_CONFIGS: list[triton.Config] = [
    triton.Config({"BLOCK_D": 32, "BLOCK_L": 64}, num_warps=2),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 64}, num_warps=4),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 128}, num_warps=4),
    triton.Config({"BLOCK_D": 128, "BLOCK_L": 64}, num_warps=4),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 256}, num_warps=8),
    triton.Config({"BLOCK_D": 128, "BLOCK_L": 128}, num_warps=8),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["D", "L", "FIR_LEN"])
@triton.jit
def _hcs_depthwise_conv_kernel(
    u_ptr,
    w_ptr,
    z_ptr,
    D,
    L,
    stride_ub,
    stride_ud,
    stride_ul,
    stride_wd,
    stride_wk,
    FIR_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """
    Depthwise causal conv: z[b, d, t] = sum_k w[d, k] * u[b, d, t - FIR_LEN + 1 + k].

    One program covers a (BLOCK_D, BLOCK_L) tile of one batch element. The
    FIR_LEN tap loop is unrolled at compile time. Input positions before 0
    are masked to zero, giving a causal (left-padded) convolution.
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_l = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_d = offs_d < D
    mask_l = offs_l < L

    u_base = u_ptr + pid_b * stride_ub + offs_d[:, None] * stride_ud
    acc = tl.zeros((BLOCK_D, BLOCK_L), dtype=tl.float32)

    for k in tl.static_range(FIR_LEN):
        w_k = tl.load(
            w_ptr + offs_d * stride_wd + k * stride_wk, mask=mask_d, other=0.0
        )
        pos = offs_l - (FIR_LEN - 1) + k
        mask_pos = mask_d[:, None] & (pos[None, :] >= 0) & (pos[None, :] < L)
        u_tile = tl.load(u_base + pos[None, :] * stride_ul, mask=mask_pos, other=0.0)
        acc += w_k[:, None].to(tl.float32) * u_tile.to(tl.float32)

    z_ptrs = (
        z_ptr
        + pid_b * stride_ub
        + offs_d[:, None] * stride_ud
        + offs_l[None, :] * stride_ul
    )
    tl.store(z_ptrs, acc, mask=mask_d[:, None] & mask_l[None, :])


def hcs_depthwise_conv(u: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Depthwise causal 1D convolution, the HCS short-filter time-mixing op.

    Equivalent to F.conv1d(u, weight, padding=fir_length - 1, groups=D)
    trimmed to length L, but in a single fused Triton launch.

    Args:
        u (torch.Tensor): Input activations, shape (B, D, L), contiguous.
        weight (torch.Tensor): Depthwise filter, shape (D, 1, fir_length),
                               contiguous. Every channel has its own filter.

    Returns:
        torch.Tensor: Convolved output, shape (B, D, L), same dtype as u.
    """
    if not u.is_contiguous():
        u = u.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if u.dim() != 3 or weight.dim() != 3:
        raise ValueError(f"expected 3-D u and weight, got {u.shape} and {weight.shape}")

    B, D, L = u.shape
    Dw, in_per_group, fir_length = weight.shape
    if Dw != D or in_per_group != 1:
        raise ValueError(f"weight {tuple(weight.shape)} is not depthwise for D={D}")

    z: torch.Tensor = torch.empty_like(u)
    # BLOCK_D / BLOCK_L are supplied by @triton.autotune; the grid is a
    # callable so it can read the chosen tile sizes from the winning config.
    grid: Callable[[triton.Config], tuple[int, int, int]] = lambda meta: (
        B,
        triton.cdiv(D, meta["BLOCK_D"]),
        triton.cdiv(L, meta["BLOCK_L"]),
    )
    _hcs_depthwise_conv_kernel[grid](
        u,
        weight,
        z,
        D,
        L,
        u.stride(0),
        u.stride(1),
        u.stride(2),
        weight.stride(0),
        weight.stride(2),
        FIR_LEN=fir_length,
    )
    return z


def hcs_conv(
    x1: torch.Tensor,
    x2: torch.Tensor,
    v: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    gated_bias: bool = False,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fully-gated HCS short conv: z = x2 * (conv(x1 * v, weight) + bias).

    Drop-in replacement for the gated fir_length < 128 branch of
    HyenaInferenceEngine.parallel_fir. It reproduces that branch exactly:
    the depthwise conv runs in fp32 for numerical parity with the F.conv1d
    path, the result is cast back to the activation dtype, bias-added,
    masked, then closed with the post-gate multiply by x2.

    Args:
        x1 (torch.Tensor): Pre-gate "key" stream, shape (B, D, L).
        x2 (torch.Tensor): Post-gate stream, shape (B, D, L).
        v (torch.Tensor): "Value" stream, shape (B, D, L).
        weight (torch.Tensor): Depthwise filter, shape (D, 1, fir_length).
        bias (torch.Tensor | None): Per-channel skip-gain, shape (D,).
        gated_bias (bool): If True the bias is applied multiplicatively
                           (bias * x1 * v); HCS uses additive bias (False).
        padding_mask (torch.Tensor | None): If a tensor, zeros masked
                                            positions after the conv, shape (B, L).

    Returns:
        torch.Tensor: Gated HCS output, shape (B, D, L), x1's dtype.
    """
    u: torch.Tensor = x1 * v
    z: torch.Tensor = hcs_depthwise_conv(u=u.float(), weight=weight.float())
    z = z.to(u.dtype)

    if bias is not None:
        if gated_bias:
            z = z + bias[None, :, None] * u
        else:
            z = z + bias[None, :, None]

    if isinstance(padding_mask, torch.Tensor):
        z = z * padding_mask[:, None]

    return x2 * z
