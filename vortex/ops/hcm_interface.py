# pyright: reportAttributeAccessIssue=none
"""
HCM -- Hyena Cascade Medium.

Fused Triton epilogues for the FFT-convolution path of
HyenaInferenceEngine.parallel_fir (the fir_length >= 128 branch). At a
128-tap filter Triton cannot out-write cuFFT for the transforms themselves,
so the win is launch-count: the elementwise glue around the three cuFFT
calls is fused into Triton kernels.

It provides _hcm_complex_mul (stage 3 of fftconv_func -- the broadcast
complex product u_f * k_f, with stage 1's 1/fft_size scale folded in) and
_hcm_bias_residual (stage 5 -- the skip-residual add y + u * bias).
"""

from typing import Callable

import torch
import triton
import triton.language as tl

# Autotuned search spaces -- both kernels are memory-bound elementwise work,
# so the winning tile is whichever best saturates bandwidth at a given shape.
# Triton benchmarks each set once per shape key and caches the winner.
_COMPLEX_MUL_CONFIGS: list[triton.Config] = [
    triton.Config({"BLOCK": 256}, num_warps=2),
    triton.Config({"BLOCK": 512}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
]

_BIAS_RESIDUAL_CONFIGS: list[triton.Config] = [
    triton.Config({"BLOCK_D": 32, "BLOCK_L": 64}, num_warps=2),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 64}, num_warps=4),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 128}, num_warps=4),
    triton.Config({"BLOCK_D": 128, "BLOCK_L": 64}, num_warps=4),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 256}, num_warps=8),
    triton.Config({"BLOCK_D": 128, "BLOCK_L": 128}, num_warps=8),
]


@triton.autotune(configs=_COMPLEX_MUL_CONFIGS, key=["DF"])
@triton.jit
def _hcm_complex_mul_kernel(
    u_ptr,
    k_ptr,
    y_ptr,
    DF,
    inv_fft_size,
    stride_batch,
    BLOCK: tl.constexpr,
):
    """
    Broadcast complex multiply: y[b] = u_f[b] * k_f[0] * inv_fft_size.

    One program covers a BLOCK-element slice of the flattened (D, F) plane
    of one batch element. Complex values are stored interleaved (real, imag)
    -- view_as_real's layout -- so element n's real part is at offset 2n and
    its imag part at 2n + 1. The filter k_f carries no batch stride: every
    batch element multiplies against the same spectrum.

    The flat (D, F) tile is grid axis 0: cdiv(DF, BLOCK) overruns the 65535
    cap on axes 1 and 2 at long context, so the small batch sits on axis 1.
    """
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < DF

    u_base = u_ptr + pid_b * stride_batch
    y_base = y_ptr + pid_b * stride_batch

    u_re = tl.load(u_base + 2 * offs, mask=mask, other=0.0)
    u_im = tl.load(u_base + 2 * offs + 1, mask=mask, other=0.0)
    k_re = tl.load(k_ptr + 2 * offs, mask=mask, other=0.0)
    k_im = tl.load(k_ptr + 2 * offs + 1, mask=mask, other=0.0)

    y_re = (u_re * k_re - u_im * k_im) * inv_fft_size
    y_im = (u_re * k_im + u_im * k_re) * inv_fft_size

    tl.store(y_base + 2 * offs, y_re, mask=mask)
    tl.store(y_base + 2 * offs + 1, y_im, mask=mask)


def _hcm_complex_mul(
    u_f: torch.Tensor, k_f: torch.Tensor, fft_size: int
) -> torch.Tensor:
    """
    Fused broadcast complex multiply with the 1/fft_size filter scale.

    Computes u_f * k_f / fft_size -- stage 3 of fftconv_func, with stage 1's
    filter normalisation folded in. u_f is the activation spectrum; k_f is
    the *unscaled* filter spectrum, already shaped for broadcast over the
    batch by adjust_filter_shape_for_broadcast.

    Args:
        u_f (torch.Tensor): Activation spectrum, complex, shape (B, D, F).
        k_f (torch.Tensor): Filter spectrum, complex, shape (1, D, F), shared
                            across the batch and not yet scaled by 1/fft_size.
        fft_size (int): The FFT length n = 2 * seqlen; its reciprocal folds
                        in as the filter normalisation.

    Returns:
        torch.Tensor: The scaled product u_f * k_f / fft_size, complex, shape
                      (B, D, F), u_f's dtype.

    Raises:
        ValueError: If the tensors are not 3-D complex, or k_f is not
                    broadcastable over the batch of u_f.
    """
    if u_f.dim() != 3 or k_f.dim() != 3:
        raise ValueError(
            f"expected 3-D u_f and k_f, got {tuple(u_f.shape)} and {tuple(k_f.shape)}"
        )
    if not u_f.is_complex() or not k_f.is_complex():
        raise ValueError("u_f and k_f must be complex tensors")

    B, D, F = u_f.shape
    if tuple(k_f.shape) != (1, D, F):
        raise ValueError(
            f"k_f {tuple(k_f.shape)} is not broadcastable over u_f {tuple(u_f.shape)}"
        )

    u_f = u_f.contiguous()
    k_f = k_f.contiguous()
    y_f: torch.Tensor = torch.empty_like(u_f)

    # Triton has no complex dtype: operate on the (..., 2) real/imag view.
    u_r = torch.view_as_real(u_f)
    k_r = torch.view_as_real(k_f)
    y_r = torch.view_as_real(y_f)

    DF = D * F
    # BLOCK is supplied by @triton.autotune; the grid is a callable so it can
    # read the chosen tile size from the winning config.
    grid: Callable[[triton.Config], tuple[int, int]] = lambda meta: (
        triton.cdiv(DF, meta["BLOCK"]),
        B,
    )
    _hcm_complex_mul_kernel[grid](
        u_r,
        k_r,
        y_r,
        DF,
        1.0 / fft_size,
        u_r.stride(0),
    )
    return y_f


@triton.autotune(configs=_BIAS_RESIDUAL_CONFIGS, key=["D", "L"])
@triton.jit
def _hcm_bias_residual_kernel(
    y_ptr,
    u_ptr,
    bias_ptr,
    out_ptr,
    D,
    L,
    stride_b,
    stride_d,
    stride_l,
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """
    Skip-residual add: out[b, d, l] = y[b, d, l] + u[b, d, l] * bias[d].

    One program covers a (BLOCK_D, BLOCK_L) tile of one batch element. y, u
    and out share a contiguous (B, D, L) layout; bias is per-channel, shape
    (D,), broadcast over batch and length. The fp32 accumulator is cast to
    out's dtype on the store -- fftconv_func's stage-6 .to(u.dtype) cast.
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_l = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_d = offs_d < D
    mask_l = offs_l < L
    mask = mask_d[:, None] & mask_l[None, :]

    offs = pid_b * stride_b + offs_d[:, None] * stride_d + offs_l[None, :] * stride_l
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    acc = y + u * bias[:, None]
    tl.store(out_ptr + offs, acc, mask=mask)


def _hcm_bias_residual(
    y: torch.Tensor, u: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """
    Fused skip-residual add -- stage 5 of fftconv_func.

    Computes y + u * bias[:, None] and writes it at u's dtype, fusing
    fftconv_func's broadcast multiply and residual add (and its stage-6
    dtype cast) into a single Triton launch.

    Args:
        y (torch.Tensor): The irfft output, shape (B, D, L).
        u (torch.Tensor): The activations, shape (B, D, L); its dtype is the
                          output dtype -- fftconv_func's stage-6 cast target.
        bias (torch.Tensor): Per-channel skip gain, shape (D,), broadcast
                             over batch and length.

    Returns:
        torch.Tensor: y + u * bias[:, None], shape (B, D, L), u's dtype.

    Raises:
        ValueError: If y and u are not matching 3-D tensors, or bias is not
                    1-D of length D.
    """
    if y.dim() != 3 or u.dim() != 3:
        raise ValueError(
            f"expected 3-D y and u, got {tuple(y.shape)} and {tuple(u.shape)}"
        )
    if y.shape != u.shape:
        raise ValueError(f"y {tuple(y.shape)} and u {tuple(u.shape)} must match")

    B, D, L = u.shape
    if bias.dim() != 1 or bias.shape[0] != D:
        raise ValueError(f"bias {tuple(bias.shape)} must be 1-D of length D={D}")

    y = y.contiguous()
    u = u.contiguous()
    bias = bias.contiguous()
    out: torch.Tensor = torch.empty_like(u)

    # BLOCK_D / BLOCK_L are supplied by @triton.autotune; the grid is a
    # callable so it can read the chosen tile sizes from the winning config.
    grid: Callable[[triton.Config], tuple[int, int, int]] = lambda meta: (
        B,
        triton.cdiv(D, meta["BLOCK_D"]),
        triton.cdiv(L, meta["BLOCK_L"]),
    )
    _hcm_bias_residual_kernel[grid](
        y,
        u,
        bias,
        out,
        D,
        L,
        u.stride(0),
        u.stride(1),
        u.stride(2),
    )
    return out
