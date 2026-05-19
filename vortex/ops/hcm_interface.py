# pyright: reportAttributeAccessIssue=none
"""
HCM -- Hyena Cascade Medium.

Fused Triton epilogues for the FFT-convolution path of
HyenaInferenceEngine.parallel_fir (the fir_length >= 128 branch). At a
128-tap filter Triton cannot out-write cuFFT for the transforms themselves,
so the win is launch-count: the elementwise glue around the three cuFFT
calls is fused into Triton kernels.

This module currently provides _hcm_complex_mul -- stage 3 of fftconv_func,
the broadcast complex product u_f * k_f, with stage 1's 1/fft_size filter
normalisation folded in.
"""

from typing import Callable

import torch
import triton
import triton.language as tl

# Autotuned search space for the flat (D, F) tile. The op is memory-bound
# elementwise, so the winner is whichever BLOCK best saturates bandwidth for
# a given problem size -- Triton benchmarks these once per DF and caches it.
_AUTOTUNE_CONFIGS: list[triton.Config] = [
    triton.Config({"BLOCK": 256}, num_warps=2),
    triton.Config({"BLOCK": 512}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["DF"])
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
