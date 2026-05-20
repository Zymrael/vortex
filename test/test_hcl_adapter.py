"""
Tests for hcl_fft_conv -- the fused FFT-conv epilogue for parallel_iir's
long_fir_threshold-is-None (HCL) branch.

The oracle is the stock branch: rfft(h)/fft_size, fft(x1v), X*H, irfft, then
the post-conv (y + x1v*D[:, None]) * x2. hcl_fft_conv reproduces it with the
_hcm_complex_mul and _hcl_bias_residual_gate kernels.
"""

import pytest
import torch

from vortex.ops.hcl_interface import hcl_fft_conv


def _hcl_branch_ref(
    h: torch.Tensor,
    x1v: torch.Tensor,
    x2: torch.Tensor,
    D: torch.Tensor,
    L: int,
    fft_size: int,
) -> torch.Tensor:
    """
    Pure-torch reference for parallel_iir's HCL FFT-conv branch + post-conv.
    """
    H = torch.fft.rfft(h.to(torch.float32), n=fft_size) / fft_size
    X = torch.fft.fft(x1v.to(torch.float32), n=fft_size)[..., : H.shape[-1]]
    y = torch.fft.irfft(X * H, n=fft_size, norm="forward")[..., :L]
    y = y.to(x1v.dtype)
    return (y + x1v * D.unsqueeze(-1)) * x2


@pytest.mark.gpu
@pytest.mark.parametrize("L", [2048, 8192, 32768])
def test_hcl_fft_conv_matches_branch(L: int) -> None:
    """
    hcl_fft_conv reproduces the stock parallel_iir HCL branch in fp32.
    """
    torch.manual_seed(0)
    D = 4096
    fft_size = 2 * L
    h = torch.randn(1, D, L, dtype=torch.float32, device="cuda")
    x1v = torch.randn(1, D, L, dtype=torch.float32, device="cuda")
    x2 = torch.randn(1, D, L, dtype=torch.float32, device="cuda")
    bias = torch.randn(D, dtype=torch.float32, device="cuda")

    y = hcl_fft_conv(h, x1v, x2, bias, L, fft_size)
    y_ref = _hcl_branch_ref(h, x1v, x2, bias, L, fft_size)

    assert y.shape == y_ref.shape == (1, D, L)
    assert y.dtype == y_ref.dtype
    max_diff = (y - y_ref).abs().max().item()
    mean_diff = (y - y_ref).abs().mean().item()
    assert max_diff < 1e-2, f"max_diff={max_diff:.2e}"
    assert mean_diff < 1e-3, f"mean_diff={mean_diff:.2e}"
