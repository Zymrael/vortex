"""
Correctness tests for the HCM fused Triton kernels.

_hcm_complex_mul is stage 3 of the HCM FFT-conv -- the elementwise product
of the activation and filter spectra, with stage 1's 1/fft_size filter
normalisation folded in. _hcm_bias_residual is stage 5 -- the skip-residual
add y + u * bias[:, None], written out at u's dtype. Each kernel is checked
against the explicit torch expression fftconv_func evaluates.
"""

import pytest
import torch

from vortex.model.engine import adjust_filter_shape_for_broadcast
from vortex.ops.hcm_interface import _hcm_bias_residual, _hcm_complex_mul

CUDA: bool = torch.cuda.is_available()


def _spectra(
    B: int, D: int, L: int, fir_length: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build the activation and filter spectra exactly as fftconv_func does.

    Returns:
        tuple[torch.Tensor, torch.Tensor, int]: u_f, the activation spectrum
        of shape (B, D, F); k_f, the unscaled filter spectrum of shape
        (1, D, F) broadcast over the batch; and the integer fft_size.
    """
    torch.manual_seed(0)
    fft_size = 2 * L
    u = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    k = torch.randn(D, fir_length, dtype=torch.float32, device="cuda")
    u_f = torch.fft.rfft(u, n=fft_size)
    k_f = adjust_filter_shape_for_broadcast(u, torch.fft.rfft(k, n=fft_size))
    return u_f, k_f, fft_size


@pytest.mark.skipif(not CUDA, reason="HCM Triton kernel requires CUDA")
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("L", [1024, 8192, 32768])
def test_hcm_complex_mul_matches_oracle(B: int, L: int) -> None:
    """
    The fused complex multiply matches u_f * (k_f / fft_size) at evo2_7b
    shapes; B=2 exercises the filter broadcast over the batch.
    """
    u_f, k_f, fft_size = _spectra(B, D=4096, L=L, fir_length=128)

    y_f = _hcm_complex_mul(u_f, k_f, fft_size)
    y_f_ref = u_f * (k_f / fft_size)

    assert y_f.shape == u_f.shape
    assert y_f.dtype == u_f.dtype
    max_diff = (y_f - y_f_ref).abs().max().item()
    mean_diff = (y_f - y_f_ref).abs().mean().item()
    assert max_diff < 1e-4, f"max_diff={max_diff:.2e}"
    assert mean_diff < 1e-5, f"mean_diff={mean_diff:.2e}"


@pytest.mark.skipif(not CUDA, reason="HCM Triton kernel requires CUDA")
def test_hcm_complex_mul_masks_ragged_tail() -> None:
    """
    The kernel masks the flat (D, F) tail when D*F does not divide the tile.
    """
    u_f, k_f, fft_size = _spectra(B=1, D=16, L=100, fir_length=4)

    y_f = _hcm_complex_mul(u_f, k_f, fft_size)
    y_f_ref = u_f * (k_f / fft_size)
    assert (y_f - y_f_ref).abs().max().item() < 1e-4


@pytest.mark.skipif(not CUDA, reason="HCM Triton kernel requires CUDA")
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("L", [1024, 8192, 32768])
def test_hcm_bias_residual_matches_oracle(B: int, L: int) -> None:
    """
    The fused bias-residual matches (y + u * bias[:, None]).to(u.dtype) at
    evo2_7b shapes; B=2 exercises the per-channel bias broadcast over batch.
    """
    torch.manual_seed(0)
    D = 4096
    y = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    u = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    bias = torch.randn(D, dtype=torch.float32, device="cuda")

    out = _hcm_bias_residual(y, u, bias)
    out_ref = (y + u * bias.unsqueeze(-1)).to(u.dtype)

    assert out.shape == y.shape
    assert out.dtype == u.dtype
    max_diff = (out - out_ref).abs().max().item()
    mean_diff = (out - out_ref).abs().mean().item()
    assert max_diff < 1e-4, f"max_diff={max_diff:.2e}"
    assert mean_diff < 1e-5, f"mean_diff={mean_diff:.2e}"


@pytest.mark.skipif(not CUDA, reason="HCM Triton kernel requires CUDA")
def test_hcm_bias_residual_masks_ragged_tile() -> None:
    """
    The kernel masks (D, L) tiles that BLOCK_D x BLOCK_L does not divide evenly.
    """
    torch.manual_seed(0)
    B, D, L = 1, 100, 300
    y = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    u = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    bias = torch.randn(D, dtype=torch.float32, device="cuda")

    out = _hcm_bias_residual(y, u, bias)
    out_ref = (y + u * bias.unsqueeze(-1)).to(u.dtype)
    assert (out - out_ref).abs().max().item() < 1e-4
