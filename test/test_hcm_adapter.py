"""
Tests for hcm_fft_conv -- the drop-in for the fir_length >= 128 (HCM) branch
of HyenaInferenceEngine.parallel_fir.

The oracle is upstream fftconv_func: with use_hcm_kernel off the engine calls
it directly, so hcm_fft_conv must reproduce it -- the flag is a no-op.
"""

import pytest
import torch

from vortex.model.engine import fftconv_func
from vortex.ops.hcm_interface import hcm_fft_conv


def _hcm_inputs(B: int, L: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build (u, weight, bias) for an HCM fftconv call at evo2_7b shapes.
    """
    torch.manual_seed(0)
    D, K = 4096, 128
    u = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    weight = torch.randn(D, 1, K, dtype=torch.float32, device="cuda")
    bias = torch.randn(D, dtype=torch.float32, device="cuda")
    return u, weight, bias


@pytest.mark.gpu
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("L", [2048, 8192, 32768])
def test_hcm_fft_conv_matches_fftconv_func(B: int, L: int) -> None:
    """
    hcm_fft_conv reproduces fftconv_func on the non-bidirectional HCM path;
    B=2 exercises the filter broadcast over the batch.
    """
    u, weight, bias = _hcm_inputs(B, L)

    z = hcm_fft_conv(u, weight, bias, None, gelu=False, bidirectional=False)
    z_ref = fftconv_func(u, weight, bias, None, gelu=False, bidirectional=False)

    assert z.shape == z_ref.shape == u.shape
    assert z.dtype == z_ref.dtype
    max_diff = (z - z_ref).abs().max().item()
    mean_diff = (z - z_ref).abs().mean().item()
    assert max_diff < 1e-2, f"max_diff={max_diff:.2e}"
    assert mean_diff < 1e-3, f"mean_diff={mean_diff:.2e}"


@pytest.mark.gpu
def test_hcm_fft_conv_rejects_unsupported_paths() -> None:
    """
    hcm_fft_conv raises on bidirectional, reverse-filter, gelu, or dropout-mask
    calls -- paths the HCM kernel does not implement.
    """
    u, weight, bias = _hcm_inputs(1, 2048)
    with pytest.raises(NotImplementedError):
        hcm_fft_conv(u, weight, bias, None, bidirectional=True)
    with pytest.raises(NotImplementedError):
        hcm_fft_conv(u, weight, bias, None, k_rev=weight)
    with pytest.raises(NotImplementedError):
        hcm_fft_conv(u, weight, bias, None, gelu=True)
    with pytest.raises(NotImplementedError):
        hcm_fft_conv(u, weight, bias, torch.ones_like(u))
