"""
Wiring tests for the use_hcl_kernel branch in HyenaInferenceEngine.parallel_iir.

These exercise the long_fir_threshold-is-None dispatch by calling parallel_iir
directly. With the kernel enabled, the HCL FFT-conv must reproduce the stock
branch; the branch must not fire when long_fir_threshold is set.
"""

import pytest
import torch

from vortex.model.engine import HyenaInferenceEngine

CUDA: bool = torch.cuda.is_available()

# evo2_7b HCL shapes: D=4096, state_size=16.
B, D, S = 1, 4096, 16
DIMS: tuple[int, int, int, int, int] = (D, 32, D // 32, 16, 256)


def _hcl_inputs(L: int, dtype: torch.dtype):
    """
    Build (z_pre, h, bias, poles, residues, t) for an HCL parallel_iir call.
    """
    z_pre = torch.randn(B, 3 * D, L, dtype=dtype, device="cuda")
    h = torch.randn(1, D, L, dtype=torch.float32, device="cuda")
    bias = torch.randn(D, dtype=dtype, device="cuda")
    poles = torch.randn(D, S, 1, dtype=torch.float32, device="cuda")
    residues = torch.randn(D, S, dtype=torch.float32, device="cuda")
    t = torch.arange(L, device="cuda")
    return z_pre, h, bias, poles, residues, t


def _call(engine, z_pre, h, bias, L, poles, residues, t, **kw):
    """
    Invoke parallel_iir on the HCL FFT path (long_fir_threshold None).
    """
    return engine.parallel_iir(
        z_pre,
        h,
        bias,
        L,
        poles=poles,
        residues=residues,
        t=t,
        dims=DIMS,
        layer_idx=0,
        long_fir_threshold=kw.pop("long_fir_threshold", None),
        **kw,
    )


@pytest.mark.skipif(not CUDA, reason="HCL kernel requires CUDA")
@pytest.mark.parametrize("L", [2048, 8192])
def test_vk_hcl_on_matches_baseline_fp32(L: int) -> None:
    """
    use_hcl_kernel on reproduces the stock parallel_iir HCL output in fp32.
    """
    torch.manual_seed(0)
    engine = HyenaInferenceEngine(layer_idx=0)
    z_pre, h, bias, poles, residues, t = _hcl_inputs(L, torch.float32)

    y_off = _call(engine, z_pre, h, bias, L, poles, residues, t)
    engine.use_hcl_kernel = True
    y_on = _call(engine, z_pre, h, bias, L, poles, residues, t)

    assert y_on.shape == y_off.shape == (B, L, D)
    assert (y_on - y_off).abs().max().item() < 1e-2


@pytest.mark.skipif(not CUDA, reason="HCL kernel requires CUDA")
def test_vk_hcl_off_by_default() -> None:
    """
    A fresh HyenaInferenceEngine has use_hcl_kernel False -- the stock path.
    """
    torch.manual_seed(0)
    engine = HyenaInferenceEngine(layer_idx=0)
    assert engine.use_hcl_kernel is False

    z_pre, h, bias, poles, residues, t = _hcl_inputs(2048, torch.float32)
    y_default = _call(engine, z_pre, h, bias, 2048, poles, residues, t)

    explicit_off = HyenaInferenceEngine(layer_idx=0, use_hcl_kernel=False)
    y_explicit = _call(explicit_off, z_pre, h, bias, 2048, poles, residues, t)

    assert (y_default - y_explicit).abs().max().item() == 0.0


@pytest.mark.skipif(not CUDA, reason="HCL kernel requires CUDA")
def test_vk_hcl_predicate_skips_long_fir() -> None:
    """
    The branch matches only long_fir_threshold is None -- a set threshold
    routes through the stock depthwise-conv path.
    """
    torch.manual_seed(0)
    engine = HyenaInferenceEngine(layer_idx=0, use_hcl_kernel=True)
    z_pre, h, bias, poles, residues, t = _hcl_inputs(2048, torch.float32)

    y = _call(engine, z_pre, h, bias, 2048, poles, residues, t, long_fir_threshold=128)
    assert y.shape == (B, 2048, D)
