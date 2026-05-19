"""
Wiring tests for the use_hcm_kernel branch in HyenaInferenceEngine.parallel_fir.

These exercise the fir_length >= 128 dispatch by calling parallel_fir directly
(no model weights needed). With the kernel enabled, the HCM cascade must
reproduce the stock fftconv_func output; the branch must not fire for the
HCS-length cascade.
"""

import pytest
import torch
import torch.nn.functional as F

from vortex.model.engine import HyenaInferenceEngine

CUDA: bool = torch.cuda.is_available()

# evo2_7b HCM cascade shapes: D=4096, fir_length=128.
B, D, K, GROUPS = 1, 4096, 128, 256
DIMS: tuple[int, int, int, int, int] = (D, 32, D // 32, 16, GROUPS)
_CASCADE_KW: dict[str, bool | int | None] = dict(
    groups=GROUPS,
    gated_bias=False,
    column_split_hyena=False,
    dim_last=False,
    fir_length=K,
    gate=True,
)


def _hcm_inputs(L: int, dtype: torch.dtype):
    """
    Build (u, weight, bias) for an HCM cascade parallel_fir call.
    """
    u = torch.randn(B, 3 * D, L, dtype=dtype, device="cuda")
    weight = torch.randn(D, 1, K, dtype=dtype, device="cuda")
    bias = torch.randn(D, dtype=dtype, device="cuda")
    return u, weight, bias


@pytest.mark.skipif(not CUDA, reason="HCM kernel requires CUDA")
@pytest.mark.parametrize("L", [2048, 8192])
def test_vk_hcm_on_matches_baseline_fp32(L: int) -> None:
    """
    use_hcm_kernel on reproduces the stock parallel_fir HCM output in fp32.
    """
    torch.manual_seed(0)
    engine = HyenaInferenceEngine(layer_idx=0)
    u, weight, bias = _hcm_inputs(L, torch.float32)

    z_off, _ = engine.parallel_fir(F.conv1d, u, weight, bias, L, DIMS, **_CASCADE_KW)

    engine.use_hcm_kernel = True
    z_on, _ = engine.parallel_fir(F.conv1d, u, weight, bias, L, DIMS, **_CASCADE_KW)

    assert z_on.shape == z_off.shape == (B, D, L)
    assert (z_on - z_off).abs().max().item() < 1e-2


@pytest.mark.skipif(not CUDA, reason="HCM kernel requires CUDA")
def test_vk_hcm_on_matches_baseline_bf16() -> None:
    """
    use_hcm_kernel on reproduces the stock HCM output in bf16, the inference dtype.
    """
    torch.manual_seed(0)
    engine = HyenaInferenceEngine(layer_idx=0)
    u, weight, bias = _hcm_inputs(2048, torch.bfloat16)

    z_off, _ = engine.parallel_fir(F.conv1d, u, weight, bias, 2048, DIMS, **_CASCADE_KW)

    engine.use_hcm_kernel = True
    z_on, _ = engine.parallel_fir(F.conv1d, u, weight, bias, 2048, DIMS, **_CASCADE_KW)

    torch.testing.assert_close(z_on, z_off, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not CUDA, reason="HCM kernel requires CUDA")
def test_vk_hcm_off_by_default() -> None:
    """
    A fresh HyenaInferenceEngine has use_hcm_kernel False, so parallel_fir
    takes the stock fftconv_func path with no behavioural change.
    """
    torch.manual_seed(0)
    engine = HyenaInferenceEngine(layer_idx=0)
    assert engine.use_hcm_kernel is False

    u, weight, bias = _hcm_inputs(2048, torch.float32)
    z_default, _ = engine.parallel_fir(F.conv1d, u, weight, bias, 2048, DIMS, **_CASCADE_KW)

    explicit_off = HyenaInferenceEngine(layer_idx=0, use_hcm_kernel=False)
    z_explicit, _ = explicit_off.parallel_fir(F.conv1d, u, weight, bias, 2048, DIMS, **_CASCADE_KW)

    # both took the stock path, so they are bitwise identical
    assert (z_default - z_explicit).abs().max().item() == 0.0


@pytest.mark.skipif(not CUDA, reason="HCM kernel requires CUDA")
def test_vk_hcm_predicate_skips_hcs_calls() -> None:
    """
    The branch matches only the fir_length >= 128 cascade -- an HCS-length
    call (fir_length=7) still routes through the stock F.conv1d path.
    """
    torch.manual_seed(0)
    engine = HyenaInferenceEngine(layer_idx=0, use_hcm_kernel=True)

    u = torch.randn(B, 3 * D, 1024, dtype=torch.float32, device="cuda")
    weight = torch.randn(D, 1, 7, dtype=torch.float32, device="cuda")
    bias = torch.randn(D, dtype=torch.float32, device="cuda")
    kw = dict(_CASCADE_KW)
    kw["fir_length"] = 7
    z, _ = engine.parallel_fir(F.conv1d, u, weight, bias, 1024, DIMS, **kw)
    assert z.shape == (B, D, 1024)
