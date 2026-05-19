"""
Wiring tests for the use_hcs_kernel branch in HyenaInferenceEngine.parallel_fir.

These exercise the engine branch + hcs_conv adapter + Triton kernel together
by calling parallel_fir directly (no model weights needed). With the kernel
enabled, the gated short-filter cascade must reproduce the stock F.conv1d
path; the branch must not fire for the featurizer or the HCM-length cascade.
"""

import pytest
import torch
import torch.nn.functional as F

from vortex.model.engine import HyenaInferenceEngine

CUDA: bool = torch.cuda.is_available()

# evo2_7b HCS cascade shapes: D=4096, hcs_filter_groups=256, fir_length=7.
B, D, K, GROUPS = 1, 4096, 7, 256
DIMS: tuple[int, int, int, int, int] = (D, 32, D // 32, 16, GROUPS)
_CASCADE_KW: dict[str, bool | int | None] = dict(
    groups=GROUPS,
    gated_bias=False,
    column_split_hyena=False,
    dim_last=False,
    fir_length=K,
    gate=True,
)


def _hcs_inputs(L: int, dtype: torch.dtype):
    """
    Build (u, weight, bias) for an HCS cascade parallel_fir call.
    """
    u = torch.randn(B, 3 * D, L, dtype=dtype, device="cuda")
    weight = torch.randn(D, 1, K, dtype=dtype, device="cuda")
    bias = torch.randn(D, dtype=dtype, device="cuda")
    return u, weight, bias


@pytest.mark.skipif(not CUDA, reason="HCS kernel requires CUDA")
@pytest.mark.parametrize("L", [1024, 8192])
def test_vk_hcs_on_matches_baseline_fp32(L: int) -> None:
    """
    use_hcs_kernel on reproduces the stock parallel_fir HCS output in fp32.
    """
    torch.manual_seed(0)
    engine = HyenaInferenceEngine(layer_idx=0)
    u, weight, bias = _hcs_inputs(L, torch.float32)

    z_off, state_off = engine.parallel_fir(
        F.conv1d, u, weight, bias, L, DIMS, **_CASCADE_KW
    )

    engine.use_hcs_kernel = True
    z_on, state_on = engine.parallel_fir(
        F.conv1d, u, weight, bias, L, DIMS, **_CASCADE_KW
    )

    assert state_off is None and state_on is None
    assert z_on.shape == z_off.shape == (B, D, L)
    assert (z_on - z_off).abs().max().item() < 1e-3


@pytest.mark.skipif(not CUDA, reason="HCS kernel requires CUDA")
def test_vk_hcs_on_matches_baseline_bf16() -> None:
    """
    use_hcs_kernel on reproduces the stock HCS output in bf16, the inference dtype.
    """
    torch.manual_seed(0)
    engine = HyenaInferenceEngine(layer_idx=0)
    u, weight, bias = _hcs_inputs(4096, torch.bfloat16)

    z_off, _ = engine.parallel_fir(F.conv1d, u, weight, bias, 4096, DIMS, **_CASCADE_KW)

    engine.use_hcs_kernel = True
    z_on, _ = engine.parallel_fir(F.conv1d, u, weight, bias, 4096, DIMS, **_CASCADE_KW)

    torch.testing.assert_close(z_on, z_off, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not CUDA, reason="HCS kernel requires CUDA")
def test_vk_hcs_off_by_default() -> None:
    """
    A fresh HyenaInferenceEngine has use_hcs_kernel False, so parallel_fir
    takes the stock path with no behavioural change.
    """
    torch.manual_seed(0)
    engine = HyenaInferenceEngine(layer_idx=0)
    assert engine.use_hcs_kernel is False

    u, weight, bias = _hcs_inputs(2048, torch.float32)
    z_default, _ = engine.parallel_fir(
        F.conv1d, u, weight, bias, 2048, DIMS, **_CASCADE_KW
    )

    explicit_off = HyenaInferenceEngine(layer_idx=0, use_hcs_kernel=False)
    z_explicit, _ = explicit_off.parallel_fir(
        F.conv1d, u, weight, bias, 2048, DIMS, **_CASCADE_KW
    )

    # both took the stock path, so they are bitwise identical
    assert (z_default - z_explicit).abs().max().item() == 0.0


@pytest.mark.skipif(not CUDA, reason="HCS kernel requires CUDA")
def test_vk_hcs_predicate_skips_non_hcs_calls() -> None:
    """
    The branch matches only the gated short cascade -- not the featurizer
    (gate=False) and not the HCM-length cascade (fir_length >= 128).
    """
    torch.manual_seed(0)
    engine = HyenaInferenceEngine(layer_idx=0, use_hcs_kernel=True)

    # featurizer: gate=False, dim_last=True, no split -> branch unreachable
    u_feat = torch.randn(B, 1024, 3 * D, dtype=torch.float32, device="cuda")
    w_feat = torch.randn(3 * D, 1, 3, dtype=torch.float32, device="cuda")
    z_feat, _ = engine.parallel_fir(
        F.conv1d,
        u_feat,
        w_feat,
        None,
        1024,
        DIMS,
        groups=None,
        dim_last=True,
        fir_length=3,
        gate=False,
    )
    assert z_feat.shape == (B, 3 * D, 1024)

    # HCM-length cascade: fir_length=128 fails the `< 128` predicate, so the
    # call still routes through the stock fftconv_func path.
    u_hcm = torch.randn(B, 3 * D, 1024, dtype=torch.float32, device="cuda")
    w_hcm = torch.randn(D, 1, 128, dtype=torch.float32, device="cuda")
    b_hcm = torch.randn(D, dtype=torch.float32, device="cuda")
    kw = dict(_CASCADE_KW)
    kw["fir_length"] = 128
    z_hcm, _ = engine.parallel_fir(F.conv1d, u_hcm, w_hcm, b_hcm, 1024, DIMS, **kw)
    assert z_hcm.shape == (B, D, 1024)
