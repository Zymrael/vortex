"""
Correctness tests for the HCS depthwise causal conv Triton kernel.

The oracle is F.conv1d on the depthwise causal path (left-pad by
fir_length - 1, trim to L) -- exactly the convolution
HyenaInferenceEngine.parallel_fir applies in its fir_length < 128 branch.
"""

import pytest
import torch
import torch.nn.functional as F

from vortex.ops.hcs_interface import hcs_depthwise_conv

CUDA: bool = torch.cuda.is_available()


def _conv1d_ref(u: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Reference depthwise causal conv via F.conv1d, trimmed to length L.
    """
    fir_length = weight.shape[-1]
    L = u.shape[-1]
    return F.conv1d(
        u, weight, bias=None, stride=1, padding=fir_length - 1, groups=u.shape[1]
    )[..., :L]


@pytest.mark.skipif(not CUDA, reason="HCS Triton kernel requires CUDA")
@pytest.mark.parametrize("L", [1024, 8192, 32768])
@pytest.mark.parametrize("fir_length", [3, 7])
def test_hcs_conv_matches_conv1d(L: int, fir_length: int) -> None:
    """
    The HCS Triton depthwise conv matches F.conv1d at evo2_7b shapes.
    """
    torch.manual_seed(0)
    B, D = 1, 4096
    u = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    weight = torch.randn(D, 1, fir_length, dtype=torch.float32, device="cuda")

    z = hcs_depthwise_conv(u, weight)
    z_ref = _conv1d_ref(u, weight)

    max_diff = (z - z_ref).abs().max().item()
    mean_diff = (z - z_ref).abs().mean().item()
    assert max_diff < 1e-3, f"max_diff={max_diff:.2e}"
    assert mean_diff < 1e-4, f"mean_diff={mean_diff:.2e}"


@pytest.mark.skipif(not CUDA, reason="HCS Triton kernel requires CUDA")
def test_hcs_conv_is_causal() -> None:
    """
    Output position t depends only on inputs at or before t.
    """
    torch.manual_seed(0)
    B, D, L, fir_length = 1, 64, 256, 7
    u = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    weight = torch.randn(D, 1, fir_length, dtype=torch.float32, device="cuda")

    z = hcs_depthwise_conv(u, weight)
    u_perturbed = u.clone()
    u_perturbed[..., L // 2] += 100.0
    z_perturbed = hcs_depthwise_conv(u_perturbed, weight)

    # positions strictly before L//2 must be unchanged
    before = (z[..., : L // 2] - z_perturbed[..., : L // 2]).abs().max().item()
    after = (z[..., L // 2 :] - z_perturbed[..., L // 2 :]).abs().max().item()
    assert before == 0.0, (
        f"non-causal: positions before the perturbation changed by {before:.2e}"
    )
    assert after > 0.0, "perturbation had no effect on later positions"


@pytest.mark.skipif(not CUDA, reason="HCS Triton kernel requires CUDA")
@pytest.mark.parametrize("D", [16, 4096, 4100])
def test_hcs_conv_ragged_channels(D: int) -> None:
    """
    The kernel masks channel tiles that do not divide BLOCK_D evenly.
    """
    torch.manual_seed(0)
    B, L, fir_length = 1, 1024, 7
    u = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    weight = torch.randn(D, 1, fir_length, dtype=torch.float32, device="cuda")

    z = hcs_depthwise_conv(u, weight)
    z_ref = _conv1d_ref(u, weight)
    assert (z - z_ref).abs().max().item() < 1e-3
