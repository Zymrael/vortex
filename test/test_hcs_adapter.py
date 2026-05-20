"""
Tests for the hcs_conv adapter -- the drop-in for the gated HCS branch of
HyenaInferenceEngine.parallel_fir.

The oracle is a pure-PyTorch transcription of that branch (engine.py, the
fir_length < 128 path plus the gate). hcs_conv must reproduce it exactly,
so the use_hcs_kernel flag off vs on is a behavioural no-op.
"""

import pytest
import torch
import torch.nn.functional as F

from vortex.ops.hcs_interface import hcs_conv


def _hcs_branch_ref(x1, x2, v, weight, bias, gated_bias, padding_mask):
    """
    Pure-PyTorch reference for the gated HCS branch of parallel_fir.
    """
    D, L = v.shape[1], v.shape[2]
    u = x1 * v
    z = F.conv1d(
        u.float(),
        weight.float(),
        bias=None,
        stride=1,
        padding=weight.shape[-1] - 1,
        groups=D,
    )[..., :L]
    z = z.to(u.dtype)
    if bias is not None:
        z = z + bias[None, :, None] * u if gated_bias else z + bias[None, :, None]
    if isinstance(padding_mask, torch.Tensor):
        z = z * padding_mask[:, None]
    return x2 * z


@pytest.mark.gpu
@pytest.mark.parametrize("L", [1024, 8192])
@pytest.mark.parametrize("with_bias", [True, False])
def test_hcs_conv_matches_engine_branch(L: int, with_bias: bool) -> None:
    """
    hcs_conv reproduces the flag-off parallel_fir HCS branch in fp32.
    """
    torch.manual_seed(0)
    B, D, K = 1, 4096, 7
    x1 = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    x2 = torch.randn_like(x1)
    v = torch.randn_like(x1)
    weight = torch.randn(D, 1, K, dtype=torch.float32, device="cuda")
    bias = torch.randn(D, dtype=torch.float32, device="cuda") if with_bias else None

    z = hcs_conv(x1, x2, v, weight, bias)
    z_ref = _hcs_branch_ref(x1, x2, v, weight, bias, False, None)
    assert (z - z_ref).abs().max().item() < 1e-3


@pytest.mark.gpu
def test_hcs_conv_gated_bias() -> None:
    """
    The gated_bias=True path applies the bias multiplicatively.
    """
    torch.manual_seed(0)
    B, D, L, K = 1, 4096, 2048, 7
    x1 = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    x2 = torch.randn_like(x1)
    v = torch.randn_like(x1)
    weight = torch.randn(D, 1, K, dtype=torch.float32, device="cuda")
    bias = torch.randn(D, dtype=torch.float32, device="cuda")

    z = hcs_conv(x1, x2, v, weight, bias, gated_bias=True)
    z_ref = _hcs_branch_ref(x1, x2, v, weight, bias, True, None)
    assert (z - z_ref).abs().max().item() < 1e-3


@pytest.mark.gpu
def test_hcs_conv_padding_mask() -> None:
    """
    A padding_mask tensor zeros masked positions in the output.
    """
    torch.manual_seed(0)
    B, D, L, K = 1, 256, 1024, 7
    x1 = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    x2 = torch.randn_like(x1)
    v = torch.randn_like(x1)
    weight = torch.randn(D, 1, K, dtype=torch.float32, device="cuda")
    mask = torch.ones(B, L, dtype=torch.float32, device="cuda")
    mask[:, L // 2 :] = 0.0

    z = hcs_conv(x1, x2, v, weight, None, padding_mask=mask)
    z_ref = _hcs_branch_ref(x1, x2, v, weight, None, False, mask)
    assert (z - z_ref).abs().max().item() < 1e-3
    assert z[..., L // 2 :].abs().max().item() == 0.0


@pytest.mark.gpu
def test_hcs_conv_bf16() -> None:
    """
    hcs_conv matches the engine branch in bf16, the real inference dtype.
    """
    torch.manual_seed(0)
    B, D, L, K = 1, 4096, 2048, 7
    x1 = torch.randn(B, D, L, dtype=torch.bfloat16, device="cuda")
    x2 = torch.randn_like(x1)
    v = torch.randn_like(x1)
    weight = torch.randn(D, 1, K, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(D, dtype=torch.bfloat16, device="cuda")

    z = hcs_conv(x1, x2, v, weight, bias)
    z_ref = _hcs_branch_ref(x1, x2, v, weight, bias, False, None)
    torch.testing.assert_close(z, z_ref, rtol=2e-2, atol=2e-2)
