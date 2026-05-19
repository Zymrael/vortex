"""
Correctness and peak-memory tests for the HCL Triton kernels.

_hcl_compute_filter builds the modal filter h[d, l] = sum_s residues[d, s] *
exp(log_poles[d, s] * t[l]) without the (D, state_size, L) intermediate that
OOMs evo2_7b at L=131k; a peak-allocation test confirms that intermediate is
never built. _hcl_bias_residual_gate fuses the FFT-conv epilogue
(y + x1v * bias[:, None]) * x2. Each kernel is checked against the explicit
torch expression parallel_iir runs.
"""

import pytest
import torch

from vortex.ops.hcl_interface import _hcl_bias_residual_gate, _hcl_compute_filter

CUDA: bool = torch.cuda.is_available()


def _modal_filter_ref(
    residues: torch.Tensor, log_poles: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Reference modal filter -- compute_filter's (D, S, L) reduction, (D, L) out.
    """
    return (residues[:, :, None] * (log_poles[:, :, None] * t).exp()).sum(1)


def _hcl_inputs(
    D: int, L: int, S: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build (residues, log_poles, t) for a stable modal filter.

    log_poles are small and negative so exp(log_poles * t) decays over the
    sequence without overflowing -- a stable SSM, as in a trained evo2.
    """
    torch.manual_seed(0)
    residues = torch.randn(D, S, dtype=torch.float32, device="cuda")
    log_poles = -torch.rand(D, S, dtype=torch.float32, device="cuda") * (8.0 / L)
    t = torch.arange(L, dtype=torch.float32, device="cuda")
    return residues, log_poles, t


@pytest.mark.skipif(not CUDA, reason="HCL Triton kernel requires CUDA")
@pytest.mark.parametrize("L", [2048, 8192, 32768])
def test_hcl_compute_filter_matches_oracle(L: int) -> None:
    """
    The tiled compute_filter kernel matches the torch modal-filter reduction
    at evo2_7b shapes (D=4096, state_size=16).
    """
    residues, log_poles, t = _hcl_inputs(D=4096, L=L)

    h = _hcl_compute_filter(residues, log_poles, t)
    h_ref = _modal_filter_ref(residues, log_poles, t)

    assert h.shape == h_ref.shape == (4096, L)
    assert h.dtype == torch.float32
    max_diff = (h - h_ref).abs().max().item()
    mean_diff = (h - h_ref).abs().mean().item()
    assert max_diff < 1e-3, f"max_diff={max_diff:.2e}"
    assert mean_diff < 1e-4, f"mean_diff={mean_diff:.2e}"


@pytest.mark.skipif(not CUDA, reason="HCL Triton kernel requires CUDA")
def test_hcl_compute_filter_masks_ragged_tile() -> None:
    """
    The kernel masks (D, L) tiles that BLOCK_D x BLOCK_L does not divide evenly.
    """
    residues, log_poles, t = _hcl_inputs(D=100, L=300)

    h = _hcl_compute_filter(residues, log_poles, t)
    h_ref = _modal_filter_ref(residues, log_poles, t)
    assert (h - h_ref).abs().max().item() < 1e-3


@pytest.mark.skipif(not CUDA, reason="HCL Triton kernel requires CUDA")
def test_hcl_compute_filter_avoids_the_intermediate() -> None:
    """
    The kernel's peak allocation stays well below the reference path, which
    materialises the (D, state_size, L) intermediate -- direct evidence the
    tiled reduction never builds it.
    """
    residues, log_poles, t = _hcl_inputs(D=4096, L=32768)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _modal_filter_ref(residues, log_poles, t)
    torch.cuda.synchronize()
    ref_peak = torch.cuda.max_memory_allocated()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _hcl_compute_filter(residues, log_poles, t)
    torch.cuda.synchronize()
    kernel_peak = torch.cuda.max_memory_allocated()

    assert kernel_peak < ref_peak * 0.5, (
        f"kernel peak {kernel_peak / 1e9:.2f} GB is not below half the "
        f"reference peak {ref_peak / 1e9:.2f} GB"
    )


@pytest.mark.skipif(not CUDA, reason="HCL Triton kernel requires CUDA")
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("L", [2048, 8192, 32768])
def test_hcl_bias_residual_gate_matches_oracle(B: int, L: int) -> None:
    """
    The fused bias-residual-gate matches (y + x1v * bias[:, None]) * x2 at
    evo2_7b shapes.
    """
    torch.manual_seed(0)
    D = 4096
    y = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    x1v = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    bias = torch.randn(D, dtype=torch.float32, device="cuda")
    x2 = torch.randn(B, D, L, dtype=torch.float32, device="cuda")

    out = _hcl_bias_residual_gate(y, x1v, bias, x2)
    out_ref = (y + x1v * bias.unsqueeze(-1)) * x2

    assert out.shape == y.shape
    assert out.dtype == x1v.dtype
    max_diff = (out - out_ref).abs().max().item()
    mean_diff = (out - out_ref).abs().mean().item()
    assert max_diff < 1e-4, f"max_diff={max_diff:.2e}"
    assert mean_diff < 1e-5, f"mean_diff={mean_diff:.2e}"


@pytest.mark.skipif(not CUDA, reason="HCL Triton kernel requires CUDA")
def test_hcl_bias_residual_gate_masks_ragged_tile() -> None:
    """
    The kernel masks (D, L) tiles that BLOCK_D x BLOCK_L does not divide evenly.
    """
    torch.manual_seed(0)
    B, D, L = 1, 100, 300
    y = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    x1v = torch.randn(B, D, L, dtype=torch.float32, device="cuda")
    bias = torch.randn(D, dtype=torch.float32, device="cuda")
    x2 = torch.randn(B, D, L, dtype=torch.float32, device="cuda")

    out = _hcl_bias_residual_gate(y, x1v, bias, x2)
    out_ref = (y + x1v * bias.unsqueeze(-1)) * x2
    assert (out - out_ref).abs().max().item() < 1e-4
