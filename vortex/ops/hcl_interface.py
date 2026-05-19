# pyright: reportAttributeAccessIssue=none
"""
HCL -- Hyena Cascade Long.

Triton kernels for the long-filter (long_fir_threshold is None) FFT-conv path
of HyenaInferenceEngine.parallel_iir. HCL is the memory-unlock kernel: the
stock compute_filter materialises a (D, state_size, L) fp32 intermediate that
OOMs evo2_7b at L=131k.

This module currently provides _hcl_compute_filter -- the tiled modal-filter
build that does the state-size reduction in-register, so that intermediate
never exists.
"""

from typing import Callable

import torch
import triton
import triton.language as tl

# Autotuned search space for the (D, L) tile. Pure elementwise + a 16-term
# in-register reduction, so the winner is whichever tile best saturates
# bandwidth -- Triton benchmarks these once per (D, L) and caches the winner.
_COMPUTE_FILTER_CONFIGS: list[triton.Config] = [
    triton.Config({"BLOCK_D": 32, "BLOCK_L": 64}, num_warps=2),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 64}, num_warps=4),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 128}, num_warps=4),
    triton.Config({"BLOCK_D": 128, "BLOCK_L": 64}, num_warps=4),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 256}, num_warps=8),
    triton.Config({"BLOCK_D": 128, "BLOCK_L": 128}, num_warps=8),
]


@triton.autotune(configs=_COMPUTE_FILTER_CONFIGS, key=["D", "L"])
@triton.jit
def _hcl_compute_filter_kernel(
    residues_ptr,
    log_poles_ptr,
    t_ptr,
    h_ptr,
    D,
    L,
    S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """
    Modal filter: h[d, l] = sum_s residues[d, s] * exp(log_poles[d, s] * t[l]).

    One program covers a (BLOCK_D, BLOCK_L) tile of h. The state-size sum (S
    terms) runs in the fp32 register accumulator, so the (D, S, L) intermediate
    that OOMs the stock compute_filter at L=131k never exists. residues and
    log_poles are (D, S) row-major; t is (L,); h is (D, L) row-major.
    """
    pid_d = tl.program_id(0)
    pid_l = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_d = offs_d < D
    mask_l = offs_l < L

    t_tile = tl.load(t_ptr + offs_l, mask=mask_l, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_D, BLOCK_L), dtype=tl.float32)
    for s in tl.static_range(S):
        r_s = tl.load(residues_ptr + offs_d * S + s, mask=mask_d, other=0.0).to(
            tl.float32
        )
        lp_s = tl.load(log_poles_ptr + offs_d * S + s, mask=mask_d, other=0.0).to(
            tl.float32
        )
        acc += r_s[:, None] * tl.exp(lp_s[:, None] * t_tile[None, :])

    h_ptrs = h_ptr + offs_d[:, None] * L + offs_l[None, :]
    tl.store(h_ptrs, acc, mask=mask_d[:, None] & mask_l[None, :])


def _hcl_compute_filter(
    residues: torch.Tensor, log_poles: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Tiled modal-filter build -- the HCL compute_filter without the OOM.

    Computes h[d, l] = sum_s residues[d, s] * exp(log_poles[d, s] * t[l]), the
    (D, L) filter compute_filter builds, with the state-size sum done
    in-register so the (D, state_size, L) intermediate never exists.

    Args:
        residues (torch.Tensor): Modal residues, shape (D, S).
        log_poles (torch.Tensor): Modal log-poles, shape (D, S); negative for
                                  a stable (decaying) filter.
        t (torch.Tensor): Time index [0, 1, ..., L-1], shape (L,).

    Returns:
        torch.Tensor: The modal filter h, shape (D, L), fp32.

    Raises:
        ValueError: If residues and log_poles are not matching 2-D tensors,
                    or t is not 1-D.
    """
    if residues.dim() != 2 or log_poles.dim() != 2:
        raise ValueError(
            f"expected 2-D residues and log_poles, got {tuple(residues.shape)} "
            f"and {tuple(log_poles.shape)}"
        )
    if residues.shape != log_poles.shape:
        raise ValueError(
            f"residues {tuple(residues.shape)} and log_poles "
            f"{tuple(log_poles.shape)} must match"
        )
    if t.dim() != 1:
        raise ValueError(f"expected 1-D t, got {tuple(t.shape)}")

    D, S = residues.shape
    L: int = t.shape[0]

    residues = residues.contiguous().float()
    log_poles = log_poles.contiguous().float()
    t = t.contiguous().float()
    h: torch.Tensor = torch.empty(D, L, dtype=torch.float32, device=residues.device)

    # BLOCK_D / BLOCK_L are supplied by @triton.autotune; the grid is a
    # callable so it can read the chosen tile sizes from the winning config.
    grid: Callable[[triton.Config], tuple[int, int]] = lambda meta: (
        triton.cdiv(D, meta["BLOCK_D"]),
        triton.cdiv(L, meta["BLOCK_L"]),
    )
    _hcl_compute_filter_kernel[grid](residues, log_poles, t, h, D, L, S)
    return h
