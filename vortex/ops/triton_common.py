"""
Shared autotune configs and grid helpers for the HC{S,M,L} Triton kernels.

Pieces here are kept intentionally small. Only put a config or helper here
when it is genuinely identical across kernels -- any kernel that needs a
different tile or grid declares its own. Each @triton.autotune-decorated
kernel keeps its own benchmark cache, so sharing the config list is a
syntactic convenience, not a performance contract.
"""

from collections.abc import Callable

import triton

# 2-D (BLOCK_D, BLOCK_L) tile sweep for memory-bound elementwise kernels over
# a (D, L) plane. Re-benchmarked per shape key by each decorated kernel.
BDL_TILE_CONFIGS: list[triton.Config] = [
    triton.Config({"BLOCK_D": 32, "BLOCK_L": 64}, num_warps=2),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 64}, num_warps=4),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 128}, num_warps=4),
    triton.Config({"BLOCK_D": 128, "BLOCK_L": 64}, num_warps=4),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 256}, num_warps=8),
    triton.Config({"BLOCK_D": 128, "BLOCK_L": 128}, num_warps=8),
]


def bdl_grid_3d(B: int, D: int, L: int) -> Callable[[dict], tuple[int, int, int]]:
    """
    Standard (B, cdiv(D, BLOCK_D), cdiv(L, BLOCK_L)) grid for the HC bias-residual
    and HCS conv kernels. Reads tile sizes from the autotune meta-dict.
    """
    return lambda meta: (
        B,
        triton.cdiv(D, meta["BLOCK_D"]),
        triton.cdiv(L, meta["BLOCK_L"]),
    )


def bdl_grid_2d(D: int, L: int) -> Callable[[dict], tuple[int, int]]:
    """
    2-D (cdiv(D, BLOCK_D), cdiv(L, BLOCK_L)) grid for kernels without a batch
    axis (the HCL filter build).
    """
    return lambda meta: (
        triton.cdiv(D, meta["BLOCK_D"]),
        triton.cdiv(L, meta["BLOCK_L"]),
    )
