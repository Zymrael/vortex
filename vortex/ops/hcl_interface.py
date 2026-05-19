# pyright: reportAttributeAccessIssue=none
"""
HCL -- Hyena Cascade Long.

Triton kernels for the long-filter (long_fir_threshold is None) FFT-conv path
of HyenaInferenceEngine.parallel_iir. HCL is the memory-unlock kernel: the
stock compute_filter materialises a (D, state_size, L) fp32 intermediate that
OOMs evo2_7b at L=131k.

It provides _hcl_compute_filter -- the tiled modal-filter build that does the
state-size reduction in-register so that intermediate never exists -- and
_hcl_bias_residual_gate, the fused FFT-conv epilogue (y + x1v * bias) * x2.
"""

from typing import Callable

import torch
import triton
import triton.language as tl

# Autotuned 2-D BLOCK_D x BLOCK_L tile, shared by both kernels -- both are
# memory-bound elementwise work over a (D, L) grid, so the winner is whichever
# tile best saturates bandwidth, benchmarked once per (D, L) and cached.
_TILE_CONFIGS: list[triton.Config] = [
    triton.Config({"BLOCK_D": 32, "BLOCK_L": 64}, num_warps=2),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 64}, num_warps=4),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 128}, num_warps=4),
    triton.Config({"BLOCK_D": 128, "BLOCK_L": 64}, num_warps=4),
    triton.Config({"BLOCK_D": 64, "BLOCK_L": 256}, num_warps=8),
    triton.Config({"BLOCK_D": 128, "BLOCK_L": 128}, num_warps=8),
]


@triton.autotune(configs=_TILE_CONFIGS, key=["D", "L"])
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


@triton.autotune(configs=_TILE_CONFIGS, key=["D", "L"])
@triton.jit
def _hcl_bias_residual_gate_kernel(
    y_ptr,
    x1v_ptr,
    bias_ptr,
    x2_ptr,
    out_ptr,
    D,
    L,
    stride_b,
    stride_d,
    stride_l,
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """
    Bias-residual + gate: out[b,d,l] = (y[b,d,l] + x1v[b,d,l]*bias[d]) * x2[b,d,l].

    One program covers a (BLOCK_D, BLOCK_L) tile of one batch element. y, x1v,
    x2 and out share a contiguous (B, D, L) layout; bias is per-channel, shape
    (D,), broadcast over batch and length. The fp32 accumulator is cast to
    out's dtype on the store -- parallel_iir's y.to(x1v.dtype) cast.
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_l = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_d = offs_d < D
    mask_l = offs_l < L
    mask = mask_d[:, None] & mask_l[None, :]

    offs = pid_b * stride_b + offs_d[:, None] * stride_d + offs_l[None, :] * stride_l
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x1v = tl.load(x1v_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    acc = (y + x1v * bias[:, None]) * x2
    tl.store(out_ptr + offs, acc, mask=mask)


def _hcl_bias_residual_gate(
    y: torch.Tensor, x1v: torch.Tensor, bias: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """
    Fused bias-residual + gate -- the HCL FFT-conv epilogue.

    Computes (y + x1v * bias[:, None]) * x2 and writes it at x1v's dtype --
    parallel_iir's post-conv `y = (y + x1v * D.unsqueeze(-1)) * x2`, fusing the
    broadcast multiply, the residual add, the gate and the dtype cast into one
    Triton launch.

    Args:
        y (torch.Tensor): The irfft output, shape (B, D, L).
        x1v (torch.Tensor): The conv input, shape (B, D, L); its dtype is the
                            output dtype.
        bias (torch.Tensor): Per-channel skip gain, shape (D,), broadcast over
                             batch and length.
        x2 (torch.Tensor): The post-gate stream, shape (B, D, L).

    Returns:
        torch.Tensor: (y + x1v * bias[:, None]) * x2, shape (B, D, L), x1v's
                      dtype.

    Raises:
        ValueError: If y, x1v and x2 are not matching 3-D tensors, or bias is
                    not 1-D of length D.
    """
    if y.dim() != 3 or x1v.dim() != 3 or x2.dim() != 3:
        raise ValueError(
            f"expected 3-D y, x1v, x2, got {tuple(y.shape)}, "
            f"{tuple(x1v.shape)}, {tuple(x2.shape)}"
        )
    if not (y.shape == x1v.shape == x2.shape):
        raise ValueError(
            f"y {tuple(y.shape)}, x1v {tuple(x1v.shape)}, x2 {tuple(x2.shape)} "
            f"must all match"
        )

    B, D, L = x1v.shape
    if bias.dim() != 1 or bias.shape[0] != D:
        raise ValueError(f"bias {tuple(bias.shape)} must be 1-D of length D={D}")

    y = y.contiguous()
    x1v = x1v.contiguous()
    x2 = x2.contiguous()
    bias = bias.contiguous()
    out: torch.Tensor = torch.empty_like(x1v)

    # BLOCK_D / BLOCK_L are supplied by @triton.autotune; the grid is a
    # callable so it can read the chosen tile sizes from the winning config.
    grid: Callable[[triton.Config], tuple[int, int, int]] = lambda meta: (
        B,
        triton.cdiv(D, meta["BLOCK_D"]),
        triton.cdiv(L, meta["BLOCK_L"]),
    )
    _hcl_bias_residual_gate_kernel[grid](
        y,
        x1v,
        bias,
        x2,
        out,
        D,
        L,
        x1v.stride(0),
        x1v.stride(1),
        x1v.stride(2),
    )
    return out
