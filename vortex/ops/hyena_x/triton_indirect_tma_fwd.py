# Copyright (c) 2024, Michael Poli.

import time 

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import numpy as np


from triton_indirect_fwd import *


@triton.autotune(
    configs=[
        triton.Config(kwargs={"_f": 16}, num_stages=1, num_warps=1, num_ctas=1, maxnreg=128),
        triton.Config(kwargs={"_f": 16}, num_stages=2, num_warps=1, num_ctas=1, maxnreg=128),
    ],
    key=['D']
)
@triton.jit
def hyena_x_fwd_bdl_tma_kernel(
    q_ptr, k_ptr, v_ptr, hq_ptr, hk_ptr, hv_ptr, o_ptr, L: tl.constexpr, L_h: tl.constexpr, D: tl.constexpr, BL: tl.constexpr, NBL: tl.constexpr, _f: tl.constexpr
):
    """
    Hyena-X forward with indirect im2col convolution. Processes each channel and batch index independently. 
    """
    i_b, i_d, i_l = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q_ptr, k_ptr, v_ptr, o_ptr = move_to_base_offset(q_ptr, k_ptr, v_ptr, o_ptr, L, D, BL)

    h_q_ptr = hq_ptr + L_h * i_d + tl.arange(0, BL) - (BL - L_h)
    h_k_ptr = hk_ptr + L_h * i_d + tl.arange(0, BL) - (BL - L_h)
    h_v_ptr = hv_ptr + L_h * i_d + tl.arange(0, BL) - (BL - L_h)

    mask = (tl.arange(0, BL) - (BL - L_h)) >= 0

    h_q, h_k, h_v = tl.load(h_q_ptr, mask=mask, other=0.0), tl.load(h_k_ptr, mask=mask, other=0.0), tl.load(h_v_ptr, mask=mask, other=0.0)
    
    b_q = load_strided_block(q_ptr, BL, L, D, NBL)
    b_k = load_strided_block(k_ptr, BL, L, D, NBL)
    b_v = load_strided_block(v_ptr, BL, L, D, NBL)
        
    if BL >= 16:
        b_hq = tl.broadcast_to(h_q[None, :], (BL, BL))
        b_hk = tl.broadcast_to(h_k[None, :], (BL, BL))
        b_hv = tl.broadcast_to(h_v[None, :], (BL, BL))

        z_q = tl.dot(b_hq, b_q) 
        z_q = tl.sum(z_q, axis=0) / BL
        z_k = tl.dot(b_hk, b_k) 
        z_k = tl.sum(z_k, axis=0) / BL
        z_v = tl.dot(b_hv, b_v) 
        z_v = tl.sum(z_v, axis=0) / BL

        b_o = z_q * z_k * z_v 
        tl.store(o_ptr + tl.arange(0, BL), b_o, boundary_check=(0))

    else:
        b_hq = tl.broadcast_to(h_q[:, None], (BL, BL))
        b_hk = tl.broadcast_to(h_k[:, None], (BL, BL))
        b_hv = tl.broadcast_to(h_v[:, None], (BL, BL))

        z_q = b_hq * b_q 
        z_q = tl.sum(z_q, axis=0) 
        z_k = b_hk * b_k 
        z_k = tl.sum(z_k, axis=0) 
        z_v = b_hv * b_v 
        z_v = tl.sum(z_v, axis=0) 

        b_o = z_q * z_k * z_v 
        tl.store(o_ptr + tl.arange(0, BL), b_o, boundary_check=(0))



def hyena_x_bdl_tma_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor, 
    block_size: int = 16,
):
    hq, hk, hv = h.split([h.shape[0] // 3, h.shape[0] // 3, h.shape[0] // 3], dim=0)

    B, D, L = q.shape 
    L_h = h.shape[-1]

    block_size = max(16, triton.next_power_of_2(L_h))
    
    BL = block_size
    NBL = L // BL 
    grid = (B, D, NBL) 
    o = torch.empty((B, D, L), device=q.device, dtype=q.dtype)
    if h.shape[1] == 1: h = h.squeeze(1)

    TMA_STORE_SIZE = 128
    desc_o = np.empty(TMA_STORE_SIZE, dtype=np.float32)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(o.data_ptr(), BL, BL, BL, BL, o.element_size(), desc_o)
    desc_o = torch.tensor(desc_o, device='cuda')


    hyena_x_fwd_bdl_tma_kernel[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        hq_ptr=hq,
        hk_ptr=hk,
        hv_ptr=hv,
        o_ptr=o,
        L=L,
        L_h=L_h,
        D=D,
        BL=BL,
        NBL=NBL,
    )
    return o