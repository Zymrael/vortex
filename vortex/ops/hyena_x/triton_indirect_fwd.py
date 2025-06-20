# Copyright (c) 2024, Michael Poli.

import time 

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


#########################
# Utilities
#########################


@triton.jit
def load_strided_block(
    x_ptr, BL:tl.constexpr, L:tl.constexpr, D:tl.constexpr, NBL:tl.constexpr
):
    i_l = tl.program_id(2)
    i, j = tl.arange(0, BL), tl.arange(0, BL)
    flatten_map = i[:, None] + j[None, :] - BL + 1 
    mask = flatten_map >= 0 if i_l == 0 else flatten_map < L
    x_ptr_offset = x_ptr + flatten_map
    b_x = tl.load(x_ptr_offset, mask=mask, other=0.0)
    return b_x


@triton.jit
def load_block(
    x_ptr, BL:tl.constexpr, L:tl.constexpr
):
    i = tl.arange(0, BL)
    mask = i < L
    x_ptr_offset = x_ptr + i
    b_x = tl.load(x_ptr_offset, mask=mask, other=0.0)
    return b_x


@triton.jit
def shift_d_bdl(
    x_ptr, L: tl.constexpr
):
    x_ptr = x_ptr + L   
    return x_ptr 


@triton.jit
def move_to_base_offset(
    q_ptr, k_ptr, v_ptr, o_ptr, L: tl.constexpr, D: tl.constexpr, BL: tl.constexpr
):
    i_b, i_d, i_l = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qkv_offset = i_b * L * D + i_d * L + i_l * BL
    q_ptr, k_ptr, v_ptr, o_ptr = q_ptr + qkv_offset, k_ptr + qkv_offset, v_ptr + qkv_offset, o_ptr + qkv_offset
    return q_ptr, k_ptr, v_ptr, o_ptr
    

@triton.jit
def move_to_base_offset_bd(
    q_ptr, k_ptr, v_ptr, o_ptr, L: tl.constexpr, D: tl.constexpr, BL: tl.constexpr, BD: tl.constexpr
):
    i_b, i_d, i_l = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qkv_offset = i_b * L * D + i_d * L * BD + i_l * BL
    q_ptr, k_ptr, v_ptr, o_ptr = q_ptr + qkv_offset, k_ptr + qkv_offset, v_ptr + qkv_offset, o_ptr + qkv_offset
    return q_ptr, k_ptr, v_ptr, o_ptr


#########################
# Kernels
#########################


@triton.autotune(
    configs=[
        triton.Config(kwargs={"_f": 16}, num_stages=1, num_warps=1, num_ctas=1, maxnreg=128),
        triton.Config(kwargs={"_f": 16}, num_stages=2, num_warps=1, num_ctas=1, maxnreg=128),
    ],
    key=['D']
)
@triton.jit
def hyena_x_fwd_bdl_kernel(
    q_ptr, k_ptr, v_ptr, hq_ptr, hk_ptr, hv_ptr, o_ptr, L: tl.constexpr, L_h: tl.constexpr, D: tl.constexpr, BL: tl.constexpr, NBL: tl.constexpr, _f: tl.constexpr
):
    """
    Hyena-X forward with indirect convolution. Processes each channel and batch index independently. 
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
        z_q = tl.sum(z_q, axis=0) / BL
        z_k = b_hk * b_k 
        z_k = tl.sum(z_k, axis=0) / BL
        z_v = b_hv * b_v 
        z_v = tl.sum(z_v, axis=0) / BL

        b_o = z_q * z_k * z_v 
        tl.store(o_ptr + tl.arange(0, BL), b_o, boundary_check=(0))


@triton.autotune(
    configs=[
        triton.Config(kwargs={"_f": 16}, num_stages=1, num_warps=1, num_ctas=1, maxnreg=128),
        triton.Config(kwargs={"_f": 16}, num_stages=2, num_warps=1, num_ctas=1, maxnreg=128),
    ],
    key=['D']
)
@triton.jit
def hyena_x_fwd_bdl_dchunk_kernel(
    q_ptr, k_ptr, v_ptr, hq_ptr, hk_ptr, hv_ptr, o_ptr, 
    L: tl.constexpr, L_h: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BL: tl.constexpr, NBL: tl.constexpr, _f: tl.constexpr
):
    """
    Hyena-X forward with indirect convolution.
    """

    i_b, i_d, i_l = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q_ptr, k_ptr, v_ptr, o_ptr = move_to_base_offset_bd(q_ptr, k_ptr, v_ptr, o_ptr, L, D, BL, BD)

    h_q_ptr = hq_ptr + L_h * i_d + tl.arange(0, BL) - (BL - L_h)
    h_k_ptr = hk_ptr + L_h * i_d + tl.arange(0, BL) - (BL - L_h)
    h_v_ptr = hv_ptr + L_h * i_d + tl.arange(0, BL) - (BL - L_h)

    mask = (tl.arange(0, BL) - (BL - L_h)) >= 0
    b_o = tl.zeros((BD, BL), dtype=tl.float32)
    
    for d in range(BD):

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

            b_o_ = z_q * z_k * z_v 
            b_o = tl.where(tl.arange(0, BD)[:, None] == d, b_o_, b_o)

        else:
            b_hq = tl.broadcast_to(h_q[:, None], (BL, BL))
            b_hk = tl.broadcast_to(h_k[:, None], (BL, BL))
            b_hv = tl.broadcast_to(h_v[:, None], (BL, BL))

            z_q = b_hq * b_q 
            z_q = tl.sum(z_q, axis=0) / BL
            z_k = b_hk * b_k 
            z_k = tl.sum(z_k, axis=0) / BL
            z_v = b_hv * b_v 
            z_v = tl.sum(z_v, axis=0) / BL

            b_o = z_q * z_k * z_v 
            b_o = tl.where(tl.arange(0, BD)[:, None] == d, b_o, b_o)

        h_q_ptr, h_k_ptr, h_v_ptr = shift_d_bdl(h_q_ptr, L_h), shift_d_bdl(h_k_ptr, L_h), shift_d_bdl(h_v_ptr, L_h)
        q_ptr, k_ptr, v_ptr = shift_d_bdl(q_ptr, L), shift_d_bdl(k_ptr, L), shift_d_bdl(v_ptr, L)

    o_offsets = tl.arange(0, BD)[:, None] * L + tl.arange(0, BL)[None, :]
    tl.store(o_ptr + o_offsets, b_o)



@triton.autotune(
    configs=[
        triton.Config(kwargs={"_f": 16}, num_stages=1, num_warps=1, num_ctas=1, maxnreg=128),
        triton.Config(kwargs={"_f": 16}, num_stages=2, num_warps=1, num_ctas=1, maxnreg=128),
    ],
    key=['D']
)
@triton.jit
def hyena_x_fwd_bdl_dchunk_independent_kernel(
    x_ptr, h_ptr, o_ptr, 
    L: tl.constexpr, L_h: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BL: tl.constexpr, NBL: tl.constexpr, _f: tl.constexpr
):
    """
    Indirect convolution, applied independently to each feature group.
    """

    i_b, i_d, i_l = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offset = i_b * L * D + i_d * L * BD + i_l * BL
    x_ptr, o_ptr = x_ptr + offset, o_ptr + offset
    
    h_ptr = h_ptr + L_h * i_d + tl.arange(0, BL) - (BL - L_h)

    mask = (tl.arange(0, BL) - (BL - L_h)) >= 0
    b_o = tl.zeros((BD, BL), dtype=tl.float32)
    
    for d in range(BD):
        h = tl.load(h_ptr, mask=mask, other=0.0)
        b_x = load_strided_block(x_ptr, BL, L, D, NBL)
            
        if BL >= 16:
            b_h = tl.broadcast_to(h[None, :], (BL, BL))
            z = tl.dot(b_h, b_x)
            z = tl.sum(z, axis=0) / BL
            b_o = tl.where(tl.arange(0, BD)[:, None] == d, z, b_o)
        else:
            b_h = tl.broadcast_to(h[:, None], (BL, BL))
            z = b_h * b_x 
            z = tl.sum(z, axis=0) / BL 
            b_o = tl.where(tl.arange(0, BD)[:, None] == d, z, b_o)

        h_ptr, x_ptr = shift_d_bdl(h_ptr, L_h), shift_d_bdl(x_ptr, L)

    o_offsets = tl.arange(0, BD)[:, None] * L + tl.arange(0, BL)[None, :]
    tl.store(o_ptr + o_offsets, b_o)


#########################
# Kernel wrapper
#########################


def hyena_x_bdl_fwd(
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

    hyena_x_fwd_bdl_kernel[grid](
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



def hyena_x_bdl_dchunk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor, 
    block_size: int = 16,
    dchunk_size: int = 16,
):
    hq, hk, hv = h.split([h.shape[0] // 3, h.shape[0] // 3, h.shape[0] // 3], dim=0)

    B, D, L = q.shape 
    L_h = h.shape[-1]

    block_size = max(16, triton.next_power_of_2(L_h))

    assert L % block_size == 0, "sequence length must be divisible by block size"
    assert D % dchunk_size == 0, "channel dimension must be divisible by dchunk size"
    
    BL = block_size
    BD = dchunk_size
    NBL = L // BL
    NBD = D // BD
    grid = (B, NBD, NBL)
    o = torch.empty((B, D, L), device=q.device, dtype=q.dtype)
    if h.shape[1] == 1: h = h.squeeze(1)

    hyena_x_fwd_bdl_dchunk_kernel[grid](
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
        BD=BD,
        BL=BL,
        NBL=NBL,
    )
    return o


def hyena_x_bdl_dchunk_independent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor, 
    block_size: int = 16,
    dchunk_size: int = 16,
):
    hq, hk, hv = h.split([h.shape[0] // 3, h.shape[0] // 3, h.shape[0] // 3], dim=0)

    B, D, L = q.shape 
    L_h = h.shape[-1]

    block_size = max(block_size, triton.next_power_of_2(L_h))

    assert L % block_size == 0, "sequence length must be divisible by block size"
    assert D % dchunk_size == 0, "channel dimension must be divisible by dchunk size"
    
    BL = block_size
    BD = dchunk_size
    NBL = L // BL
    NBD = D // BD
    grid = (B, NBD, NBL)
    oq = torch.empty((B, D, L), device=q.device, dtype=q.dtype)
    ok = torch.empty((B, D, L), device=q.device, dtype=q.dtype)
    ov = torch.empty((B, D, L), device=q.device, dtype=q.dtype)

    if h.shape[1] == 1: h = h.squeeze(1)

    hyena_x_fwd_bdl_dchunk_independent_kernel[grid](
        x_ptr=q,
        h_ptr=hq,
        o_ptr=oq,
        L=L,
        L_h=L_h,
        D=D,
        BD=BD,
        BL=BL,
        NBL=NBL,
    )
    hyena_x_fwd_bdl_dchunk_independent_kernel[grid](
        x_ptr=k,
        h_ptr=hk,
        o_ptr=ok,
        L=L,
        L_h=L_h,
        D=D,
        BD=BD,
        BL=BL,
        NBL=NBL,
    )
    hyena_x_fwd_bdl_dchunk_independent_kernel[grid](
        x_ptr=v,
        h_ptr=hv,
        o_ptr=ov,
        L=L,
        L_h=L_h,
        D=D,
        BD=BD,
        BL=BL,
        NBL=NBL,
    )
    return oq * ok * ov


#########################
# Reference implementation
#########################

def hyena_x_fwd_ref(
        qkv, h, 
): 
    """
    PyTorch reference implementation of Hyena-X forward
    """
    if len(h.shape) == 2: h = h.unsqueeze(1)
    B, D, L = qkv.shape 
    L_h = h.shape[2]
    z = F.conv1d(qkv, h, groups=D, padding= L_h - 1)[..., :L]
    q, k, v = z.split([D // 3, D // 3, D // 3], dim=1)
    return q * k * v

def hyena_x_fwd_causal_conv1d(
    qkv, h, 
):
    if len(h.shape) > 2: h = h.squeeze(1)
    B, D, L = qkv.shape 
    z = causal_conv1d_fn(qkv, h)
    q, k, v = z.split([D // 3, D // 3, D // 3], dim=1)
    return q * k * v

def hyena_x_fwd_im2col(
    qkv, h, 
):
    """
    Hyena-X forward with indirect im2col convolution
    """
    if len(h.shape) > 2: h = h.squeeze(1)
    B, D, L = qkv.shape 
    assert L % 2 == 0, "sequence length must be even"
    L_h = h.shape[-1]
    
    qkv_padded = F.pad(qkv, (L_h - 1, 0))    
    qkv_transformed = torch.empty((B, D, L, L_h), device=qkv.device, dtype=qkv.dtype)
    for i in range(L):
        qkv_transformed[..., i, :] = qkv_padded[..., i:i+L_h].flip(-1) 

    z = torch.einsum('bdlk,dk->bdl', qkv_transformed, h.flip(-1))
    q, k, v = z.split([D // 3, D // 3, D // 3], dim=1)
    return q * k * v

def full_hyena_x_fwd(
    u, h, lin_featurizer, lin_out_featurizer, impl='base_pytorch'
):
    """
    PyTorch reference implementation of Hyena-X forward
    """
    qkv = lin_featurizer(u)
    if impl == 'causal_conv1d':
        qkv = qkv.transpose(1, 2)
        z = hyena_x_fwd_causal_conv1d(qkv, h)
        z = z.transpose(1, 2)
    elif impl == "base_pytorch":
        qkv = qkv.transpose(1, 2)
        z = hyena_x_fwd_ref(qkv, h)
        z = z.transpose(1, 2)
    elif impl == 'im2col_pytorch':
        qkv = qkv.transpose(1, 2)
        z = hyena_x_fwd_im2col(qkv, h)
        z = z.transpose(1, 2)
    elif impl == 'bdl_indirect':
        qkv = qkv.transpose(1, 2)
        z = hyena_x_bdl_fwd(qkv, h)
        z = z.transpose(1, 2)
    elif impl == 'bdl_indirect_dchunk':
        qkv = qkv.transpose(1, 2)
        z = hyena_x_bdl_dchunk_fwd(qkv, h)
        z = z.transpose(1, 2)
    else:
        raise NotImplementedError("BDL custom implementation not implemented")
    return lin_out_featurizer(z)


#########################
# Func wrapper
#########################

class HyenaXInnerFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, h):
        L_h = h.shape[-1]
        if 2 <= L_h <= 4 and causal_conv1d_fn is not None:
            qkv = torch.cat([q, k, v], dim=1)
            y = hyena_x_fwd_causal_conv1d(qkv, h)
        elif 4 <= L_h <= 32:
            y = hyena_x_bdl_fwd(q, k, v, h)
        else:
            qkv = torch.cat([q, k, v], dim=1)
            y = hyena_x_fwd_ref(qkv, h)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented")


