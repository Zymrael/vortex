import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


import logging
from dataclasses import dataclass
from einops import rearrange
from .bwd import two_pass_bwd_grouped
from .fwd import two_pass_fwd_grouped, two_pass_fwd_grouped_refactor

logger = logging.getLogger(__name__)
from hyena_ops.kernel_utils import (
    BwdKernelConfig,
    FwdKernelConfig,
)

try:
    from local_causal_conv1d import old_causal_conv1d_fn
except ImportError:
    old_causal_conv1d_fn = None

from hyena_ops.utils import toeplitz


@dataclass(eq=False)
class DefaultTwoPassChunkedGateConvGateFwdConfig(FwdKernelConfig):
    schedule: str = "default"
    autotune: bool = True
    CHUNK_SIZE: int = 128
    BLOCK_D: int = 32
    THREADBLOCK_SWIZZLE: str = "row"


@dataclass(eq=False)
class DefaultTwoPassChunkedGateConvGateBwdConfig(BwdKernelConfig):
    schedule: str = "default"
    autotune: bool = True
    CHUNK_SIZE: int = 128
    BLOCK_D: int = 32
    THREADBLOCK_SWIZZLE: str = "row"
    LOAD_TOEPLITZ: bool = False
    LOAD_BX_LAG: bool = False


class TwoPassChunkedGateConvGate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        h: torch.Tensor,
        schedule: str = "default",
        use_refactor_path: bool = True,
        autotune: bool = False,
        fwd_kernel_cfg: FwdKernelConfig = None,
        bwd_kernel_cfg: BwdKernelConfig = None,
        # Always return y2 in fwd for backwards pass where y2 = T_local @ B*x + T_hat @ B_lag*x_lag
        # return_y2: bool = True,
    ):
        if fwd_kernel_cfg is None:
            fwd_kernel_cfg = DefaultTwoPassChunkedGateConvGateFwdConfig()
        if bwd_kernel_cfg is None:
            bwd_kernel_cfg = DefaultTwoPassChunkedGateConvGateBwdConfig()

        return_toeplitz = bwd_kernel_cfg.LOAD_TOEPLITZ
        return_bx_lag = bwd_kernel_cfg.LOAD_BX_LAG
        schedule = schedule if autotune else fwd_kernel_cfg.schedule

        # Chunk size for backwards must be the same as the forward chunk size if re-using toeplitz
        if return_toeplitz:
            bwd_kernel_cfg.CHUNK_SIZE = fwd_kernel_cfg.CHUNK_SIZE
        # TMA
        if not autotune and fwd_kernel_cfg.USE_TMA:
            raise NotImplementedError("Skip TMA for now")
            # kernel = two_pass_fwd_grouped_tma
        else:
            kernel = two_pass_fwd_grouped_refactor if use_refactor_path else two_pass_fwd_grouped

        x = x if x.is_contiguous() else x.contiguous()
        B = B if B.is_contiguous() else B.contiguous()
        C = C if C.is_contiguous() else C.contiguous()
        h = h if h.is_contiguous() else h.contiguous()

        out = kernel(
            x,
            B,
            C,
            h,
            autotune=autotune,
            schedule=schedule,
            return_toeplitz=return_toeplitz,
            # Always return y2 in fwd for backwards
            return_y2=True,
            return_bx_lag=return_bx_lag,
            CHUNK_SIZE=fwd_kernel_cfg.CHUNK_SIZE,
            BLOCK_D=fwd_kernel_cfg.BLOCK_D,
            NUM_PIPELINE_STAGES=fwd_kernel_cfg.NUM_PIPELINE_STAGES,
            THREADBLOCK_SWIZZLE=fwd_kernel_cfg.THREADBLOCK_SWIZZLE,
            num_warps=fwd_kernel_cfg.num_warps,
            num_stages=fwd_kernel_cfg.num_stages,
            num_ctas=fwd_kernel_cfg.num_ctas,
            return_autotune_result=True if autotune else False,
        )

        if use_refactor_path:
            bx_lag, y2, y = out
            T, T_hat = None, None
        else:
            y, T, T_hat, y2, bx_lag = out

        ctx.save_for_backward(x, B, C, h, T, T_hat, y2, bx_lag)
        ctx.fwd_kernel_cfg = fwd_kernel_cfg
        ctx.bwd_kernel_cfg = bwd_kernel_cfg

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, B, C, h, T, T_hat, y2, bx_lag = ctx.saved_tensors
        dy = dy if dy.is_contiguous() else dy.contiguous()
        x = x if x.is_contiguous() else x.contiguous()
        B = B if B.is_contiguous() else B.contiguous()
        C = C if C.is_contiguous() else C.contiguous()
        h = h if h.is_contiguous() else h.contiguous()

        kernel_config: BwdKernelConfig = ctx.bwd_kernel_cfg

        if kernel_config.USE_TMA:
            raise NotImplementedError("Skip TMA for now")
            # kernel = two_pass_bwd_grouped_tma
        else:
            kernel = two_pass_bwd_grouped

        dx, dB, dC, dh = kernel(
            dy,
            x,
            B,
            C,
            h,
            y2=y2,
            T=T,
            T_hat=T_hat,
            bx_lag=bx_lag,
            autotune=False,
            version=kernel_config.version,
            CHUNK_SIZE=kernel_config.CHUNK_SIZE,
            BLOCK_D=kernel_config.BLOCK_D,
            NUM_PIPELINE_STAGES=kernel_config.NUM_PIPELINE_STAGES,
            THREADBLOCK_SWIZZLE=kernel_config.THREADBLOCK_SWIZZLE,
            num_warps=kernel_config.num_warps,
            num_stages=kernel_config.num_stages,
            num_ctas=kernel_config.num_ctas,
        )

        return (
            dx,
            dB,
            dC,
            dh,
            None,  # autotune
            None,  # schedule
            None,  # FwdKernelConfig
            None,  # BwdKernelConfig
        )


def two_pass_chunked_gate_conv_gate(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    h: torch.Tensor,
    return_toeplitz: bool = True,
    # Only used if autotune
    schedule: str = "default",
    # autotune only applies to fwd pass
    autotune: bool = False,
    # manual kernel configs
    fwd_kernel_cfg: FwdKernelConfig = None,
    bwd_kernel_cfg: BwdKernelConfig = None,
):
    assert autotune or (fwd_kernel_cfg is not None), "Must specify fwd_kernel_cfg if not autotuning"
    if autotune and (fwd_kernel_cfg is not None):
        warnings.warn("WARNING: Both autotune and fwd_kernel_cfg specified, using fwd_kernel_cfg")
        autotune = False
    assert bwd_kernel_cfg is not None, "Must specify bwd_kernel_cfg, autotune not supported for bwd pass"
    if return_toeplitz:
        # Check that fwd returns toeplitz and that bwd uses load_toeplitz
        if fwd_kernel_cfg is not None:
            fwd_kernel_cfg.RETURN_TOEPLITZ = True
        bwd_kernel_cfg.LOAD_TOEPLITZ = True

    return TwoPassChunkedGateConvGate.apply(
        x,
        B,
        C,
        h,
        schedule,
        autotune,
        fwd_kernel_cfg,
        bwd_kernel_cfg,
    )


def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    apply_gated_bias=False,
    return_final_states=False,
    final_states_out=None,
    activation=None,
    *args,
    **kwargs,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)

    seqlen = x.shape[-1]
    dim, width = weight.shape
    if apply_gated_bias:
        gated_bias = bias
        bias = None
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    out = out[..., :seqlen]

    if apply_gated_bias:
        out = out + x * gated_bias

    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not return_final_states else (out, final_states_out)


def hyena_se(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    h: torch.Tensor,
    bias: torch.Tensor = None,
    apply_gated_bias: bool = False,
    schedule: str = "default",
    autotune: bool = False,
    fwd_kernel_cfg: FwdKernelConfig = None,
    bwd_kernel_cfg: BwdKernelConfig = None,
):
    return TwoPassChunkedGateConvGate.apply(
        x,
        h,
        C,
        bias,
        apply_gated_bias,
        schedule,
        autotune,
        fwd_kernel_cfg,
        bwd_kernel_cfg,
    )


def reshape_inputs(x, k, q, num_groups, input_layout, inner_layout):
    """
    Reshapes feature groups and filters to match a target grouped layout.
    `num_groups` is inferred from the filter `h`.
    """
    if input_layout == "bdl":
        B, D, L = x.shape
        if inner_layout == "bdl":
            return x, k, q
        elif inner_layout == "blgdg":
            x = x.transpose(1, 2).reshape(B, L, num_groups, D // num_groups)
            k = k.transpose(1, 2).reshape(B, L, num_groups, D // num_groups)
            q = q.transpose(1, 2).reshape(B, L, num_groups, D // num_groups)
            return x, k, q
            # rearrange(x, "b (g dg) l -> b l g dg", g=num_groups)
            # k = rearrange(k, "b (g dg) l -> b l g dg", g=num_groups)
            # q = rearrange(q, "b (g dg) l -> b l g dg", g=num_groups)
            # return x, k, q
        else:
            raise NotImplementedError("inner_layout must be bdl or blgdg")

    elif input_layout == "blgdg":
        B, L, G, DG = x.shape
        assert G == num_groups, "number of groups in feature groups and filter must match"
        if inner_layout == "bdl":
            x = rearrange(x, "b l g dg -> b (g dg) l", g=num_groups)
            k = rearrange(k, "b l g dg -> b (g dg) l", g=num_groups)
            q = rearrange(q, "b l g dg -> b (g dg) l", g=num_groups)
            return x, k, q
        elif inner_layout == "bgdgl":
            x = rearrange(x, "b l g dg -> b g dg l", g=num_groups)
            k = rearrange(k, "b l g dg -> b g dg l", g=num_groups)
            q = rearrange(q, "b l g dg -> b g dg l", g=num_groups)
            return x, k, q
        else:
            raise NotImplementedError("inner_layout must be bdl or bgdgl")
    elif input_layout == "bgdgl":
        B, G, D, L = x.shape
        assert G == num_groups, "number of groups in feature groups and filter must match"
        if inner_layout == "bdl":
            x = rearrange(x, "b g d l -> b (g d) l", g=num_groups)
            B = rearrange(B, "b g d l -> b (g d) l", g=num_groups)
            C = rearrange(C, "b g d l -> b (g d) l", g=num_groups)
            return x, B, C
        elif inner_layout == "bgdgl":
            return x, k, q
        else:
            raise NotImplementedError("inner_layout must be bdl or bgdgl")


def reshape_outputs(y, num_groups, output_layout, inner_layout):
    if inner_layout == "bdl":
        if output_layout == "bld":
            return rearrange(y, "b d l -> b l d")
        elif output_layout == "blgdg":
            return rearrange(y, "b (g dg) l -> b l g dg", g=num_groups)
        elif output_layout == "bdl":
            return y
        else:
            raise NotImplementedError("output_layout must be bld, bdl or blgdg")
    elif inner_layout == "blgdg":
        B, L, G, DG = y.shape
        if output_layout == "bld":
            return y.reshape(B, L, G * DG)
        elif output_layout == "blgdg":
            return y
        else:
            raise NotImplementedError("output_layout must be bld or blgdg")


def inner_hyena_se_ref(
    x: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    h: torch.Tensor,
    bias: torch.Tensor = None,
    input_layout: str = "bdl",
    output_layout: str = "bdl",
    inner_layout: str = "bdl",
    apply_gated_bias: bool = False,
    schedule: str = "default",
    autotune: bool = False,
    fwd_kernel_cfg: FwdKernelConfig = None,
    bwd_kernel_cfg: BwdKernelConfig = None,
    use_fast_causal_conv1d: bool = False,
    kernel_path: str = "unfused_gcg_fast",
):
    """
    Args:
        input_layout: "bdl" or "blgdg" or "bgdgl"
        output_layout: "bdl" or "blgdg" or "bgdgl"
    Note: `h` has shape (g, 1, hl), where
    `g` = number of feature channel groups (num_groups) and `dg` = number of channels per group
    `g` is thus the number of distinct filters
    """
    num_groups = h.shape[0]
    x, k, q = reshape_inputs(x, k, q, num_groups, input_layout, inner_layout)
    assert k.shape == q.shape == x.shape, "feature groups must have the same shape"
    if apply_gated_bias and kernel_path == "unfused_gcg_fast":
        warnings.warn("Gated bias not supported for unfused_gcg_fast kernel")

    if kernel_path in ["unfused_gcg", "unfused_gcg_fast"]:
        assert inner_layout == "bdl", "unfused_gcg only supports bdl inner layout"
        b, d, l = x.shape

        if num_groups != d:
            h = h.squeeze(1).repeat(d // num_groups, 1)
        kx = k * x
        if kernel_path == "unfused_gcg_fast" and h.shape[-1] <= 4:
            y = old_causal_conv1d_fn(kx, h, bias=bias)
        else:
            y = causal_conv1d_ref(
                kx,
                h,
                bias=bias,
                apply_gated_bias=apply_gated_bias,
                initial_states=None,
                return_final_states=False,
                final_states_out=None,
                activation=None,
            )
        y = q * y
        y = reshape_outputs(y, num_groups, output_layout, inner_layout)
        return y
    else:
        raise NotImplementedError("Reference cgcg kernel path not integrated in hyena_se_ref")


def inner_hyena_se(
    x: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    h: torch.Tensor,
    bias: torch.Tensor = None,
    input_layout: str = "bdl",
    output_layout: str = "bdl",
    inner_layout: str = "bdl",
    apply_gated_bias: bool = False,
    schedule: str = "default",
    autotune: bool = True,
    fwd_kernel_cfg: FwdKernelConfig = None,
    bwd_kernel_cfg: BwdKernelConfig = None,
    use_fast_causal_conv1d: bool = False,
    kernel_path: str = "cgcg_v1",
):
    """
    Args:
        input_layout: "bdl" or "blgdg" or "bgdgl"
        output_layout: "bdl" or "blgdg" or "bgdgl"
    Note: `h` has shape (g, 1, hl), where
    `g` = number of feature channel groups (num_groups) and `dg` = number of channels per group
    `g` is thus the number of distinct filters
    """
    num_groups = h.shape[0]
    assert inner_layout == "blgdg", "inner_layout must be blgdg"
    x, k, q = reshape_inputs(x, k, q, num_groups, input_layout, inner_layout)
    assert k.shape == q.shape == x.shape, "feature groups must have the same shape"
    if len(h.shape) == 2:
        h = h.unsqueeze(1)

    y = TwoPassChunkedGateConvGate.apply(
        x,
        k,
        q,
        h,
        schedule,
        autotune,
        fwd_kernel_cfg,
        bwd_kernel_cfg,
    )
    y = reshape_outputs(x, num_groups, output_layout, inner_layout)
    return y


class HyenaSE(nn.Module):
    def __init__(
        self,
        num_groups: int,
        input_layout: str = "bdl",
        output_layout: str = "bdl",
        inner_layout: str = "bdl",
        apply_gated_bias: bool = False,
        schedule: str = "default",
        autotune: bool = False,
        fwd_kernel_cfg: FwdKernelConfig = None,
        bwd_kernel_cfg: BwdKernelConfig = None,
        use_fast_causal_conv1d: bool = False,
        kernel_path: str = "unfused_gcg_fast",
    ):
        super().__init__()
        self.num_groups = num_groups
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.inner_layout = inner_layout
        self.apply_gated_bias = apply_gated_bias
        self.schedule = schedule
        self.autotune = autotune
        self.fwd_kernel_cfg = fwd_kernel_cfg
        self.bwd_kernel_cfg = bwd_kernel_cfg
        self.use_fast_causal_conv1d = use_fast_causal_conv1d
        self.kernel_path = kernel_path


# this oen calls our kernels
class HyenaSEFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        x1,
        x2,
        h,
        bias,
        apply_gated_bias,
        input_layout,
        schedule,
        autotune,
        fwd_kernel_cfg,
        bwd_kernel_cfg,
        use_fast_causal_conv1d,
    ):
        x, B, C, h = prepare_inputs(x, B, C, h, input_layout)

        Bx = B * x
        if use_fast_causal_conv1d:
            y = old_causal_conv1d_fn(Bx, h, bias=bias, apply_gated_bias=apply_gated_bias)
        else:
            y = causal_conv1d_ref(
                Bx,
                h,
                bias=bias,
                apply_gated_bias=apply_gated_bias,
                initial_states=None,
                return_final_states=False,
                final_states_out=None,
                activation=None,
            )
        return C * y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("Backward pass not implemented")


class CausalConv1D(nn.Conv1d):
    def __init__(self, gemm_conv1d=False, *args, **kwargs):
        super(CausalConv1D, self).__init__(*args, **kwargs)
        self.gemm_conv1d = gemm_conv1d
        self.filter_len = self.weight.shape[-1]
        if gemm_conv1d:
            # TODO: generalize to different sizes
            toeplitz_matrix = toeplitz(self.weight.squeeze(1), size=256).to(self.weight.device).to(self.weight.dtype)
            self.register_buffer("toeplitz_matrix", toeplitz_matrix)

    def forward(self, input):
        if self.gemm_conv1d:
            # matmul with toeplitz
            # only works with B=1
            # self.T is a (d, l, l) tensor
            B, D, L = input.shape
            custom_output = torch.bmm(self.toeplitz_matrix, input.reshape(-1, L, 1)).reshape(B, D, L)
        else:
            w = rearrange(self.weight, "d 1 w -> d w")
            if self.filter_len <= 4:
                custom_output = old_causal_conv1d_fn(input, w, bias=self.bias, activation=None, seq_idx=None)
            else:
                custom_output = causal_conv1d_ref(input, w, bias=self.bias, activation=None, seq_idx=None)
        return custom_output


class HyenaFeaturizer(nn.Module):
    def __init__(
        self,
        d_model: int,
        feat_filter_len: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.feat_dense = nn.Linear(d_model, 3 * d_model, bias=False)
        feat_conv = nn.Conv1d(
            3 * d_model,
            3 * d_model,
            groups=3 * d_model,
            kernel_size=feat_filter_len,
            bias=False,
        )
        feat_conv_filter = rearrange(feat_conv.weight, "d 1 lh -> d lh")
        self.register_parameter("feat_conv_filter", nn.Parameter(feat_conv_filter))

    def forward(self, x: torch.Tensor):
        x = self.feat_dense(x)
        x = x.transpose(1, 2)
        if old_causal_conv1d_fn is not None:
            x = old_causal_conv1d_fn(x, self.feat_conv_filter, bias=None, activation=None, seq_idx=None)
        else:
            x = self.causal_conv1d_ref(x, self.feat_conv_filter, bias=None, activation=None, seq_idx=None)
        return x


class OldHyenaSE(nn.Module):
    def __init__(
        self,
        d_model,
        l_max,
        order=2,
        inner_filter_len=4,
        num_heads=1,
        inner_factor=1,
        num_blocks=1,
        fused_bias_fc=False,
        outer_mixing=False,
        dropout=0.0,
        filter_dropout=0.0,
        filter_cls="hyena-filter",
        post_order_ffn=False,
        jit_filter=False,
        short_filter_order=3,
        activation="id",
        return_state=False,
        use_inner_filter=False,
        **filter_args,
    ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            filter_order: (int): Width of the FFN parametrizing the implicit filter. Defaults to 64
            num_heads: (int): Number of heads. Defaults to 1
            inner_factor: (int): Width multiplier. Defaults to 1
            num_blocks: (int): Number of blocks in sequence length. Defaults to 1
            fused_bias_fc: (bool): Whether to use fused bias FC. Defaults to False
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
            post_order_ffn: (bool): Apply a dense layer between steps of the recurrence. Defaults to False
            jit_filter: (bool): Whether JIT the implicit filter function. Defaults to False
            short_filter_order: (int): Length of the explicit input convolutional filter. Defaults to 3
            activation: (str): type of act between kernel output and FF (default identity)
            return_state: (bool): whether to return a state
        """
        super().__init__()
        assert d_model % num_heads == 0, f"Model dimension {d_model} must be divisible by num heads {num_heads}"
        assert (
            l_max % num_blocks == 0
        ), f"Maximum signal length {l_max} must be divisible by block dimension {num_blocks}"
        block_dim = l_max // num_blocks
        head_dim = d_model // num_heads

        self.d_model = d_model
        self.order = order
        self.l_max = l_max
        self.num_heads = num_heads
        self.inner_factor = inner_factor
        self.block_dim = block_dim
        self.head_dim = head_dim
        self.inner_filter_len = inner_filter_len
        self.post_order_ffn = post_order_ffn
        self.short_filter_order = short_filter_order
        self.num_blocks = num_blocks
        self.filter_dropout = filter_dropout
        self.jit_filter = jit_filter
        self.outer_mixing = outer_mixing
        self.activation = activation
        self.return_state = return_state
        self.dropout = nn.Dropout(dropout)
        self.use_inner_filter = use_inner_filter
        self.setup_projections(fused_bias_fc, inner_factor)
        self.setup_filters(filter_cls, filter_args)

    def setup_projections(self, fused_bias_fc, inner_factor):
        "Initializes input and output projections (over the width dimension)"
        linear_cls = nn.Linear
        self.out_proj = linear_cls(self.d_model * inner_factor, self.d_model, bias=True)
        self.in_proj = linear_cls(self.d_model, (self.order + 1) * self.d_model, bias=True)
        if self.post_order_ffn:
            self.ord_proj_w = nn.Parameter(
                torch.randn(self.order, self.num_heads, self.num_heads) / math.sqrt(self.head_dim)
            )

    def setup_filters(self, filter_cls, filter_args):
        "Initializes the explicit and implicit filters"
        assert self.order >= 2, f"Order must be at least 2, (got {self.order})"
        total_width = self.d_model * self.inner_factor * (self.order + 1)

        self.short_filter = CausalConv1D(
            in_channels=total_width,
            out_channels=total_width,
            kernel_size=self.short_filter_order,
            groups=total_width,
            padding=self.short_filter_order - 1,
        )
        self.inner_filter = CausalConv1D(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=self.inner_filter_len,
            groups=self.d_model,
            padding=self.inner_filter_len - 1,
        )

    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = u.transpose(1, 2)
        uc = self.short_filter(u)[..., :l_filter]
        q, k, v = uc.split(self.d_model, dim=1)

        # y = gate_gate(x=v, k=k, q=q)
        v = k * v
        if self.use_inner_filter:
            v = self.inner_filter(v)
        y = q * v
        y = y.transpose(1, 2)
        y = self.out_proj(y)

        if self.return_state:
            return y, None
        return y

    @property
    def d_output(self):
        return self.d_model


class HyenaSE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = None,
        num_groups: int = None,
        feat_filter_len: int = 3,
        inner_filter_len: int = 4,
        output_layout: str = "bld",
        inner_layout: str = "bdl",
        apply_gated_bias: bool = False,
        schedule: str = "default",
        autotune: bool = False,
        fwd_kernel_cfg: FwdKernelConfig = None,
        bwd_kernel_cfg: BwdKernelConfig = None,
        use_fast_causal_conv1d: bool = False,
        kernel_path: str = "unfused_gcg_fast",
    ):
        super().__init__()
        self.d_model = d_model
        self.input_layout = "bdl"
        self.output_layout = output_layout
        self.inner_layout = inner_layout
        num_groups = self.d_model if num_groups is None else num_groups
        self.num_groups = num_groups
        self.featurizer = HyenaFeaturizer(d_model, feat_filter_len)
        self.output_featurizer = nn.Linear(d_model, d_model)
        self.kernel_path = kernel_path
        inner_conv = nn.Conv1d(
            num_groups,
            num_groups,
            groups=num_groups,
            kernel_size=inner_filter_len,
            bias=False,
        )
        inner_conv_filter = rearrange(inner_conv.weight, "g 1 lh -> g lh")
        self.register_parameter("inner_conv_filter", nn.Parameter(inner_conv_filter))
        if kernel_path in ["unfused_gcg_fast", "unfused_gcg"]:
            self.inner_fn = inner_hyena_se_ref
            self.inner_layout = "bdl"
        else:
            self.inner_fn = inner_hyena_se
            self.inner_layout = "blgdg"
        # handle bias

    def forward(self, x: torch.Tensor):
        z = self.featurizer(x)
        q, k, x = torch.split(z, self.d_model, dim=1)
        y = self.inner_fn(
            q,
            k,
            x,
            self.inner_conv_filter,
            bias=None,
            input_layout=self.input_layout,
            output_layout=self.output_layout,
            inner_layout=self.inner_layout,
            apply_gated_bias=False,
            kernel_path=self.kernel_path,
        )
        y = self.output_featurizer(y)
        return x


class HyenaMR(nn.Module):
    pass


class HyenaLI(nn.Module):
    pass


@torch.jit.script
def gate_gate(x, k, q):
    return x * k * q
