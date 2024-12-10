# Copyright (c) 2024, Michael Poli.

# Copyright (c) Together
# This software is distributed under the terms of the Apache License, Version 2.0
# Author: Michael Poli
# Note: MP and PP utilities are removed for ease of use and editing.

import torch
import torch.nn as nn
import torch.nn.functional as F

from vortex.model.cache import InferenceParams, HyenaCascadeFIRInferenceParams, HyenaCascadeIIRInferenceParams
from vortex.model.engine import HyenaInferenceEngine
from vortex.model.layers import ParallelGatedMLP, RMSNorm, VocabParallelEmbedding, TELinear
from vortex.model.utils import column_split, interleave, print_rank_0
from vortex.logging import initialize_vortex_logger, activations_logger

import logging 
from tqdm import tqdm

from vortex.model.attention import MHA

try:
    from vortex.positional_embeddings import swap_mha_rope
except ImportError:
    "could not import swap_mha_rope from src.positional_embeddings"


class AttentionBlock(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.config = config
        self.pre_norm, self.post_norm = RMSNorm(config), RMSNorm(config)
        self.layer_idx = layer_idx
        self.print_activations = config.get("print_activations", False)
        self.proj_groups = config.get("proj_groups", 1)
        dtype = config.get("attn_block_dtype", torch.bfloat16)
        mlp_dtype = config.get("mlp_dtype", torch.bfloat16)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads

        self.counter = 0
        self.inner_mha_cls = MHA(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_heads_kv=config.num_attention_heads // self.proj_groups,
            rotary_emb_dim=config.hidden_size // config.num_attention_heads,
            qkv_proj_bias=config.get("qkv_proj_bias", True),
            rotary_emb_base=config.get("rotary_emb_base", 1000000),
            causal=True,
            layer_idx=layer_idx,
            out_proj_bias=config.get("mha_out_proj_bias", True),
            use_flash_attn=self.config.use_flash_attn,
        ).to(dtype=dtype)

        # check if using interpolated rotary pos emb from config, and swap the rope emb
        if config.get("use_interpolated_rotary_pos_emb", False):
            swap_mha_rope(
                mha=self.inner_mha_cls,
                kwargs_new_rope={"scaling_factor": config.get("rotary_emb_scaling_factor", 1.0)},
            )

        if self.config.get("smeared_gqa", False):
            self.inner_mha_cls.num_heads_kv = self.inner_mha_cls.num_heads
        self.inner_mha_cls.rotary_emb.register_buffer("inv_freq", self.inner_mha_cls.rotary_emb.inv_freq)

        self.mlp = ParallelGatedMLP(config, layer_idx).to(dtype=mlp_dtype)

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        if (
            type(padding_mask) == torch.Tensor
        ):  # workaround for masking bug in FA. This works because Wqkv does not have bias
            # and attention scores will be also automatically zeroed.
            u = u * padding_mask[..., None]

        if self.print_activations:
            activations_logger.info(f"pre mha: {u}")

        u = (
            self.inner_mha_cls(
                self.pre_norm(u),
                inference_params=inference_params,
            )
            + u
        )
        if self.print_activations:
            activations_logger.info(f"post mha: {u}")

        if type(padding_mask) == torch.Tensor:  # guard against bias
            u = u * padding_mask[..., None]
        
        if self.print_activations:
            activations_logger.info(f"pre mlp: {u} {u.min()} {u.max()} {self.mlp.__class__}")
            activations_logger.info(f"post mlp norm: {self.post_norm(u)} {self.post_norm(u).min()} {self.post_norm(u).max()}")
            activations_logger.info(f"post mlp: {self.mlp(self.post_norm(u))} {self.mlp(self.post_norm(u)).min()} {self.mlp(self.post_norm(u)).max()}")
        
        u = self.mlp(self.post_norm(u)) + u
        return u, None


class HyenaCascade(nn.Module):
    def __init__(self, 
            config, 
            layer_idx,
            hyena_filter_groups=None,
            fir_inner_filter_length=None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hyena_filter_groups = hyena_filter_groups
        self.print_activations = config.get("print_activations", False)
        self.ground_truth_activations_path = config.get("ground_truth_activations_path", None)

        self.use_flashfft = config.get("use_flashfft", False)
        self.state_size = config.state_size
        self.hidden_size = config.hidden_size
        self.num_filters = config.num_filters
        self.inference_mode = config.get("inference_mode", True)
        self.counter = 0
        self.column_split_hyena = config.get("column_split_hyena", True)
        self.hyena_flip_x1x2 = config.get("hyena_flip_x1x2", False)

        assert self.hidden_size % self.num_filters == 0 and self.num_filters <= self.hidden_size

        # attention heads are not used except to split post short_filter
        # projections in the same way as the checkpoint
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads

        self.fir_inner_filter_length = fir_inner_filter_length
        self.short_filter_length = config.short_filter_length
        self.short_filter_weight = nn.Parameter(torch.randn(3 * config.hidden_size, 1, config.short_filter_length))
        self.short_filter_bias = (
            nn.Parameter(torch.randn(3 * config.hidden_size)) if config.short_filter_bias else None
        )

        self.engine = HyenaInferenceEngine(
            layer_idx=layer_idx, 
            ground_truth_activations_path=self.ground_truth_activations_path,
            print_activations=self.print_activations,
            hyena_flip_x1x2=config.get("hyena_flip_x1x2", False)
        )
        self.use_flash_depthwise = config.get("use_flash_depthwise", False)
        self.data_dtype = None

        if self.use_flash_depthwise:
            try:
                from flashfftconv import FlashDepthwiseConv1d

                self.fir_fn = FlashDepthwiseConv1d(
                    channels=3 * self.hidden_size,
                    kernel_size=self.short_filter_length,
                    padding=self.short_filter_length - 1,
                    weights=self.short_filter_weight,
                    bias=self.short_filter_bias,
                    device=None,
                    dtype=self.config.get("depthwise_dtype", torch.bfloat16),
                )
            except ImportError:
                "flashfftconv not installed"
        else:
            self.fir_fn = F.conv1d

            self.fir_inner_fn = F.conv1d

        self.fftconv_fn = None
        self.long_fir_threshold = config.get("long_fir_threshold", None)
        if self.long_fir_threshold is not None:
            assert self.use_flashfft is False, "long_fir_threshold not compatible with fused flashfft"

        self.num_systems = self.hyena_filter_groups
        self.channels_per_group = self.hidden_size // self.hyena_filter_groups

        if self.fir_inner_filter_length:

            self.h = nn.Parameter(torch.randn(self.hyena_filter_groups, 1, fir_inner_filter_length))

            if fir_inner_filter_length >= 128:
                self.D = nn.Parameter(torch.zeros(self.hidden_size))

            if fir_inner_filter_length < 128:
                self.D = None

        else:
            log_poles = torch.randn(self.num_systems, self.state_size, 1, dtype=torch.float32)

            # TODO: bring over init from internals
            # poles[..., 0] = 1e-2 * torch.randn(self.num_systems, self.state_size, 1)
            #poles[..., 1] = 1e-3 * torch.randn(self.num_systems, self.state_size, 1)

            self.log_poles = nn.Parameter(log_poles)
            self.residues = nn.Parameter(torch.randn(self.num_systems, self.state_size, dtype=torch.float32))
            self.D = nn.Parameter(torch.zeros(self.hidden_size))
            self.h = None
        self.t = None

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        if inference_params is not None and self.layer_idx in inference_params.fir_state_dict.keys():
            return self.sequential_forward(u, inference_params)

        else:
            return self.parallel_forward(u, inference_params, padding_mask)

    def parallel_forward(self, u, inference_params=None, padding_mask=None):
        L = u.shape[1]
        dims = (
            self.hidden_size,
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
            self.state_size,
            self.hyena_filter_groups,
        )
        if self.print_activations:
            activations_logger.info(f"pre 1 parallel fir: {u}, {u.min()}, {u.max()}")
        z_pre, fir_state = self.engine.parallel_fir(
            self.fir_fn,
            u,
            self.short_filter_weight,
            self.short_filter_bias,
            L,
            dims=dims,
            gate=False,
            column_split_hyena=self.column_split_hyena,
            fir_length=self.short_filter_length,
            inference_params=inference_params,
            padding_mask=padding_mask,
            dim_last=True,
        )
        if inference_params:
            inference_params.fir_state_dict[self.layer_idx] = fir_state
        
        if self.config.interleave:
            z_pre = interleave(z_pre)

        if self.h is None:
            h, _, _, _ = self.compute_filter(L, u.device)
        else:
            h = self.h
        
        D = self.D
        if self.hyena_filter_groups > 1:
            h = h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)
        
        # if inference_params is not None, we plan to perform generation:
        # prefilling is handled by the engine.
        if self.fir_inner_filter_length is not None:
            if self.print_activations:
                activations_logger.info(f"pre 2 parallel fir: {z_pre}, {z_pre.min()}, {z_pre.max()}, {self.fir_inner_filter_length}")
            y, fir_inner_state = self.engine.parallel_fir(
                self.fir_inner_fn,
                z_pre,
                h,
                D,
                L,
                dims=dims,
                gate=True,
                gated_bias=self.fir_inner_filter_length >= 128,
                dim_last=False,
                column_split_hyena=self.column_split_hyena,
                fir_length=self.fir_inner_filter_length,
                inference_params=inference_params,
                padding_mask=padding_mask,
                groups=self.hyena_filter_groups,
            )
            if self.print_activations:
                activations_logger.info(f"post 2 parallel fir: {y}, {y.min()}, {y.max()}")
            y = y.permute(0, 2, 1)
            if inference_params:
                inference_params.fir_inner_state_dict[self.layer_idx] = fir_inner_state
        else:
            if self.print_activations:
                activations_logger.info(f"pre 2 parallel iir: {z_pre}, {z_pre.min()}, {z_pre.max()}")
            y = self.engine.parallel_iir(
                z_pre,
                h,
                D,
                L,
                t=self.t,
                poles=self.log_poles,
                residues=self.residues,
                dims=dims,
                inference_params=inference_params,
                layer_idx=self.layer_idx,
                prefill_style=self.config.get("prefill_style", "fft"),
                use_flashfft=self.use_flashfft,
                fftconv_fn=self.fftconv_fn,
                column_split_hyena=self.column_split_hyena,
                long_fir_threshold=self.long_fir_threshold,
                padding_mask=padding_mask,
            )
            if self.print_activations:
                activations_logger.info(f"post 2 parallel iir: {y}, {y.min()}, {y.max()}")

        return y, inference_params

    def sequential_forward(self, u, inference_params):
        if self.data_dtype is None:
            self.data_dtype = u.dtype

        if len(u.shape) > 2:
            u = u[:, -1]

        z_pre, fir_state = self.engine.step_fir(
            u, inference_params.fir_state_dict[self.layer_idx], weight=self.short_filter_weight, bias=self.short_filter_bias
        )
        inference_params.fir_state_dict[self.layer_idx] = fir_state

        if self.config.interleave:
            z_pre = interleave(z_pre)
            
        x2, x1, v = (
            column_split(z_pre, self.num_attention_heads, self.hidden_size_per_attention_head)
            if self.column_split_hyena
            else z_pre.split([self.hidden_size, self.hidden_size, self.hidden_size], dim=1)
        )

        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        if self.fir_inner_filter_length is not None:
            if self.hyena_filter_groups > 1:
                h = self.h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)
            else:
                h = self.h

            
            y, fir_inner_state = self.engine.step_fir(
                x1 * v,
                inference_params.fir_inner_state_dict[self.layer_idx],
                weight=h,
                bias=self.D,
                flip_filter=self.fir_inner_filter_length >= 128,
                gated_bias=self.fir_inner_filter_length >= 128,
            )
            y = y * x2
            inference_params.fir_inner_state_dict[self.layer_idx] = fir_inner_state
        else:
            y, iir_state = self.engine.step_iir(
                x2,
                x1,
                v,
                self.D,
                self.residues,
                self.log_poles,
                inference_params.state_dict[self.layer_idx],
                iir_groups=1,
            )
            inference_params.state_dict[self.layer_idx] = iir_state

        y = y.to(dtype=self.data_dtype)
        return y[:, None], inference_params

    def update_time(self, L, device):
        """
        Set [0, 1, ..., L-1] where L is the length of the current batch of inputs.
        If L is greater than the length of the previous batch, then the time vector is
        reinitialized. Otherwise, the time vector is truncated from cache.
        """
        if self.t is None:
            self.t = torch.arange(L, device=device)[None, None]
        elif self.t.shape[-1] < L:
            self.t = torch.arange(L, device=device)[None, None]
        else:
            self.t = self.t[..., :L]

    def compute_filter(self, L, device):
        self.update_time(L, device)
        filter_dtype = torch.float32
        residues, log_poles = (
            self.residues.to(filter_dtype),
            self.log_poles.to(filter_dtype),
        )
        h = (residues[...,None] * (log_poles * self.t).exp()).sum(1)[None] # B, D, L
        return h, filter_dtype, log_poles, residues


class ParallelGatedConvBlock(nn.Module):
    def __init__(self, 
            config, 
            layer_idx,
            hyena_filter_groups=None,
            fir_inner_filter_length=None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.print_activations = config.get("print_activations", False)
        self.ground_truth_activations_path = config.get("ground_truth_activations_path", None)
        self.low_mem_mode = config.get("low_mem_mode", False)
        self.fir_inner_filter_length = fir_inner_filter_length
        self.hyena_filter_groups = hyena_filter_groups if hyena_filter_groups is not None else config.hidden_size
        dtype = config.get("hyena_block_dtype", torch.bfloat16)
        mlp_dtype = config.get("mlp_dtype", torch.bfloat16)
        self.pre_norm, self.post_norm = RMSNorm(config).to(dtype=dtype), RMSNorm(config).to(dtype=dtype)
        self.filter = HyenaCascade(config, layer_idx, hyena_filter_groups=self.hyena_filter_groups, fir_inner_filter_length=fir_inner_filter_length).to(dtype=dtype)
        self.projections = TELinear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=config.qkv_proj_bias,
            init_method=torch.nn.init.xavier_uniform_,
            use_fp8=config.get("use_fp8_input_projections", False),
        )
        
        nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.qkv_proj_bias)
        self.out_filter_dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.hyena_out_proj_bias).to(dtype)
        self.mlp = ParallelGatedMLP(config, layer_idx).to(dtype=mlp_dtype)

        self.proj_norm_fn = self.proj_norm
        self.res_mlp_norm_fn = self.res_mlp_norm

        if self.config.get("compile", False):
            self.proj_norm_fn = torch.compile(self.proj_norm, fullgraph=True, dynamic=False, mode="reduce-overhead")
            self.res_mlp_norm_fn = torch.compile(
                self.res_mlp_norm, fullgraph=True, dynamic=False, mode="reduce-overhead"
            )

    def pad_to_multiple(self, x, multiple=16):
        """Pad input tensor to multiple of 16 only when FP8 is enabled"""
        if not self.config.get("use_fp8_input_projections", False):
            return x
        
        batch_size, seq_len, hidden_dim = x.size()
        pad_len = (multiple - (seq_len % multiple)) % multiple
        if pad_len == 0:
            return x
        return F.pad(x, (0, 0, 0, pad_len))
        
    def proj_norm(self, x):
        if self.print_activations:
            activations_logger.info(f"pre mixer norm: {x} {x.min()} {x.max()} {self.projections.__class__}")
            activations_logger.info(f"post mixer norm: {self.pre_norm(x)} {self.pre_norm(x).min()} {self.pre_norm(x).max()}")

            if self.ground_truth_activations_path:
                pre_norm_savanna = torch.load(f"{self.ground_truth_activations_path}/pre_mixer_norm_{self.layer_idx}.pt")
                post_norm_savanna = torch.load(f"{self.ground_truth_activations_path}/post_mixer_norm_{self.layer_idx}.pt")
            
                activation_diff = (x.squeeze() - pre_norm_savanna.squeeze()).abs()
                activations_logger.info(f"pre mixer norm activation_diff: {activation_diff.max()}, {activation_diff.mean()}")
                activation_diff = (self.pre_norm(x).squeeze() - post_norm_savanna.squeeze()).abs()
                activations_logger.info(f"post mixer norm activation_diff: {activation_diff.max()}, {activation_diff.mean()}")
                activations_logger.info(f"pre norm scale: {self.pre_norm.scale}, {self.pre_norm.scale.min()}, {self.pre_norm.scale.max()}")

        original_seq_len = x.size(1)
        normalized = self.pre_norm(x)
        normalized = self.pad_to_multiple(normalized)
        projected = self.projections(normalized)

        original_seq_len = x.size(1)
        # Slice back to original sequence length if padding was added
        if isinstance(projected, tuple):
            projected = projected[0]
        if projected.size(1) > original_seq_len:
            projected = projected[:, :original_seq_len, :]
        
        return projected

    def res_mlp_norm(self, x):
        if self.print_activations:
            activations_logger.info(f"pre mlp: {x} {x.min()} {x.max()} {self.mlp.__class__}")
            activations_logger.info(f"post mlp norm: {self.post_norm(x)} {self.post_norm(x).min()} {self.post_norm(x).max()}")
            activations_logger.info(f"post mlp: {self.mlp(self.post_norm(x))} {self.mlp(self.post_norm(x)).min()} {self.mlp(self.post_norm(x)).max()}")
            if self.ground_truth_activations_path:
                pre_mlp_savanna = torch.load(f"{self.ground_truth_activations_path}/pre_mlp_{self.layer_idx}.pt")
                post_mlp_savanna = torch.load(f"{self.ground_truth_activations_path}/post_mlp_norm_{self.layer_idx}.pt")
            
                activation_diff = (x.squeeze() - pre_mlp_savanna.squeeze()).abs()
                activations_logger.info(f"pre mlp activation_diff: {activation_diff.max()}, {activation_diff.mean()}")
                activation_diff = (self.post_norm(x).squeeze() - post_mlp_savanna.squeeze()).abs()
                activations_logger.info(f"post mlp norm activation_diff: {activation_diff.max()}, {activation_diff.mean()}")
        return self.mlp(self.post_norm(x)) + x

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        z = self.proj_norm_fn(u)

        if type(padding_mask) == torch.Tensor:  # guard against bias
            z = z * padding_mask[..., None]

        if self.print_activations:
            activations_logger.info(f"pre filter: {z} {z.min()} {z.max()} {self.filter.__class__}")
            if self.ground_truth_activations_path:
                z_savanna = torch.load(f"{self.ground_truth_activations_path}/pre_filter_{self.layer_idx}.pt")
                activation_diff = (z - z_savanna.squeeze()).abs()
                activations_logger.info(f"pre filter activation_diff: {activation_diff.max()}, {activation_diff.mean()}")
        z, inference_params = self.filter(z, inference_params=inference_params, padding_mask=padding_mask)
        
        
        if self.print_activations:
            activations_logger.info(f"post postgate: {z} {z.min()} {z.max()} {self.filter.__class__}")
            activations_logger.info(f"post out proj: {self.out_filter_dense(z)} {self.out_filter_dense(z).min()} {self.out_filter_dense(z).max()} {self.out_filter_dense.__class__}")
            activations_logger.info(f"post mixer: {self.out_filter_dense(z) + u} {(self.out_filter_dense(z) + u).min()} {(self.out_filter_dense(z) + u).max()}")
            if self.ground_truth_activations_path:
                z_savanna = torch.load(f"{self.ground_truth_activations_path}/post_filter_{self.layer_idx}.pt")
                activation_diff = (z - z_savanna.squeeze()).abs()
                activations_logger.info(f"post filter activation_diff: {activation_diff.max()}, {activation_diff.mean()}")    

                z_savanna = torch.load(f"{self.ground_truth_activations_path}/post_out_proj_{self.layer_idx}.pt")
                z_ = F.linear(z, self.out_filter_dense.weight)
                activation_diff = (z_ - z_savanna.squeeze()).abs()
                activations_logger.info(f"post out proj activation_diff: {activation_diff.max()}, {activation_diff.mean()}")
        
        z_in = self.out_filter_dense(z) + u

        #if self.layer_idx == 0:
        #    z_in = z_savanna.squeeze() + u + self.out_filter_dense.bias

        if type(padding_mask) == torch.Tensor:  # guard against bias
            z_in = z_in * padding_mask[..., None]

        y = self.res_mlp_norm_fn(z_in)

        return y, inference_params


def get_block(config, layer_idx, flash_fft=None):
    if layer_idx in config.attn_layer_idxs:
        return AttentionBlock(config, layer_idx)
    elif layer_idx in config.hcl_layer_idxs:
        block = ParallelGatedConvBlock(config, layer_idx)
        if config.get("use_flashfft", "False"):
            block.filter.fftconv_fn = flash_fft
        return block
    elif layer_idx in config.hcm_layer_idxs:
        block = ParallelGatedConvBlock(
            config, layer_idx, hyena_filter_groups=config.hcm_filter_groups, fir_inner_filter_length=config.hcm_filter_length)
        return block
    elif layer_idx in config.hcs_layer_idxs:
        block = ParallelGatedConvBlock(config, layer_idx, hyena_filter_groups=config.hcs_filter_groups, fir_inner_filter_length=config.hcs_filter_length)
        return block
    else:
        raise NotImplementedError


class StripedHyena(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.print_activations = config.get("print_activations", False)

        if self.print_activations:
            self.logger = initialize_vortex_logger("basic_logger")
        else:
            self.logger = initialize_vortex_logger("basic_logger", level=100)

        self.ground_truth_activations_path = config.get("ground_truth_activations_path", None)
        self.logger.info(f"Initializing StripedHyena with config: {config}")
        self.embedding_layer = VocabParallelEmbedding(config)
        self.norm = RMSNorm(config) if config.get("final_norm", True) else None
        self.unembed = self.embedding_layer if config.tie_embeddings else VocabParallelEmbedding(config)

        if config.get("use_flashfft", "True"):
            try:
                from flashfftconv import FlashFFTConv

                self.flash_fft = FlashFFTConv(config.seqlen, dtype=torch.bfloat16)
            except ImportError:
                "flashfftconv not installed"
        else:
            self.flash_fft = None

        self.logger.info(f"Initializing {config.num_layers} blocks...")
        self.blocks = nn.ModuleList()
        for layer_idx in tqdm(range(config.num_layers)):
            self.blocks.append(get_block(config, layer_idx, flash_fft=self.flash_fft))
            self.logger.info(f"Parameter count for block {layer_idx}: {sum(p.numel() for p in self.blocks[-1].parameters())}")

        self.logger.info(f"Initialized model")

    def forward(self, x, inference_params_dict=None, padding_mask=None):
        L = x.shape[1]
        if self.print_activations:
            activations_logger.info(f"pre embedding: {x}, {x.min()}, {x.max()}")
        
        x = self.embedding_layer.embed(x)
        
        if self.print_activations:
            activations_logger.info(f"post embedding: {x}, {x.min()}, {x.max()}")
        
        if inference_params_dict is not None:
            x, inference_params_dict_out = self.stateful_forward(
                x,
                inference_params_dict=inference_params_dict,
            )
        else:
            x, inference_params_dict_out = self.stateless_forward(x, padding_mask=padding_mask)

        if self.print_activations:
            activations_logger.info(f"pre norm: {x}, {x.min()}, {x.max()}")

        x = self.norm(x)
        
        if self.print_activations:
            activations_logger.info(f"post norm: {x}, {x.min()}, {x.max(), {self.norm.scale}}")

        x = self.unembed.unembed(x)
        return x, inference_params_dict_out

    def block_idx_to_name(self, block_idx):
        if block_idx in self.config.attn_layer_idxs:
            return "mha"
        elif block_idx in self.config.hcl_layer_idxs:
            return "hcl"
        elif block_idx in self.config.hcm_layer_idxs:
            return "hcm"
        elif block_idx in self.config.hcs_layer_idxs:
            return "hcs"
        else:
            raise ValueError(f"Block index {block_idx} not found")

    def stateful_forward(self, x, inference_params_dict=None):
        for block_idx, block in enumerate(self.blocks):
            inference_params = inference_params_dict[self.block_idx_to_name(block_idx)]

            if self.print_activations:
                activations_logger.info(f"pre block {block_idx}: {x}, {x.min()}, {x.max()} {block.__class__}")
                if self.ground_truth_activations_path:
                    x_savanna = torch.load(f"{self.ground_truth_activations_path}/pre_block_{block_idx}.pt")
                    activation_diff = (x - x_savanna.squeeze()).abs()
                    activations_logger.info(f"pre block {block_idx} activation_diff: {activation_diff.max()}, {activation_diff.mean()}")

            x, _ = block(x, inference_params=inference_params)
            
            if self.print_activations:
                activations_logger.info(f"post block {block_idx}: {x}, {x.min()}, {x.max()}")
                if self.ground_truth_activations_path:
                    x_savanna = torch.load(f"{self.ground_truth_activations_path}/post_block_{block_idx}.pt")
                    activation_diff = (x - x_savanna.squeeze()).abs()
                    activations_logger.info(f"post block {block_idx} activation_diff: {activation_diff.max()}, {activation_diff.mean()}")

        return x, inference_params_dict

    def stateless_forward(self, x, padding_mask=None):
        if type(padding_mask) == torch.Tensor:
            x = x * padding_mask[..., None]

        for block_idx, block in enumerate(self.blocks):
            if self.print_activations:
                activations_logger.info(f"pre block {block_idx}: {x}, {x.min()}, {x.max()} {block.__class__}")
                if self.ground_truth_activations_path:
                    x_savanna = torch.load(f"{self.ground_truth_activations_path}/pre_block_{block_idx}.pt")
                    activation_diff = (x - x_savanna.squeeze()).abs()
                    activations_logger.info(f"pre block {block_idx} activation_diff: {activation_diff.max()}, {activation_diff.mean()}")

            x, _ = block(x, inference_params=None, padding_mask=padding_mask)
            if self.print_activations:
                activations_logger.info(f"post block {block_idx}: {x}, {x.min()}, {x.max()}")
                if self.ground_truth_activations_path:
                    x_savanna = torch.load(f"{self.ground_truth_activations_path}/post_block_{block_idx}.pt")
                    activation_diff = (x - x_savanna.squeeze()).abs()
                    activations_logger.info(f"post block {block_idx} activation_diff: {activation_diff.max()}, {activation_diff.mean()}")

        return x, None

    def initialize_inference_params(self):
        inference_params_dict = {
            "mha": InferenceParams(
                max_seqlen=self.config.get("max_seqlen", 8192),
                max_batch_size=self.config.get("max_batch_size", 1),
                seqlen_offset=0,
            ),
            "hcl": HyenaCascadeIIRInferenceParams(
                fir_filter_length=self.config.short_filter_length,
                state_dim=self.config.state_size,
                seqlen_offset=0,
            ),
            "hcm": HyenaCascadeFIRInferenceParams(
                fir_filter_length=self.config.short_filter_length,
                fir_inner_filter_length=self.config.hcm_filter_length,
                seqlen_offset=0,
            ),
            "hcs": HyenaCascadeFIRInferenceParams(
                fir_filter_length=self.config.short_filter_length,
                fir_inner_filter_length=self.config.hcs_filter_length,
                seqlen_offset=0,
            ),
        }
        return inference_params_dict

    def precompute_filters(self, L, device):
        for block_idx, block in enumerate(self.blocks):
            if type(block) == ParallelGatedConvBlock:
                if type(block.filter) == HyenaCascade:
                    L = block.filter.long_fir_threshold or L
                    print_rank_0(f"Precomputing filters, L={L}...")

                    filter_dtype = torch.float16 if L >= 2048 else torch.float32

                    block.filter._set_time(L, device)
                    residues, poles = (
                        block.filter.residues.to(torch.float16),
                        block.filter.poles.to(torch.float16),
                    )

                    block.filter.h = (residues * poles**block.filter.t).real.sum(1)[None]
                    block.filter.h = block.filter.h.to(dtype=filter_dtype)

    def load_poles_residues(self, path):
        "Load different poles and residues for each layer."
        for block_idx, block in enumerate(self.blocks):
            if type(block) == ParallelGatedConvBlock:
                if type(block.filter) == HyenaCascade:
                    self.logger.info(f"Loading approximatepoles and residues for block {block_idx}")
                    poles = torch.load(path + f"/approx_poles_{block_idx+1}.pt", map_location="cpu")
                    poles = torch.view_as_real(poles)
                    residues = torch.load(path + f"/approx_residues_{block_idx+1}.pt", map_location="cpu")
                    residues = torch.view_as_real(residues)
                    poles = poles.permute(1, 0, 2).unsqueeze(-2)
                    residues = residues.permute(1, 0, 2).unsqueeze(-2)

                    block.filter.poles = nn.Parameter(poles)
                    block.filter.residues = nn.Parameter(residues)
    
    def custom_load_state_dict(self, state_dict, strict=True):
        """
        Post-processes the state_dict to convert savanna checkpoints to vortex checkpoints.
        """
        self.logger.info(f"Loading state dict: {state_dict}, with strict: {strict}")
        self.load_state_dict(state_dict, strict=strict)

        if self.config.get("column_split", True):
            self.logger.info("Adjusting Wqkv for column split (permuting rows)")
            for layer_idx, block in enumerate(self.blocks):
                if type(block) == AttentionBlock:
                    Wqkv = state_dict[f"blocks.{layer_idx}.inner_mha_cls.Wqkv.weight"]
                    try:
                        bias = state_dict[f"blocks.{layer_idx}.inner_mha_cls.Wqkv.bias"]
                    except:
                        bias = None

                    size_att_head = block.hidden_size_per_attention_head

                    Wqkv = Wqkv.permute(1, 0)
                    Wqkv = Wqkv.reshape(block.hidden_size, block.num_attention_heads, 3, size_att_head)
                    Wq, Wk, Wv = Wqkv.unbind(dim=-2)
                    Wq = Wq.reshape(block.hidden_size, -1)
                    Wk = Wk.reshape(block.hidden_size, -1)
                    Wv = Wv.reshape(block.hidden_size, -1)
                    Wqkv = torch.cat([Wq, Wk, Wv], dim=-1)
                    Wqkv = Wqkv.permute(1, 0)
                    block.inner_mha_cls.Wqkv.weight.data = Wqkv

                    if bias is not None:
                        bias = bias.reshape(block.num_attention_heads, 3, size_att_head)
                        bias_q, bias_k, bias_v = bias.unbind(dim=-2)
                        bias_q = bias_q.reshape(block.hidden_size)
                        bias_k = bias_k.reshape(block.hidden_size)
                        bias_v = bias_v.reshape(block.hidden_size)
                        bias = torch.cat([bias_q, bias_k, bias_v], dim=0)
                        try:
                            block.inner_mha_cls.Wqkv.bias.data = bias
                        except:
                            # catch cases with strict_load False and spurious biases in the checkpoint
                            pass

    
    def to_bfloat16_except_pr_lc(self):
        """Convert all parameters to bfloat16 except for the poles and residues.

        Particularly important for longer prompts.
        """
        excluded_shapes = [
            (4096, 1, 128)
        ]
        for k, p in self.named_parameters():
            if "log_poles" not in k and "residues" not in k and p.shape not in excluded_shapes:
                p.data = p.data.to(torch.bfloat16)