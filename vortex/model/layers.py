# Copyright (c) 2024, Michael Poli.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from vortex.model.utils import grab_first_if_tuple

try:
    from transformer_engine.pytorch import Linear
    from transformer_engine.common.recipe import Format, DelayedScaling
    import transformer_engine.pytorch as te
except:
    print("WARNING: transformer_engine not installed. Using default recipe.")
    raise NotImplementedError


def set_format_recipe():
    fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max"
    )
    return fp8_format, fp8_recipe


class TELinear(Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group().
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        init_method: Callable,
        bias: bool = True,
        skip_bias_add: bool = False,
        use_fp8: bool = False,
        **kwargs,
    ):
        # Parameters are initialized at higher precision even if fp8
        # is used
        params_dtype = torch.bfloat16

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias

        self.use_fp8_input_projections = use_fp8
        if use_fp8:
            self.fp8_format, self.fp8_recipe = set_format_recipe()

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=False,
            fuse_wgrad_accumulation=False,
            tp_group=None,
            tp_size=1,
            init_method=init_method,
            params_dtype=params_dtype,
            parallel_mode="column",
            bias=bias,
            return_bias=self.te_return_bias,
            **kwargs,
        )

    def forward(self, x):
        if self.use_fp8_input_projections:
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                out = super().forward(x)
        else:
            out = super().forward(x)

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None


class FlexLinear:
    """
    Megatron and Transformer Engine linear layer compatible with fp8, bf16, fp16 and fp32
    """

    def __new__(
        self,
        input_size,
        output_size,
        config,
        parallel_mode: str,
        bias: bool = False,
        skip_bias_add: bool = True,
        use_fp8: bool = False,
        input_is_parallel=False,  # for row parallel
        gather_output: bool = True,  # for column parallel
        parallel_output: bool = False,  # for row parallel
        **kwargs,
    ):
        # use_fp8 = config.use_fp8_linears
        self.config = config
        instance = None

        if use_fp8:
            instance = TELinear(
                input_size=input_size,
                output_size=output_size,
                config=self.config,
                parallel_mode=parallel_mode,
                bias=bias,
                skip_bias_add=skip_bias_add,
                **kwargs,
            )

        return instance


class RMSNorm(torch.nn.Module):
    def __init__(self, config):
        super(RMSNorm, self).__init__()
        self.eps, self.hidden_size = config.eps, config.hidden_size
        self.scale = torch.nn.Parameter(torch.ones(self.hidden_size))
        self.register_parameter("scale", self.scale)
        self.scale = self.scale.to(config.params_dtype)
        self.use_flash_rmsnorm = config.get("use_flash_rmsnorm", False)

        if self.use_flash_rmsnorm:
            from flash_attn.ops.rms_norm import rms_norm as rmsnorm_func

            self.rmsnorm_func = rmsnorm_func

    def forward(self, x):
        if self.use_flash_rmsnorm:
            return self.rmsnorm_func(x, self.scale, self.eps)
        else:
            y = x / (
                x.norm(2, dim=-1, keepdim=True) * self.hidden_size ** (-1.0 / 2)
                + self.eps
            )
            return self.scale * y


class ParallelGatedMLP(nn.Module):
    def __init__(
        self,
        config,
        layer_idx,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        multiple_of = config.get("inner_size_multiple_of", 64)
        self.act_type = config.get("mlp_activation", "gelu")
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError

        self.multiple_of = multiple_of * config.model_parallel_size

        inner_size = int(2 * config.hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        )
        inner_size = config.get("inner_mlp_size", inner_size)

        self.l1 = nn.Linear(
            in_features=config.hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=config.hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=config.hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        z1, z2 = grab_first_if_tuple(z1), grab_first_if_tuple(z2)
        if self.layer_idx > 0:
            self.act = nn.Identity()
        y = self.l3(self.act(z1) * z2)
        return grab_first_if_tuple(y)


class Embedding(nn.Module):
    _train_dtype = "bf16"

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )

    def embed(self, input_ids, position_ids=None, tokentype_ids=None):
        embeddings = self.word_embeddings(input_ids)
        return embeddings

    def unembed(self, u):
        weight = self.word_embeddings.weight
        return torch.matmul(u, weight)


class VocabParallelEmbedding(nn.Embedding):
    "Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/embedding.py"

    def __init__(self, config):
        vocab_size, process_group, padding_idx = (
            config.vocab_size,
            config.get("process_group", None),
            config.get("padding_idx", None),
        )
        self.process_group = process_group
        if process_group is not None:
            world_size = torch.distributed.get_world_size(process_group)
            if vocab_size % world_size != 0:
                raise ValueError(
                    f"vocab_size ({vocab_size}) must be divisible by "
                    f"world_size ({world_size})"
                )
            if world_size > 1 and padding_idx is not None:
                raise RuntimeError("ParallelEmbedding does not support padding_idx")
        else:
            world_size = 1
        super().__init__(
            vocab_size // world_size,
            embedding_dim=config.hidden_size,
            padding_idx=padding_idx,
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.process_group is None:
            return super().forward(input)
        else:
            rank = torch.distributed.get_rank(self.process_group)
            vocab_size = self.num_embeddings
            vocab_start_index, vocab_end_index = (
                rank * vocab_size,
                (rank + 1) * vocab_size,
            )
            # Create a mask of valid vocab ids (1 means it needs to be masked).
            input_ids_mask = (input < vocab_start_index) | (input >= vocab_end_index)
            input = input - vocab_start_index
            input[input_ids_mask] = 0
            embeddings = self.forward(input)
            embeddings[input_ids_mask] = 0.0
            # Reduce to the global process group
            torch.distributed.all_reduce(embeddings, group=self.process_group)
            return embeddings

    def unembed(self, u: Tensor) -> Tensor:
        if self.process_group is None:
            return u @ self.weight.T
        else:
            raise NotImplementedError


class VocabParallelUnembedding(VocabParallelEmbedding):
    def forward(self, input: Tensor) -> Tensor:
        return self.unembed(input)
