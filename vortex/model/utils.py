import torch


def grab_first_if_tuple(x):
    if x.__class__.__name__ == "tuple":
        return x[0]
    else:
        return x


def interleave(z_pre):
    if len(z_pre.shape) == 3:  # non-cached
        x1 = z_pre[:, 0::3, :]
        x2 = z_pre[:, 1::3, :]
        v = z_pre[:, 2::3, :]
        z_pre = torch.cat([x1, x2, v], dim=1)
        return z_pre
    else:
        x1 = z_pre[..., 0::3]
        x2 = z_pre[..., 1::3]
        v = z_pre[..., 2::3]
        z_pre = torch.concat([x1, x2, v], dim=-1)
        return z_pre


def column_split(x, num_heads, head_size):
    """Split a tensor with `num_heads` alongside the head dimension, instead of
    across heads. Fixed to three projections
    """
    # FIXME: merge cases
    if len(x.shape) == 2:
        x_reshaped = x.reshape(
            x.shape[0],
            num_heads,
            3 * head_size,
        )

        x2, x1, v = (
            x_reshaped[..., :head_size],
            x_reshaped[..., head_size : 2 * head_size],
            x_reshaped[..., 2 * head_size :],
        )
        x2, x1, v = (
            x2.reshape(x2.shape[0], -1),
            x1.reshape(x1.shape[0], -1),
            v.reshape(v.shape[0], -1),
        )
        return x2, x1, v
    else:
        x = x.reshape(
            x.shape[0],
            num_heads,
            3 * head_size,
            x.shape[2],
        )
        x2, x1, v = (
            x[:, :, :head_size],
            x[
                :,
                :,
                head_size : 2 * head_size,
            ],
            x[:, :, 2 * head_size :],
        )
        x2, x1, v = (
            x2.reshape(x2.shape[0], -1, x2.shape[-1]),
            x1.reshape(x1.shape[0], -1, x1.shape[-1]),
            v.reshape(v.shape[0], -1, v.shape[-1]),
        )
        return x2, x1, v


def get_init_from_string(init_str):
    if type(init_str) == str:
        if init_str == "torch.nn.init.zeros_":
            return torch.nn.init.zeros_
        elif init_str == "torch.nn.init.xavier_uniform_":
            return torch.nn.init.xavier_uniform_
        elif init_str == "torch.nn.init.xavier_normal_":
            return torch.nn.init.xavier_normal_
        else:
            raise ValueError(f"Unrecognized init {init_str}")


def print_rank_0(message, debug=False, end="\n"):
    """Print from rank 0 only."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True, end=end)
    else:
        print(message, flush=True, end=end)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class VocabUtility:
    """Split the vocabulary into `world_size` chunks amd return the
    first and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [first, last]"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size, rank, world_size
    ):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )
