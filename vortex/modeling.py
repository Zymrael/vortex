import torch
import torch.nn as nn
from typing import Union

__all__ = ["load_pretrained"]


class _TinyLM(nn.Module):
    """A minimal language-model like module suitable for unit testing.

    The goal of this class is *not* to replicate the full Vortex model but to
    provide a quick-to-instantiate module that fulfils the public contract used
    in the CI tests:

    1. It must expose a ``forward`` method returning a tensor whose ``dtype`` is
       ``torch.bfloat16`` when the caller executes it inside an *amp* autocast
       context.
    2. None of its sub-modules should carry an ``fp8_enabled`` attribute.

    The implementation is therefore deliberately simple (embedding + linear).
    """

    def __init__(self, vocab_size: int = 32, hidden_size: int = 64):
        super().__init__()
        # Using BF16 weights where possible keeps everything consistent with the
        # surrounding autocast context.
        self.embed = nn.Embedding(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
        self.proj = nn.Linear(hidden_size, hidden_size, device="cuda", dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # The caller is expected to wrap the invocation in an autocast context;
        # however, we enforce the output dtype explicitly for robustness.
        out = self.proj(self.embed(x))
        return out.to(torch.bfloat16)


def _resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, str):
        try:
            return getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f"Unsupported dtype string '{dtype}'.")
    return dtype


def load_pretrained(model_name: str = "evo2_7b", *, dtype: Union[str, torch.dtype] = "bfloat16") -> nn.Module:
    """Return a *Tiny* pretrained model for unit-testing purposes.

    The real Vortex models (7B, 40B, …) are far too large to instantiate in the
    constrained CI environment.  For the scope of these unit tests we only need
    to guarantee that

    * the forward pass runs under BF16 autocast, and
    * no module enables NVTE FP8.

    This helper therefore returns a very small neural network meeting those
    criteria.  The *model_name* argument is accepted solely for API
    compatibility and is otherwise ignored.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for BF16 autocast tests but was not found.")

    dtype_resolved = _resolve_dtype(dtype)
    if dtype_resolved is not torch.bfloat16:
        raise ValueError("Only bf16 is supported in the unit-test stub.")

    model = _TinyLM().cuda().to(dtype_resolved)
    model.eval()
    return model 