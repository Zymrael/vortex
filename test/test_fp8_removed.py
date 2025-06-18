"""Tests ensuring that the codebase no longer depends on NVTE's FP8 pathway.

This test instantiates a tiny pretrained model via ``vortex.modeling.load_pretrained`` and
verifies two conditions:

1. A forward pass executed under BF16 autocast returns BF16 tensors.
2. No sub-module in the model exposes ``fp8_enabled = True``.
"""

import pathlib
import sys

import torch

# Ensure the repository root is importable regardless of where pytest is invoked
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if _REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, _REPO_ROOT.as_posix())

from vortex.modeling import load_pretrained


def test_forward_bf16():
    model = load_pretrained("evo2_7b", dtype="bfloat16")

    x = torch.randint(0, 20, (1, 16), device="cuda")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        y = model(x)

    assert y.dtype == torch.bfloat16, "Output must be in BF16 precision"
    for m in model.modules():
        assert not getattr(m, "fp8_enabled", False), "FP8 must be disabled in all modules" 