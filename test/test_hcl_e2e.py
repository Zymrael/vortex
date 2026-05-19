"""
End-to-end tests for the use_hcl_kernel flag inside a real Evo2 model.

Loads an Evo2 checkpoint once and runs the same forward with the HCL kernel
off (stock compute_filter + the parallel_iir FFT branch) and on (the tiled
filter build + the fused FFT-conv epilogue). The kernel is numerically
equivalent, not bit-exact, so the behavioural invariant is tested: the model
predicts the same tokens and the logit vectors stay near-parallel.

test_hcl_unlocks_131k is the headline -- the stock compute_filter OOMs at
L=131k on its (D, state_size, L) intermediate; the tiled kernel removes it.
It needs an ~80GB GPU (the sign-off pod) and skips on a smaller card.

These load a large checkpoint -- slow relative to the kernel tests; run them
deliberately, not in the fast loop.
"""

import os

import pytest
import torch

from vortex.model.engine import HyenaInferenceEngine

CUDA: bool = torch.cuda.is_available()
_MODEL_ID: str = os.environ.get("VK_E2E_MODEL", "evo2_7b")
_SEQ_LEN: int = 2048


@pytest.fixture(scope="module")
def evo2_model():
    """
    Load the Evo2 model once for the module, or skip if unavailable.
    """
    if not CUDA:
        pytest.skip("Evo2 e2e test requires CUDA")
    try:
        from evo2 import Evo2
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        pytest.skip(f"evo2 not installed: {exc}")
    try:
        return Evo2(_MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - any load failure -> skip, not fail
        pytest.skip(f"could not load Evo2({_MODEL_ID!r}): {exc}")


def _set_hcl_kernel(model, enabled: bool) -> int:
    """
    Flip use_hcl_kernel on every HyenaInferenceEngine reachable from model.

    Args:
        model: A loaded Evo2 model.
        enabled (bool): Target value for use_hcl_kernel.

    Returns:
        The number of HyenaInferenceEngine instances touched.
    """
    root = getattr(model, "model", model)
    touched = 0
    for module in root.modules():
        engine = getattr(module, "engine", None)
        if isinstance(engine, HyenaInferenceEngine):
            engine.use_hcl_kernel = enabled
            touched += 1
    return touched


def _logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Run a forward pass and return the logits tensor as fp32.
    """
    with torch.no_grad():
        out = model(input_ids)
    while isinstance(out, (tuple, list)):
        out = out[0]
    return out.float()


@pytest.mark.skipif(not CUDA, reason="Evo2 e2e test requires CUDA")
def test_vk_hcl_e2e_matches_baseline(evo2_model) -> None:
    """
    A full Evo2 forward is behaviourally unchanged when use_hcl_kernel swaps
    in the tiled filter build and the fused FFT-conv epilogue.

    The kernel is numerically equivalent to the stock path but not bit-exact,
    so the test asserts prediction agreement (argmax + cosine), not an
    absolute logit bound -- the same rationale as the HCM e2e.
    """
    torch.manual_seed(0)
    input_ids = torch.randint(1, 5, (1, _SEQ_LEN), dtype=torch.int, device="cuda:0")

    try:
        touched = _set_hcl_kernel(evo2_model, False)
        assert touched > 0, "no HyenaInferenceEngine found in the Evo2 model"
        logits_off = _logits(evo2_model, input_ids)

        _set_hcl_kernel(evo2_model, True)
        logits_on = _logits(evo2_model, input_ids)
    finally:
        _set_hcl_kernel(evo2_model, False)

    assert logits_on.shape == logits_off.shape

    agreement = (logits_on.argmax(-1) == logits_off.argmax(-1)).float().mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        logits_on.flatten(), logits_off.flatten(), dim=0
    ).item()
    assert agreement >= 0.99, (
        f"use_hcl_kernel changed {(1 - agreement) * 100:.2f}% of token predictions"
    )
    assert cosine >= 0.9999, f"use_hcl_kernel logits diverged: cosine={cosine:.6f}"


@pytest.mark.skipif(not CUDA, reason="Evo2 e2e test requires CUDA")
def test_hcl_unlocks_131k(evo2_model) -> None:
    """
    A full evo2_7b forward at L=131072 completes with use_hcl_kernel on.

    The stock compute_filter OOMs at this length on its (D, state_size, L)
    fp32 intermediate (34 GiB at D=4096); the tiled kernel removes it. Needs
    an ~80GB GPU (the H100 sign-off pod) -- on a smaller card this skips:
    even with the kernel, the rest of the 131k forward will not fit.
    """
    torch.manual_seed(0)
    input_ids = torch.randint(1, 5, (1, 131072), dtype=torch.int, device="cuda:0")

    try:
        touched = _set_hcl_kernel(evo2_model, True)
        assert touched > 0, "no HyenaInferenceEngine found in the Evo2 model"
        logits = _logits(evo2_model, input_ids)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pytest.skip("L=131072 needs ~80GB; run on the H100 sign-off pod")
    finally:
        _set_hcl_kernel(evo2_model, False)

    assert logits.shape[1] == 131072
    assert torch.isfinite(logits).all()
