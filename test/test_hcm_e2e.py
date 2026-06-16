"""
End-to-end test for the use_hcm_kernel flag inside a real Evo2 model.

Loads an Evo2 checkpoint once and runs the same forward twice -- with the
HCM kernel off (stock fftconv_func) and on (the fused Triton HCM FFT-conv).
The flag changes only the HCM conv implementation -- a fused kernel that is
numerically equivalent to fftconv_func but not bit-exact -- so the model's
token predictions must agree and the logit vectors stay near-parallel.

The model id defaults to evo2_7b (the variant cached on the dev box) and is
overridable with VK_E2E_MODEL. The checkpoint is large; the test skips
cleanly if it cannot be loaded.
"""

import os

import pytest
import torch

from vortex.model.engine import HyenaInferenceEngine

_MODEL_ID: str = os.environ.get("VK_E2E_MODEL", "evo2_7b")
_SEQ_LEN: int = 2048


@pytest.fixture(scope="module")
def evo2_model():
    """
    Load the Evo2 model once for the module, or skip if unavailable.
    """
    if not torch.cuda.is_available():
        pytest.skip("Evo2 e2e test requires CUDA")
    try:
        from evo2 import Evo2
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        pytest.skip(f"evo2 not installed: {exc}")
    try:
        return Evo2(_MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - any load failure -> skip, not fail
        pytest.skip(f"could not load Evo2({_MODEL_ID!r}): {exc}")


def _set_hcm_kernel(model, enabled: bool) -> int:
    """
    Flip use_hcm_kernel on every HyenaInferenceEngine reachable from model.

    The flag is read off the engine instance (normally set from the
    use_hcm_kernel config key at build time); toggling it on the loaded
    model lets one checkpoint serve both halves of the comparison.

    Args:
        model: A loaded Evo2 model.
        enabled (bool): Target value for use_hcm_kernel.

    Returns:
        The number of HyenaInferenceEngine instances touched.
    """
    root = getattr(model, "model", model)
    touched = 0
    for module in root.modules():
        engine = getattr(module, "engine", None)
        if isinstance(engine, HyenaInferenceEngine):
            engine.use_hcm_kernel = enabled
            touched += 1
    return touched


def _logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Run a forward pass and return the logits tensor as fp32.

    Evo2 wraps its output in nested tuples, so descend to the first tensor.
    """
    with torch.no_grad():
        out = model(input_ids)
    while isinstance(out, (tuple, list)):
        out = out[0]
    return out.float()


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_vk_hcm_e2e_matches_baseline(evo2_model) -> None:
    """
    A full Evo2 forward is behaviourally unchanged when use_hcm_kernel swaps
    in the fused HCM kernel.

    hcm_fft_conv is numerically equivalent to fftconv_func (~2e-7 relative),
    not bit-exact. One HCM layer carries ~1e5-magnitude activations, so that
    relative precision becomes a sizeable absolute logit difference once the
    downstream blocks amplify it -- an absolute logit tolerance is meaningless
    here. The behavioural invariant is tested instead: the model predicts the
    same tokens and the logit vectors stay near-parallel.
    """
    torch.manual_seed(0)
    input_ids = torch.randint(1, 5, (1, _SEQ_LEN), dtype=torch.int, device="cuda:0")

    try:
        touched = _set_hcm_kernel(evo2_model, False)
        assert touched > 0, "no HyenaInferenceEngine found in the Evo2 model"
        logits_off = _logits(evo2_model, input_ids)

        _set_hcm_kernel(evo2_model, True)
        logits_on = _logits(evo2_model, input_ids)
    finally:
        _set_hcm_kernel(evo2_model, False)

    assert logits_on.shape == logits_off.shape

    agreement = (logits_on.argmax(-1) == logits_off.argmax(-1)).float().mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        logits_on.flatten(), logits_off.flatten(), dim=0
    ).item()
    assert agreement >= 0.99, (
        f"use_hcm_kernel changed {(1 - agreement) * 100:.2f}% of token predictions"
    )
    assert cosine >= 0.9999, f"use_hcm_kernel logits diverged: cosine={cosine:.6f}"
