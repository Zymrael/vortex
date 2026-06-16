import pytest
import torch


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    Auto-skip tests marked @pytest.mark.gpu when no CUDA device is present.
    """
    _ = config
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="requires CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
