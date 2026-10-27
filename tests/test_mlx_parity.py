"""MLX backend vs the official PyTorch implementation. Needs `uv sync --extra mlx --group parity`."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
mx = pytest.importorskip("mlx.core")

from foidfinder.backends.mlx.model import PixAIMLX


@pytest.fixture(scope="module")
def reference(official_model_dir, sample_images):
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(official_model_dir, trust_remote_code=True, dtype=torch.float32).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(official_model_dir, trust_remote_code=True)
    with torch.no_grad():
        model = model.to(device)
        return [
            torch.sigmoid(model(proc(im)["pixel_values"].to(device))).float().cpu().numpy()[0] for im in sample_images
        ]


@pytest.mark.parametrize(("dtype", "tol"), [("float32", 1e-4), ("float16", 1e-2)])
def test_matches_reference(official_model_dir, sample_images, reference, dtype, tol):
    ours = PixAIMLX(official_model_dir, dtype=getattr(mx, dtype))
    for image, ref in zip(sample_images, reference, strict=True):
        got = np.array(ours(ours.preprocess(image)[None]))[0]
        assert np.abs(got - ref).max() < tol
        # Within tol of 0.5, rounding may flip a decision.
        assert np.all(got[ref > 0.5 + tol] > 0.5) and np.all(got[ref < 0.5 - tol] <= 0.5)
