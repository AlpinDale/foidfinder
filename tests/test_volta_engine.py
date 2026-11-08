"""Volta FP16 inference against the FP32 implementation, including CUDA graph replay across images."""

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
    pytest.skip("needs a Volta GPU", allow_module_level=True)


def test_graph_replay_tracks_fp32(sample_images):
    from foidfinder.backends.cuda import CUDABackend
    from foidfinder.backends.cuda.preprocess import preprocess
    from foidfinder.backends.cuda.reference import PixAITorch

    model_dir = Path(os.environ.get("FOIDFINDER_MODEL_DIR", "models/pixai-tagger-v1.0"))
    if not (model_dir / "model.safetensors").is_file():
        pytest.skip("official checkpoint is missing")
    backend = CUDABackend(model_dir)
    ref = PixAITorch(model_dir, dtype=torch.float32).eval()
    images = sample_images[:2]
    with torch.inference_mode():
        expected = [ref(preprocess([im]).float()).sigmoid().cpu().numpy()[0] for im in images]
    first = backend.predict(images[0])
    second = backend.predict(images[1])
    repeated = backend.predict(images[0])
    np.testing.assert_allclose(first, expected[0], atol=0.02, rtol=0)
    np.testing.assert_allclose(second, expected[1], atol=0.02, rtol=0)
    np.testing.assert_array_equal(first, repeated)
