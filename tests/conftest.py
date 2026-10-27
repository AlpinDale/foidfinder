import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter


@pytest.fixture(scope="session")
def official_model_dir() -> Path:
    """The official checkpoint with its reference code (`foidfinder download --reference`), or skip."""
    model_dir = Path(os.environ.get("FOIDFINDER_MODEL_DIR", "models/pixai-tagger-v1.0"))
    if not (model_dir / "tagger_pipeline.py").exists():
        pytest.skip(f"{model_dir} lacks the official model and code (foidfinder download --reference)")
    return model_dir


@pytest.fixture(scope="session")
def sample_images() -> list[Image.Image]:
    """Downscale, upscale and exact-size paths of the resize, plus transparency composited onto white."""
    rng = np.random.default_rng(0)
    blur = ImageFilter.GaussianBlur(3)
    images = [
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).filter(blur)
        for w, h in ((1500, 2100), (700, 400), (1008, 1008))
    ]
    return [*images, Image.fromarray(rng.integers(0, 256, (600, 900, 4), dtype=np.uint8))]
