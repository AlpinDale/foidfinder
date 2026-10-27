"""MLX backend: PixAI in fp16 on the Apple Silicon GPU."""

from __future__ import annotations

import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .model import PixAIMLX


class MLXBackend:
    name = "mlx"

    def __init__(self, model_dir: Path):
        self.model = PixAIMLX(model_dir, mx.float16)
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False)
        self.device = chip.stdout.strip() or "Apple GPU"

    def predict(self, image: Image.Image) -> np.ndarray:
        return np.array(self.model(self.model.preprocess(image)[None]))[0]
