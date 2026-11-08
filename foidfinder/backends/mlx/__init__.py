"""MLX backend: PixAI or Haruka in fp16 on the Apple Silicon GPU, chosen by the model directory's config.json."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .haruka import HarukaMLX
from .model import PixAIMLX


class MLXBackend:
    name = "mlx"

    def __init__(self, model_dir: Path):
        config = json.loads((Path(model_dir) / "config.json").read_text())
        haruka = config.get("model", {}).get("architecture") == "haruka-eva02-tree"
        self.model = (HarukaMLX if haruka else PixAIMLX)(model_dir, mx.float16)
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False)
        self.device = chip.stdout.strip() or "Apple GPU"

    def predict(self, image: Image.Image) -> np.ndarray:
        return np.array(self.model(self.model.preprocess(image)[None]))[0]
