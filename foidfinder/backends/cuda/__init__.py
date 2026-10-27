"""CUDA backend: PixAI through the int8 engine (engine.py) on NVIDIA GPUs with compute capability 8.0+."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .engine import PixAICuda
from .preprocess import preprocess


class CUDABackend:
    name = "cuda"

    def __init__(self, model_dir: Path):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend: no NVIDIA GPU visible")
        if torch.cuda.get_device_capability() < (8, 0):
            raise RuntimeError(f"CUDA backend needs compute capability 8.0+, found {torch.cuda.get_device_name()}")
        self.engine = PixAICuda(model_dir)
        self.device = torch.cuda.get_device_name()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> np.ndarray:
        return torch.sigmoid(self.engine(preprocess([image])))[0].cpu().numpy()
