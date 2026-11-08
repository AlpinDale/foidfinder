"""CUDA inference: FP16 on Volta, custom int8 kernels on compute capability 8.0+."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .preprocess import preprocess


class CUDABackend:
    name = "cuda"

    def __init__(self, model_dir: Path):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend: no NVIDIA GPU visible")
        if "categories" in json.loads((Path(model_dir) / "config.json").read_text()):
            raise RuntimeError("Haruka Tagger runs on the MLX backend only")
        self.volta = None
        capability = torch.cuda.get_device_capability()
        if capability == (7, 0):
            if "sm_70" not in torch.cuda.get_arch_list():
                raise RuntimeError("Volta requires the volta extra (PyTorch 2.8 with CUDA 12.6)")
            from .volta import PixAIVolta

            self.volta = PixAIVolta(model_dir)
        elif capability >= (8, 0):
            from .engine import PixAICuda

            self.engine = PixAICuda(model_dir)
        else:
            raise RuntimeError("CUDA backend supports Volta (7.0) or compute capability 8.0+")
        self.device = torch.cuda.get_device_name()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> np.ndarray:
        if self.volta is not None:
            return self.volta.predict(image)
        return torch.sigmoid(self.engine(preprocess([image])))[0].cpu().numpy()
