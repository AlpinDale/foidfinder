"""Compute backends, imported lazily so only the selected one's dependencies are needed."""

from __future__ import annotations

import platform
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import ClassVar, Protocol

import numpy as np
from PIL import Image

NAMES = ("mlx", "cuda")


class Backend(Protocol):
    name: ClassVar[str]
    device: str

    def predict(self, image: Image.Image) -> np.ndarray:
        """Decoded image -> [num_tags] float32 probabilities. Call from one thread only."""
        ...


def detect() -> str:
    """The backend this machine can run: MLX on Apple Silicon, else CUDA if a GPU is visible."""
    if sys.platform == "darwin" and platform.machine() == "arm64" and find_spec("mlx"):
        return "mlx"
    if find_spec("torch"):
        import torch

        if torch.cuda.is_available():
            return "cuda"
    raise RuntimeError(
        "no supported accelerator: install the mlx extra on Apple Silicon, or the cuda extra on a machine with an "
        "NVIDIA GPU"
    )


def load(name: str, model_dir: Path) -> Backend:
    """name: "auto", "mlx" or "cuda"."""
    if name == "auto":
        name = detect()
    if name == "mlx":
        from .mlx import MLXBackend

        return MLXBackend(model_dir)
    if name == "cuda":
        from .cuda import CUDABackend

        return CUDABackend(model_dir)
    raise ValueError(f"unknown backend {name!r}; choose auto, {', '.join(NAMES)}")
