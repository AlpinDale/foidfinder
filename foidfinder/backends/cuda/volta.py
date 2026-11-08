"""FP16 inference on Volta, which lacks the instructions used by the int8 engine."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .preprocess import preprocess
from .reference import PixAITorch


class PixAIVolta:
    @torch.inference_mode()
    def __init__(self, model_dir: Path):
        self.model = PixAITorch(model_dir).eval()
        self.input = torch.zeros(1, 1008, 1008, 3, device="cuda", dtype=torch.float16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.model(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.probs = self.model(self.input).sigmoid()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> np.ndarray:
        self.input.copy_(preprocess([image]))
        self.graph.replay()
        return self.probs.cpu().numpy()[0]
