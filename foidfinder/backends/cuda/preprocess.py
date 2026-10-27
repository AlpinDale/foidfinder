"""PixAI's RescalePadProcessor on the GPU. torchvision's resize is F.interpolate(antialias=True), so this
matches the reference up to float rounding."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...image import rgb_on_white

SIZE = 1008


@torch.no_grad()
def preprocess(images: list[Image.Image], device: str = "cuda") -> torch.Tensor:
    """PIL images -> [B, 1008, 1008, 3] fp16 normalized pixels (the engine's input layout)."""
    out = torch.full(
        (len(images), SIZE, SIZE, 3), -1.0, device=device, dtype=torch.half
    )  # -1 is black after normalization
    for b, image in enumerate(images):
        rgb = torch.from_numpy(np.array(rgb_on_white(image))).to(device, non_blocking=True)
        x = rgb.permute(2, 0, 1)[None].float().div_(255.0)
        h, w = x.shape[-2:]
        r = min(SIZE / h, SIZE / w)
        nh, nw = int(h * r), int(w * r)
        if (nh, nw) != (h, w):
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
        top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
        out[b, top : top + nh, left : left + nw] = (x[0].permute(1, 2, 0) * 2.0 - 1.0).half()
    return out
