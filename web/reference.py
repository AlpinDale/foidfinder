"""Reference data for bench.html: foidfinder's preprocessing and the fp32 MLX forward of a compressed Haruka
checkpoint, for the in-browser engine to match.

usage: uv run python web/reference.py <model dir from quantize-haruka> <image dir> [count]
Writes web/dev-ref/: pixels.u8 [n, 448, 448, 3], probs.f32 [n, tags], copies of the images, manifest.json.
"""

import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from foidfinder.backends.mlx.haruka import HarukaMLX
from foidfinder.image import decode

model_dir, image_dir = Path(sys.argv[1]), Path(sys.argv[2])
count = int(sys.argv[3]) if len(sys.argv) > 3 else 64
out = Path(__file__).parent / "dev-ref"
out.mkdir(exist_ok=True)

paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"})[:count]
model = HarukaMLX(model_dir, mx.float32)
pixels = np.stack([np.array(model.preprocess(decode(p.read_bytes()))) for p in paths])
probs = np.concatenate([np.array(model(mx.array(pixels[i : i + 1]))) for i in range(len(paths))])
pixels.tofile(out / "pixels.u8")
probs.astype(np.float32).tofile(out / "probs.f32")
for p in paths:
    shutil.copy(p, out / p.name)
manifest = {"n": len(paths), "files": [p.name for p in paths], "tags": int(probs.shape[1])}
(out / "manifest.json").write_text(json.dumps(manifest))
print(f"wrote {out}: {len(paths)} images")
