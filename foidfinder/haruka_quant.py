"""Compressed Haruka checkpoint for downloading into the browser.

Each large matrix (backbone linears, pool key/value, tag classifier) is quantized with GPTQ to integer codes times
one fp16 step per output row, step = rms(row) / divisor. The codes are zigzag-mapped to unsigned integers, split
into byte planes and each plane gzipped, so the browser's DecompressionStream decodes them. Weights are
reconstructed once at load (code * step, fp16); the runtime multiplies ordinary fp16 matrices.

model.safetensors:
  <name>.step        f16 [rows]
  <name>.plane<i>    u8  [gzip bytes]  byte i of zigzag(code), row-major [rows, cols]
  everything else    f16, as written by foidfinder.haruka.convert
config.json["quantization"]: {"format", "layers": {name: {"shape": [rows, cols], "planes": n}}}
"""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from .haruka import METADATA, MODEL, THRESHOLDS, WEIGHTS, fold, vocab

FORMAT = "foidfinder-haruka-rowstep1"
# Step = rms(row) / divisor. Per weight, the backbone is about 4x as sensitive as the classifier, so it gets a
# finer step; see the README's "Browser checkpoint" section for the measurements.
DIVISOR = {"backbone": 40, "head": 32}


def quantized_layers(depth: int = MODEL["depth"]) -> list[str]:
    """In forward (addmm call) order."""
    blocks = [f"blocks.{i}.{n}.weight" for i in range(depth) for n in ("qkv", "proj", "fc1", "fc2")]
    return [*blocks, "pool.kv.weight", "head.weight"]


def gptq(W: np.ndarray, H: np.ndarray, step: np.ndarray, damp: float = 0.01, block: int = 128) -> np.ndarray:
    """GPTQ (Frantar et al.) onto the grid code * step[row], with no clipping: int32 codes [rows, cols].
    H = X^T X of the layer's inputs; columns are visited in decreasing diag(H) order."""
    W = np.array(W, np.float64)
    H = np.array(H, np.float64)
    rows, cols = W.shape
    dead = np.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    perm = np.argsort(-np.diag(H), kind="stable")
    W, H = W[:, perm], H[perm][:, perm]
    H[np.diag_indices(cols)] += damp * np.diag(H).mean()
    U = np.linalg.cholesky(np.linalg.inv(H)).T  # upper Cholesky factor of H^-1
    codes = np.zeros((rows, cols), np.int32)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1, U1 = W[:, i1:i2].copy(), U[i1:i2, i1:i2]
        Err = np.zeros_like(W1)
        for i in range(i2 - i1):
            c = np.round(W1[:, i] / step)
            codes[:, i1 + i] = c
            err = (W1[:, i] - c * step) / U1[i, i]
            W1[:, i:] -= np.outer(err, U1[i, i:])
            Err[:, i] = err
        W[:, i2:] -= Err @ U[i1:i2, i2:]
    return codes[:, np.argsort(perm)]


def row_step(W: np.ndarray, divisor: int) -> np.ndarray:
    return (np.sqrt((W.astype(np.float64) ** 2).mean(1)) / divisor).astype(np.float16)


def encode(codes: np.ndarray) -> list[bytes]:
    """int codes -> gzipped byte planes of zigzag(code)."""
    c = codes.astype(np.int64).ravel()
    z = ((c << 1) ^ (c >> 63)).astype(np.uint32)
    n = max(1, (int(z.max()).bit_length() + 7) // 8)
    return [gzip.compress(((z >> (8 * i)) & 0xFF).astype(np.uint8).tobytes(), 9, mtime=0) for i in range(n)]


def decode(planes: list[bytes], shape: tuple[int, int]) -> np.ndarray:
    z = np.zeros(shape[0] * shape[1], np.uint32)
    for i, p in enumerate(planes):
        z |= np.frombuffer(gzip.decompress(p), np.uint8).astype(np.uint32) << (8 * i)
    return ((z >> 1).astype(np.int32) ^ -(z & 1).astype(np.int32)).reshape(shape)


def capture_hessians(model_dir: Path, images: Iterable[Path], names: list[str]) -> tuple[dict[str, np.ndarray], int]:
    """H = X^T X of every quantized matrix's input over the calibration images (fp32 MLX forward)."""
    import mlx.core as mx

    from .backends.mlx import haruka as hk
    from .image import decode as decode_image

    model = hk.HarukaMLX(model_dir, mx.float32)
    calls: list = []

    class Capture:  # every linear in the forward is an addmm, in quantized_layers() order
        def __getattr__(self, name):
            return getattr(mx, name)

        @staticmethod
        def addmm(c, a, b):
            calls.append(a)
            return mx.addmm(c, a, b)

    H: dict[str, mx.array] = {}
    n = 0
    hk.mx = Capture()
    try:
        for path in images:
            calls.clear()
            model._forward_impl(model.w, model.preprocess(decode_image(path.read_bytes()))[None])
            for k, a in zip(names, calls, strict=True):
                a = a.reshape(-1, a.shape[-1]).astype(mx.float32)
                H[k] = a.T @ a if k not in H else H[k] + a.T @ a
            mx.eval(list(H.values()))
            n += 1
    finally:
        hk.mx = mx
    return {k: np.array(v, np.float64) for k, v in H.items()}, n


def quantize(src: Path, out_dir: Path, calib: list[Path], log: Callable[[str], None] = print) -> None:
    """src: a download of KaraKaraWitch/haruka-tagger; calib: calibration images (about 1,500, like the traffic)."""
    from safetensors.numpy import load_file, save_file

    from .haruka import convert

    folded = fold(load_file(str(src / WEIGHTS)))
    staging = out_dir / ".fp32"
    convert(src, staging, np.float32)  # the fp32 model whose activations calibrate GPTQ
    names = quantized_layers()
    t = time.time()
    H, n = capture_hessians(staging, calib, names)
    log(f"calibrated on {n} images in {time.time() - t:.0f}s")

    tensors: dict[str, np.ndarray] = {}
    layers: dict[str, dict] = {}
    for k in names:
        W = folded.pop(k)
        step = row_step(W, DIVISOR["head" if k == "head.weight" else "backbone"])
        planes = encode(gptq(W, H[k], step.astype(np.float64)))
        tensors[k + ".step"] = step
        for i, p in enumerate(planes):
            tensors[f"{k}.plane{i}"] = np.frombuffer(p, np.uint8)
        layers[k] = {"shape": list(W.shape), "planes": len(planes)}
        log(f"{k}: {sum(map(len, planes)) * 8 / W.size:.2f} bits/weight")
    tensors.update({k: np.ascontiguousarray(v, dtype=np.float16) for k, v in folded.items()})
    save_file(tensors, str(out_dir / "model.safetensors"), {"format": FORMAT})
    tags, categories = vocab(src / METADATA)
    config = {
        "model": MODEL,
        "quantization": {"format": FORMAT, "layers": layers},
        "tags": tags,
        "categories": categories,
        "thresholds": THRESHOLDS,
    }
    (out_dir / "config.json").write_text(json.dumps(config))
    for f in staging.iterdir():
        f.unlink()
    staging.rmdir()


def load(model_dir: Path) -> dict[str, np.ndarray]:
    """All weights of a compressed checkpoint, the quantized matrices reconstructed as fp32 code * step."""
    from safetensors.numpy import load_file

    layers = json.loads((model_dir / "config.json").read_text())["quantization"]["layers"]
    raw = load_file(str(model_dir / "model.safetensors"))
    out = {k: v for k, v in raw.items() if not k.split(".")[-1].startswith(("step", "plane"))}
    for k, spec in layers.items():
        planes = [raw[f"{k}.plane{i}"].tobytes() for i in range(spec["planes"])]
        out[k] = decode(planes, tuple(spec["shape"])) * raw[k + ".step"].astype(np.float32)[:, None]
    return out
