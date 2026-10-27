"""PixAI Tagger v1.0 (SAM3 ViTDet classifier) in MLX.

The 72x72 patch grid splits exactly into 3x3 windows of 24x24, so tokens are permuted into window order once and
never back. Windowed attention is then a reshape. Global blocks and the pooling head only need their RoPE and
position tables permuted the same way.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from ...image import rgb_on_white

Weights = dict[str, Any]

_ROPE_THETA = 10000.0


def _expect(config: dict, key: str, value) -> None:
    if config.get(key) != value:
        raise ValueError(f"unsupported PixAI config: {key}={config.get(key)!r}, expected {value!r}")


def _axial_rope(end: int, scale: float, head_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin tables [end*end, head_dim/2] for SAM3's axial 2D RoPE (x freqs, then y freqs)."""
    freqs = 1.0 / (_ROPE_THETA ** (np.arange(0, head_dim, 4)[: head_dim // 4].astype(np.float32) / head_dim))
    t = np.arange(end * end, dtype=np.float32)
    tx = (t % end) * np.float32(scale)
    ty = np.floor(t / end) * np.float32(scale)
    angles = np.concatenate([np.outer(tx, freqs), np.outer(ty, freqs)], axis=1)
    return np.cos(angles), np.sin(angles)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate interleaved (even, odd) pairs of the last dim. x: [B, H, L, D]; cos/sin: [L, D/2]."""
    shape = x.shape
    x = x.reshape(*shape[:-1], shape[-1] // 2, 2)
    x0, x1 = x[..., 0], x[..., 1]
    return mx.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1).reshape(shape)


def layer_norm(x: mx.array, wb: tuple[mx.array, mx.array], eps: float = 1e-5) -> mx.array:
    return mx.fast.layer_norm(x, wb[0], wb[1], eps)


def linear(x: mx.array, wb: tuple[mx.array, mx.array]) -> mx.array:
    """torch-layout Linear: weight [out, in], bias [out]."""
    return mx.addmm(wb[1], x, wb[0].T)


class PixAIMLX:
    def __init__(self, model_dir: str | Path, dtype: mx.Dtype = mx.bfloat16):
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text())
        for key, value in {
            "use_rope": True,
            "use_interp_rope": True,
            "rope_pt_size": None,
            "use_abs_pos": True,
            "tile_abs_pos": True,
            "rel_pos_blocks": [],
            "ln_pre": True,
            "ln_post": False,
            "retain_cls_token": False,
            "pretrain_use_cls_token": True,
            "init_values": None,
            "bias_patch_embed": False,
            "act_layer": "GELU",
            "norm_layer": "LayerNorm",
        }.items():
            _expect(config, key, value)

        self.image_size = config["img_size"]
        self.patch = config["patch_size"]
        self.dim = config["embed_dim"]
        self.heads = config["num_heads"]
        self.depth = config["depth"]
        self.window = config["window_size"]
        self.global_blocks = frozenset(config["global_att_blocks"])
        grid = self.image_size // self.patch
        pre_grid = config["pretrain_img_size"] // self.patch
        if self.image_size % self.patch or grid % self.window or grid % pre_grid:
            raise ValueError("PixAI config needs grid divisible by window and pretrain grid")
        self.grid = grid
        n_win = grid // self.window
        head_dim = self.dim // self.heads

        # Window-order position k -> row-major grid index.
        wr, wc, r, c = np.meshgrid(*(np.arange(n) for n in (n_win, n_win, self.window, self.window)), indexing="ij")
        perm = ((wr * self.window + r) * grid + wc * self.window + c).reshape(-1)

        raw = mx.load(str(model_dir / "model.safetensors"))
        w: Weights = {}

        def cast(a: mx.array) -> mx.array:
            return a.astype(dtype)

        # Conv(k=s=patch, no bias) as a matmul over (kh, kw, c)-ordered patches.
        w["patch"] = cast(raw.pop("patch_embed.proj.weight").transpose(0, 2, 3, 1).reshape(self.dim, -1))
        # Tiled absolute pos embed: grid (row, col) uses pretrain (row % pre, col % pre).
        pos = np.asarray(raw.pop("pos_embed")[0, 1:]).reshape(pre_grid, pre_grid, self.dim)
        pos = np.tile(pos, (grid // pre_grid, grid // pre_grid, 1)).reshape(grid * grid, self.dim)[perm]
        w["pos"] = cast(mx.array(pos))
        w["ln_pre"] = (cast(raw.pop("ln_pre.weight")), cast(raw.pop("ln_pre.bias")))

        blocks = []
        for i in range(self.depth):
            p = f"blocks.{i}."
            blocks.append(
                {
                    "norm1": (cast(raw.pop(p + "norm1.weight")), cast(raw.pop(p + "norm1.bias"))),
                    "qkv": (cast(raw.pop(p + "attn.qkv.weight")), cast(raw.pop(p + "attn.qkv.bias"))),
                    "proj": (cast(raw.pop(p + "attn.proj.weight")), cast(raw.pop(p + "attn.proj.bias"))),
                    "norm2": (cast(raw.pop(p + "norm2.weight")), cast(raw.pop(p + "norm2.bias"))),
                    "fc1": (cast(raw.pop(p + "mlp.fc1.weight")), cast(raw.pop(p + "mlp.fc1.bias"))),
                    "fc2": (cast(raw.pop(p + "mlp.fc2.weight")), cast(raw.pop(p + "mlp.fc2.bias"))),
                }
            )
        w["blocks"] = blocks

        w["pool_norm"] = (cast(raw.pop("head_pool.norm.weight")), cast(raw.pop("head_pool.norm.bias")))
        w["pool_pos"] = cast(raw.pop("head_pool.pos_embed")[mx.array(perm)])
        w["pool_q"] = cast(raw.pop("head_pool.q").reshape(1, self.heads, 1, head_dim))
        w["pool_kv"] = (cast(raw.pop("head_pool.kv.weight")), cast(raw.pop("head_pool.kv.bias")))
        w["pool_proj"] = (cast(raw.pop("head_pool.proj.weight")), cast(raw.pop("head_pool.proj.bias")))
        w["head"] = (cast(raw.pop("head.weight")), cast(raw.pop("head.bias")))
        if raw:
            raise ValueError(f"unused PixAI weights: {sorted(raw)[:8]}")

        # Windowed blocks: per-window coords, rope_pt_size == window -> scale 1.
        # Global blocks: full-grid coords interpolated back to the window's range.
        cos_w, sin_w = _axial_rope(self.window, 1.0, head_dim)
        cos_g, sin_g = _axial_rope(grid, self.window / grid, head_dim)
        w["rope_win"] = (cast(mx.array(cos_w)), cast(mx.array(sin_w)))
        w["rope_glob"] = (cast(mx.array(cos_g[perm])), cast(mx.array(sin_g[perm])))

        mx.eval(w)
        self.weights = w
        self.dtype = dtype
        self._forward = mx.compile(partial(self._forward_impl, n_win=n_win, head_dim=head_dim))

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Fit inside size x size (aspect kept, antialiased bilinear), center on black, normalize to [-1, 1].

        Resizes in float32 ('F' mode, per channel) like the reference's torchvision resize on a float
        tensor; resizing uint8 instead costs a full quantization step and shifts probabilities ~0.02.
        Returns HWC float32.
        """
        image = rgb_on_white(image)
        size = self.image_size
        w, h = image.size
        r = min(size / h, size / w)
        new_h, new_w = int(h * r), int(w * r)
        # Normalizing before resizing gives the same result (resize is affine) and saves a pass.
        pixels = np.asarray(image, dtype=np.float32) * np.float32(2.0 / 255.0) - np.float32(1.0)
        if (new_w, new_h) != (w, h):
            pixels = np.stack(
                [
                    np.asarray(
                        Image.fromarray(np.ascontiguousarray(pixels[..., c]), mode="F").resize(
                            (new_w, new_h), Image.Resampling.BILINEAR
                        )
                    )
                    for c in range(3)
                ],
                axis=-1,
            )
        out = np.full((size, size, 3), -1.0, dtype=np.float32)
        top, left = (size - new_h) // 2, (size - new_w) // 2
        out[top : top + new_h, left : left + new_w] = pixels
        return out

    def _forward_impl(self, w: Weights, x: mx.array, *, n_win: int, head_dim: int) -> mx.array:
        B = x.shape[0]
        g, win, p, C, H = self.grid, self.window, self.patch, self.dim, self.heads
        L = g * g
        # [B, S, S, 3] -> window-ordered patches [B, L, p*p*3] (wr, wc, r, c | kh, kw, ch).
        x = x.astype(self.dtype).reshape(B, n_win, win, p, n_win, win, p, 3)
        x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, L, p * p * 3)
        x = x @ w["patch"].T + w["pos"]
        x = layer_norm(x, w["ln_pre"])

        for i, blk in enumerate(w["blocks"]):
            if i in self.global_blocks:
                b, n, (cos, sin) = B, L, w["rope_glob"]
            else:
                b, n, (cos, sin) = B * n_win * n_win, win * win, w["rope_win"]
            h = layer_norm(x, blk["norm1"]).reshape(b, n, C)
            qkv = linear(h, blk["qkv"]).reshape(b, n, 3, H, head_dim).transpose(2, 0, 3, 1, 4)
            q = _apply_rope(qkv[0], cos, sin)
            k = _apply_rope(qkv[1], cos, sin)
            o = mx.fast.scaled_dot_product_attention(q, k, qkv[2], scale=head_dim**-0.5)
            o = o.transpose(0, 2, 1, 3).reshape(B, L, C)
            x = x + linear(o, blk["proj"])
            h = linear(layer_norm(x, blk["norm2"]), blk["fc1"])
            x = x + linear(_gelu(h), blk["fc2"])

        h = layer_norm(x, w["pool_norm"]) + w["pool_pos"]
        k, v = mx.split(linear(h, w["pool_kv"]), 2, axis=-1)
        k = k.reshape(B, L, H, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, H, head_dim).transpose(0, 2, 1, 3)
        q = mx.broadcast_to(w["pool_q"], (B, H, 1, head_dim))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5).reshape(B, C)
        logits = linear(linear(o, w["pool_proj"]), w["head"])
        return mx.sigmoid(logits.astype(mx.float32))

    def __call__(self, batch: np.ndarray | mx.array) -> mx.array:
        """batch: [B, S, S, 3] preprocessed images -> lazy [B, num_tags] float32 probabilities."""
        return self._forward(self.weights, mx.array(batch))


def _gelu(x: mx.array) -> mx.array:
    # erf GELU, as torch.nn.GELU()
    return x * (1 + mx.erf(x * 0.7071067811865476)) * 0.5
