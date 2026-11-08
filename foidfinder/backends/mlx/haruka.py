"""Haruka Tagger (EVA02-B/14 at 448 px + 6x3-query slot-sigmoid tree pool) in MLX.

Loads the folded checkpoint written by `foidfinder.haruka.convert`: input normalization lives in the patch
embedding, attention scales in the query/key weights, and QKV and SwiGLU projections are single matrices.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from ...image import rgb_on_white
from .kernels import resize_pad, rope_qk, swiglu_layer_norm

EPS = 1e-6


def rope_tables(grid: int, ref: int, head_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """timm RotaryEmbeddingCat (EVA02): axial, interleaved pairs, positions rescaled to the pretrain grid.

    Returns cos/sin [1 + grid*grid, head_dim]; row 0 (CLS) is the identity rotation."""
    bands = 1.0 / (10000.0 ** (np.arange(head_dim // 4, dtype=np.float32) / (head_dim // 4)))
    t = np.arange(grid, dtype=np.float32) / grid * ref
    ty, tx = np.meshgrid(t, t, indexing="ij")
    ang = np.concatenate([ty.reshape(-1, 1) * bands, tx.reshape(-1, 1) * bands], axis=1)  # [N, head_dim/2]
    ang = np.repeat(ang, 2, axis=1)
    cos = np.concatenate([np.ones((1, head_dim), np.float32), np.cos(ang)])
    sin = np.concatenate([np.zeros((1, head_dim), np.float32), np.sin(ang)])
    return cos, sin


def _layer_norm(x: mx.array, w: mx.array, b: mx.array) -> mx.array:
    return mx.fast.layer_norm(x, w, b, EPS)


class HarukaMLX:
    def __init__(self, model_dir: str | Path, dtype: mx.Dtype = mx.float16):
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text())
        cfg = config["model"]
        self.size, self.patch = cfg["img_size"], cfg["patch_size"]
        self.dim, self.heads, self.depth = cfg["dim"], cfg["heads"], cfg["depth"]
        self.pool_heads, self.coarse, self.leaves = cfg["pool_heads"], cfg["coarse_queries"], cfg["leaf_queries"]
        self.grid = self.size // self.patch
        self.dtype = dtype
        if "quantization" in config:  # compressed for download (foidfinder.haruka_quant)
            from ...haruka_quant import load

            raw = {k: mx.array(v) for k, v in load(model_dir).items()}
        else:
            raw = mx.load(str(model_dir / "model.safetensors"))
        w = {k: v.astype(dtype) for k, v in raw.items()}
        cos, sin = rope_tables(self.grid, cfg["rope_ref_grid"], self.dim // self.heads)
        w["rope_cs"] = mx.array(np.stack([cos[:, 0::2], sin[:, 0::2]], axis=-1))  # fp32 [tokens, head_dim/2, 2]
        mx.eval(w)
        self.w = w
        self._forward = mx.compile(self._forward_impl)

    def preprocess(self, image: Image.Image) -> mx.array:
        """The model card's preprocessing (composite onto white, letterboxed bicubic resize) with the resize on the
        GPU, bit-exact: uint8 HWC. Normalization is folded into the patch embedding."""
        rgb = np.asarray(rgb_on_white(image), dtype=np.uint8)
        h, w = rgb.shape[:2]
        s = self.size
        if (w, h) == (s, s):
            return mx.array(rgb)
        r = min(s / w, s / h)
        return resize_pad(rgb, max(1, int(w * r)), max(1, int(h * r)), s)

    def _forward_impl(self, w: dict, x: mx.array) -> mx.array:
        B = x.shape[0]
        g, p, D, H = self.grid, self.patch, self.dim, self.heads
        hd = D // H
        x = x.astype(self.dtype).reshape(B, g, p, g, p, 3).transpose(0, 1, 3, 2, 4, 5).reshape(B, g * g, p * p * 3)
        x = mx.concatenate([mx.broadcast_to(w["cls"], (B, 1, D)), x @ w["patch_w"].T + w["patch_b"]], axis=1)
        N = x.shape[1]
        x = x.reshape(B * N, D)

        def heads(t: mx.array) -> mx.array:  # [B*N, D] rows -> [B, H, N, hd] view
            return t.reshape(B, N, H, hd).transpose(0, 2, 1, 3)

        for i in range(self.depth):
            pre = f"blocks.{i}."
            h = _layer_norm(x, w[pre + "norm1.weight"], w[pre + "norm1.bias"])
            qkv = mx.addmm(w[pre + "qkv.bias"], h, w[pre + "qkv.weight"].T)
            qk = rope_qk(qkv, w["rope_cs"], N, D, hd)
            o = mx.fast.scaled_dot_product_attention(
                heads(qk[:, :D]), heads(qk[:, D:]), heads(qkv[:, 2 * D :]), scale=1.0
            )
            o = o.transpose(0, 2, 1, 3).reshape(B * N, D)
            x = x + mx.addmm(w[pre + "proj.bias"], o, w[pre + "proj.weight"].T)
            h = _layer_norm(x, w[pre + "norm2.weight"], w[pre + "norm2.bias"])
            gx = mx.addmm(w[pre + "fc1.bias"], h, w[pre + "fc1.weight"].T)
            h = swiglu_layer_norm(gx, w[pre + "mlp_norm.weight"], w[pre + "mlp_norm.bias"])
            x = x + mx.addmm(w[pre + "fc2.bias"], h, w[pre + "fc2.weight"].T)

        x = _layer_norm(x, w["norm.weight"], w["norm.bias"])
        return mx.sigmoid(self._pool_head(w, x.reshape(B, N, D)).astype(mx.float32))

    def _pool_head(self, w: dict, x: mx.array) -> mx.array:
        """Two-level query tree over the backbone tokens [B, N, D] -> logits [B, num_tags]."""
        B, _, D = x.shape
        Hp, C, Q = self.pool_heads, self.coarse, self.leaves
        hd = D // Hp

        def heads(t: mx.array) -> mx.array:  # [B, M, D] -> [B, Hp, M, hd]
            return t.reshape(B, t.shape[1], Hp, hd).transpose(0, 2, 1, 3)

        def merge(t: mx.array) -> mx.array:  # [B, Hp, M, hd] -> [B, M, D]
            return t.transpose(0, 2, 1, 3).reshape(B, t.shape[2], D)

        def slot_attend(q: mx.array, bias: mx.array) -> tuple[mx.array, mx.array]:
            """Queries [B, M, D] compete per token (softmax over queries); each token is also gated by sigmoid(logit + bias)."""
            logits = (heads(q) @ kt).astype(mx.float32)  # [B, Hp, M, N]; the 1/sqrt(hd) scale is in the keys
            attn = mx.softmax(logits, axis=2) * mx.sigmoid(logits + bias[:, None])
            return attn, merge(attn.astype(x.dtype) @ v)

        xn = _layer_norm(x, w["pool.norm.weight"], w["pool.norm.bias"])
        kv = mx.addmm(w["pool.kv.bias"], xn, w["pool.kv.weight"].T)
        k, v = heads(kv[..., :D]), heads(kv[..., D:])
        kt = k.swapaxes(-1, -2)

        _, z = slot_attend(mx.broadcast_to(w["pool.q_coarse"], (B, C, D)), w["pool.q_coarse_bias"])
        z_coarse = _layer_norm(z, w["pool.trunk_norm.weight"], w["pool.trunk_norm.bias"])
        z_parent = mx.repeat(z_coarse, Q // C, axis=1)  # each trunk feeds its Q/C child leaves
        q_leaf = _layer_norm(z_parent + w["pool.q_child"], w["pool.child_q_norm.weight"], w["pool.child_q_norm.bias"])
        attn, z_child = slot_attend(q_leaf, w["pool.q_child_bias"])
        z_tree = _layer_norm(z_parent + z_child, w["pool.leaf_norm.weight"], w["pool.leaf_norm.bias"])
        mass = attn.sum(axis=-1).mean(axis=1)[..., None]  # [B, Q, 1], patch mass per leaf
        z_tree = z_tree * (mass / (mass + 4.0)).astype(x.dtype)
        leaf = (z_tree[:, :, None, :] @ w["pool.leaf.weight"][None]).reshape(B, Q, -1) + w["pool.leaf.bias"]
        return mx.addmm(w["head.bias"], leaf.reshape(B, -1), w["head.weight"].T)

    def __call__(self, batch: np.ndarray | mx.array) -> mx.array:
        """batch: [B, S, S, 3] uint8 -> lazy [B, num_tags] float32 probabilities."""
        return self._forward(self.w, batch if isinstance(batch, mx.array) else mx.array(batch))
