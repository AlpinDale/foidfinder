"""Haruka Tagger conversion: the published `haruka-tagger.safetensors` + `haruka-tagger_metadata.json` -> the folded
fp16 safetensors + config.json that backends/mlx/haruka.py loads. Runs once; needs only numpy and safetensors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
MODEL = {
    "name": "Haruka Tagger",
    "architecture": "haruka-eva02-tree",
    "img_size": 448,
    "patch_size": 14,
    "dim": 768,
    "heads": 12,
    "depth": 12,
    "mlp_hidden": 2048,
    "pool_heads": 16,
    "coarse_queries": 6,
    "leaf_queries": 18,
    "leaf_dim": 64,
    "rope_ref_grid": 16,
}
# The model card's recommendations; rating is reported as the argmax.
THRESHOLDS = {"general": 0.35, "character": 0.35, "copyright": 0.35, "artist": 0.25, "year": 0.25, "rating": 0.5}
WEIGHTS, METADATA = "haruka-tagger.safetensors", "haruka-tagger_metadata.json"


def fold(sd: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """timm EVA02 + tree MHAttnPool state dict (fp32 numpy) -> the tensors HarukaMLX reads."""
    D, H = MODEL["dim"], MODEL["heads"]
    p = MODEL["patch_size"]
    bb = "backbone."
    out: dict[str, np.ndarray] = {}

    # Patch embed over (kh, kw, c)-ordered raw 0..255 pixels with ImageNet normalization folded in.
    w = sd[bb + "patch_embed.proj.weight"].transpose(0, 2, 3, 1)  # [D, kh, kw, c]
    b = sd[bb + "patch_embed.proj.bias"] - (w * (IMAGENET_MEAN / IMAGENET_STD)).sum(axis=(1, 2, 3))
    out["patch_w"] = (w / (255.0 * IMAGENET_STD)).reshape(D, p * p * 3)
    pos = sd[bb + "pos_embed"][0]
    out["cls"] = sd[bb + "cls_token"].reshape(1, D) + pos[:1]
    out["patch_b"] = b + pos[1:]

    scale = (D // H) ** -0.5
    for i in range(MODEL["depth"]):
        src, dst = f"{bb}blocks.{i}.", f"blocks.{i}."
        a = src + "attn."
        out[dst + "qkv.weight"] = np.concatenate(
            [sd[a + "q_proj.weight"] * scale, sd[a + "k_proj.weight"], sd[a + "v_proj.weight"]]
        )
        out[dst + "qkv.bias"] = np.concatenate(
            [sd[a + "q_proj.bias"] * scale, np.zeros(D, np.float32), sd[a + "v_proj.bias"]]
        )
        m = src + "mlp."
        out[dst + "fc1.weight"] = np.concatenate([sd[m + "fc1_g.weight"], sd[m + "fc1_x.weight"]])
        out[dst + "fc1.bias"] = np.concatenate([sd[m + "fc1_g.bias"], sd[m + "fc1_x.bias"]])
        for name, key in (
            ("norm1", src + "norm1"),
            ("norm2", src + "norm2"),
            ("proj", a + "proj"),
            ("mlp_norm", m + "norm"),
            ("fc2", m + "fc2"),
        ):
            out[f"{dst}{name}.weight"] = sd[key + ".weight"]
            out[f"{dst}{name}.bias"] = sd[key + ".bias"]
    out["norm.weight"], out["norm.bias"] = sd[bb + "norm.weight"], sd[bb + "norm.bias"]

    # Tree attention pool. The 1/sqrt(head_dim) logit scale goes into the key projection.
    Hp = MODEL["pool_heads"]
    pool_scale = (D // Hp) ** -0.5
    hp = "head_pool."
    wk, wv = np.split(sd[hp + "kv.weight"], 2)
    bk, bv = np.split(sd[hp + "kv.bias"], 2)
    out["pool.kv.weight"] = np.concatenate([wk * pool_scale, wv])
    out["pool.kv.bias"] = np.concatenate([bk * pool_scale, bv])
    for name in ("norm", "trunk_norm", "child_q_norm", "leaf_norm"):
        out[f"pool.{name}.weight"], out[f"pool.{name}.bias"] = sd[hp + name + ".weight"], sd[hp + name + ".bias"]
    out["pool.q_coarse"] = sd[hp + "q_coarse"].reshape(-1, D)
    out["pool.q_coarse_bias"] = sd[hp + "q_coarse_bias"].reshape(-1)
    out["pool.q_child"] = sd[hp + "q_child"].reshape(-1, D)
    out["pool.q_child_bias"] = sd[hp + "q_child_bias"].reshape(-1)
    out["pool.leaf.weight"] = sd[hp + "leaf_proj_weight"]  # [leaves, D, leaf_dim]
    out["pool.leaf.bias"] = sd[hp + "leaf_proj_bias"]
    out["head.weight"], out["head.bias"] = sd["head.weight"], sd["head.bias"]
    return out


def vocab(path: Path) -> tuple[list[str], list[str]]:
    meta = json.loads(path.read_text())
    tags = [meta["idx_to_tag"][str(i)] for i in range(int(meta["total_tags"]))]
    return tags, [meta["tag_to_category"].get(t, "general") for t in tags]


def convert(src: Path, out_dir: Path, dtype: type = np.float16) -> None:
    """src: a download of KaraKaraWitch/haruka-tagger (weights + metadata)."""
    from safetensors.numpy import load_file, save_file

    tags, categories = vocab(src / METADATA)
    sd = load_file(str(src / WEIGHTS))
    if sd["head.weight"].shape[0] != len(tags):
        raise ValueError(f"weights have {sd['head.weight'].shape[0]} tags, metadata {len(tags)}")
    tensors = {k: np.ascontiguousarray(v, dtype=dtype) for k, v in fold(sd).items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "model.safetensors"))
    config = {"model": MODEL, "tags": tags, "categories": categories, "thresholds": THRESHOLDS}
    (out_dir / "config.json").write_text(json.dumps(config))
