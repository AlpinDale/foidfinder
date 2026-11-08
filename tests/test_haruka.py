"""Haruka Tagger on MLX: GPU preprocessing vs Pillow, and the converted model vs the model card's PyTorch code.

The parity tests need FOIDFINDER_HARUKA_SOURCE (a local copy of KaraKaraWitch/haruka-tagger) and torch + timm
(`uv sync --extra mlx --group parity`).
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

mx = pytest.importorskip("mlx.core")

from foidfinder.backends.mlx.kernels import resize_pad


def _card_pixels(image: Image.Image, s: int = 448) -> np.ndarray:
    """The model card's preprocessing (composite onto white, letterboxed bicubic resize), uint8 HWC."""
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        image = bg.convert("RGB")
    w, h = image.size
    if (w, h) != (s, s):
        r = min(s / w, s / h)
        nw, nh = max(1, int(w * r)), max(1, int(h * r))
        canvas = Image.new("RGB", (s, s), (255, 255, 255))
        canvas.paste(image.resize((nw, nh), Image.Resampling.BICUBIC), ((s - nw) // 2, (s - nh) // 2))
        image = canvas
    return np.asarray(image)


@pytest.mark.parametrize(
    ("w", "h"),
    [
        (850, 1200),  # horizontal pass first
        (448, 900),  # height shrinks over twice as much: Pillow runs the vertical pass first
        (3, 5000),
        (5000, 3),
        (100, 37),  # upscale
        (448, 300),  # width already at the target
        (1, 1),
    ],
)
def test_gpu_resize_is_bit_exact_with_pillow(w, h):
    rgb = np.random.default_rng(w * h).integers(0, 256, (h, w, 3), dtype=np.uint8)
    s = 448
    r = min(s / w, s / h)
    nw, nh = max(1, int(w * r)), max(1, int(h * r))
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BICUBIC), ((s - nw) // 2, (s - nh) // 2))
    np.testing.assert_array_equal(np.array(resize_pad(rgb, nw, nh, s)), np.asarray(canvas))


SOURCE = os.environ.get("FOIDFINDER_HARUKA_SOURCE")


@pytest.fixture(scope="module")
def haruka(tmp_path_factory):
    if not (SOURCE and (Path(SOURCE) / "haruka-tagger.safetensors").is_file()):
        pytest.skip("set FOIDFINDER_HARUKA_SOURCE to a local copy of KaraKaraWitch/haruka-tagger")
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    from safetensors.torch import load_file
    from torch import nn

    from foidfinder.backends.mlx.haruka import HarukaMLX
    from foidfinder.haruka import convert

    class MHAttnPool(nn.Module):  # from the model card
        def __init__(self, dim=768, num_heads=16, num_queries=18, leaf_dim=64, pool_mode="slot_sigmoid"):
            super().__init__()
            self.num_heads, self.head_dim, self.num_queries, self.pool_mode = (
                num_heads,
                dim // num_heads,
                num_queries,
                pool_mode,
            )
            self.norm = nn.LayerNorm(dim)
            self.kv = nn.Linear(dim, dim * 2)
            self.num_coarse = min(6, num_queries)
            self.branches = num_queries // self.num_coarse
            self.out_dim = num_queries * leaf_dim
            self.q_coarse = nn.Parameter(torch.zeros(1, self.num_coarse, dim))
            self.trunk_norm = nn.LayerNorm(dim)
            self.q_child = nn.Parameter(torch.zeros(1, self.num_coarse, self.branches, dim))
            self.child_q_norm = nn.LayerNorm(dim)
            self.leaf_norm = nn.LayerNorm(dim)
            self.q_coarse_bias = nn.Parameter(torch.full((1, 1, self.num_coarse, 1), -2.0))
            self.q_child_bias = nn.Parameter(torch.full((1, 1, num_queries, 1), -2.0))
            self.leaf_proj_weight = nn.Parameter(torch.empty(num_queries, dim, leaf_dim))
            self.leaf_proj_bias = nn.Parameter(torch.zeros(num_queries, leaf_dim))

        def forward(self, x):
            B, N, D = x.shape
            x = self.norm(x)
            k, v = self.kv(x).chunk(2, dim=-1)
            k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            scale = self.head_dim**-0.5
            q_c = (
                self.q_coarse.expand(B, self.num_coarse, D)
                .view(B, self.num_coarse, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            logits_c = (q_c @ k.transpose(-2, -1)) * scale
            attn_c = torch.softmax(logits_c, dim=-2) * torch.sigmoid(logits_c + self.q_coarse_bias)
            z_coarse = self.trunk_norm((attn_c @ v).transpose(1, 2).contiguous().view(B, self.num_coarse, D))
            q_leaf = self.child_q_norm((z_coarse.unsqueeze(2) + self.q_child).view(B, self.num_queries, D))
            q_l = q_leaf.view(B, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
            logits_l = (q_l @ k.transpose(-2, -1)) * scale
            attn_l = torch.softmax(logits_l, dim=-2) * torch.sigmoid(logits_l + self.q_child_bias)
            z_child = (attn_l @ v).transpose(1, 2).contiguous().view(B, self.num_coarse, self.branches, D)
            z_tree = self.leaf_norm((z_coarse.unsqueeze(2) + z_child).view(B, self.num_queries, D))
            mass_q = attn_l.sum(dim=-1).mean(dim=1).unsqueeze(-1)
            z_tree = z_tree * (mass_q / (mass_q + 4.0))
            leaf_vecs = torch.einsum("bqd,qdf->bqf", z_tree, self.leaf_proj_weight) + self.leaf_proj_bias
            return leaf_vecs.reshape(B, self.out_dim)

    class Reference(nn.Module):
        def __init__(self, tags):
            super().__init__()
            self.backbone = timm.create_model(
                "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", pretrained=False, num_classes=0, global_pool=""
            )
            self.head_pool = MHAttnPool()
            self.head = nn.Linear(self.head_pool.out_dim, tags)

        def forward(self, x):
            return self.head(self.head_pool(self.backbone.forward_features(x)))

    sd = load_file(str(Path(SOURCE) / "haruka-tagger.safetensors"))
    ref = Reference(sd["head.weight"].shape[0])
    ref.load_state_dict(sd, strict=True)
    out = tmp_path_factory.mktemp("haruka")
    convert(Path(SOURCE), out)
    return ref.eval(), out, HarukaMLX


def test_preprocessing_matches_model_card(haruka, sample_images):
    _, out, HarukaMLX = haruka
    model = HarukaMLX(out)
    for image in sample_images:
        np.testing.assert_array_equal(np.array(model.preprocess(image)), _card_pixels(image))


def test_fp16_model_tracks_fp32_reference(haruka, sample_images):
    import torch

    ref_model, out, HarukaMLX = haruka
    model = HarukaMLX(out)
    tol = 0.01  # measured 0.008 on 400 real images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    for image in sample_images:
        pixels = _card_pixels(image)
        with torch.no_grad():
            x = (torch.from_numpy(pixels).permute(2, 0, 1)[None].float() / 255 - mean) / std
            ref = torch.sigmoid(ref_model(x))[0].numpy()
        got = np.array(model(mx.array(pixels[None])))[0]
        assert np.abs(got - ref).max() < tol
        assert np.all(got[ref > 0.35 + tol] >= 0.35) and np.all(got[ref < 0.35 - tol] < 0.35)


def test_compressed_codes_round_trip():
    from foidfinder.haruka_quant import decode, encode

    # Signs, zero, and codes needing a second byte plane (|code| > 127 after zigzag); one row is all zeros.
    codes = np.random.default_rng(0).integers(-300, 301, (5, 128)).astype(np.int32)
    codes[0, :4] = [0, -1, 1, -128]
    codes[1] = 0
    planes = encode(codes)
    assert len(planes) == 2
    assert np.array_equal(decode(planes, codes.shape), codes)
