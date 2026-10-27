"""PixAI Tagger v1.0 in plain PyTorch, window-ordered like backends/mlx/model.py. The int8 engine reuses its stem,
head and RoPE tables."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

S, P, C, H, D, DEPTH, WIN = 1008, 14, 1024, 16, 64, 32, 24
G = S // P  # 72
NW = G // WIN  # 3
L = G * G  # 5184
LW = WIN * WIN  # 576
GLOBAL = frozenset((7, 15, 23, 31))
PRE = 24  # pretrain grid (336 / 14)


def window_perm() -> torch.Tensor:
    """Window-order position -> row-major grid index."""
    wr, wc, r, c = torch.meshgrid(*(torch.arange(n) for n in (NW, NW, WIN, WIN)), indexing="ij")
    return ((wr * WIN + r) * G + wc * WIN + c).reshape(-1)


def axial_rope(end: int, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [end*end, D/2] exactly as the reference computes freqs_cis (fp32)."""
    freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 4)[: D // 4].float() / D))
    t = torch.arange(end * end, dtype=torch.float32)
    tx = (t % end).float() * scale
    ty = torch.div(t, end, rounding_mode="floor").float() * scale
    ang = torch.cat([torch.outer(tx, freqs), torch.outer(ty, freqs)], dim=-1)
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair rotation in fp32 (as the reference), cast back. x [..., L, D]; cos/sin [L, D/2]."""
    xf = x.float().unflatten(-1, (D // 2, 2))
    x0, x1 = xf[..., 0], xf[..., 1]
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2).to(x.dtype)


class PixAITorch(torch.nn.Module):
    def __init__(
        self,
        model_dir: str | Path,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        *,
        raw: dict[str, torch.Tensor] | None = None,
        with_blocks: bool = True,
    ):
        """raw: preloaded state dict (default: model_dir/model.safetensors). with_blocks=False keeps only the
        stem, head and layout tables (the CUDA engine brings its own int8 blocks)."""
        super().__init__()
        model_dir = Path(model_dir)
        self.dtype = dtype
        if raw is None:
            raw = load_file(str(model_dir / "model.safetensors"), device=device)
        perm = window_perm().to(device)

        def cast(t: torch.Tensor) -> torch.Tensor:
            return t.to(dtype).contiguous()

        self.patch_w = cast(raw["patch_embed.proj.weight"].permute(0, 2, 3, 1).reshape(C, -1))
        pos = raw["pos_embed"][0, 1:].reshape(PRE, PRE, C).repeat(G // PRE, G // PRE, 1).reshape(L, C)
        self.pos = cast(pos[perm])
        self.ln_pre = (cast(raw["ln_pre.weight"]), cast(raw["ln_pre.bias"]))
        self.blocks = []
        for i in range(DEPTH if with_blocks else 0):
            p = f"blocks.{i}."
            self.blocks.append(
                {
                    k: cast(raw[p + n])
                    for k, n in {
                        "n1w": "norm1.weight",
                        "n1b": "norm1.bias",
                        "qkv_w": "attn.qkv.weight",
                        "qkv_b": "attn.qkv.bias",
                        "proj_w": "attn.proj.weight",
                        "proj_b": "attn.proj.bias",
                        "n2w": "norm2.weight",
                        "n2b": "norm2.bias",
                        "fc1_w": "mlp.fc1.weight",
                        "fc1_b": "mlp.fc1.bias",
                        "fc2_w": "mlp.fc2.weight",
                        "fc2_b": "mlp.fc2.bias",
                    }.items()
                }
            )
        self.pool_norm = (cast(raw["head_pool.norm.weight"]), cast(raw["head_pool.norm.bias"]))
        self.pool_pos = cast(raw["head_pool.pos_embed"][perm])
        self.pool_q = cast(raw["head_pool.q"])
        self.pool_kv = (cast(raw["head_pool.kv.weight"]), cast(raw["head_pool.kv.bias"]))
        self.pool_proj = (cast(raw["head_pool.proj.weight"]), cast(raw["head_pool.proj.bias"]))
        self.head = (cast(raw["head.weight"]), cast(raw["head.bias"]))
        cw, sw = axial_rope(WIN, 1.0)
        cg, sg = axial_rope(G, WIN / G)
        self.rope_win = (cw.to(device), sw.to(device))
        self.rope_glob = (cg.to(device)[perm], sg.to(device)[perm])

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, S, 3] -> window-ordered patches [B*L, P*P*3] in (kh, kw, c) order."""
        B = x.shape[0]
        x = x.to(self.dtype).view(B, NW, WIN, P, NW, WIN, P, 3).permute(0, 1, 4, 2, 5, 3, 6, 7)
        return x.reshape(B * L, P * P * 3)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, S, 3] normalized pixels -> logits [B, num_classes] fp32."""
        B = x.shape[0]
        T = B * L
        x = self.patchify(x) @ self.patch_w.t()
        x = (x.view(B, L, C) + self.pos).view(T, C)
        x = F.layer_norm(x, (C,), *self.ln_pre, 1e-5)
        for i, blk in enumerate(self.blocks):
            h = F.layer_norm(x, (C,), blk["n1w"], blk["n1b"], 1e-5)
            qkv = F.linear(h, blk["qkv_w"], blk["qkv_b"])
            if i in GLOBAL:
                nb, n, (cos, sin) = B, L, self.rope_glob
            else:
                nb, n, (cos, sin) = B * NW * NW, LW, self.rope_win
            q, k, v = qkv.view(nb, n, 3, H, D).permute(2, 0, 3, 1, 4)
            o = F.scaled_dot_product_attention(apply_rope(q, cos, sin), apply_rope(k, cos, sin), v)
            o = o.transpose(1, 2).reshape(T, C)
            x = x + F.linear(o, blk["proj_w"], blk["proj_b"])
            h = F.layer_norm(x, (C,), blk["n2w"], blk["n2b"], 1e-5)
            h = F.gelu(F.linear(h, blk["fc1_w"], blk["fc1_b"]))
            x = x + F.linear(h, blk["fc2_w"], blk["fc2_b"])
        return self.pool_head(x.view(B, L, C))

    def pool_head(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = F.layer_norm(x, (C,), *self.pool_norm, 1e-5) + self.pool_pos
        k, v = F.linear(h, *self.pool_kv).view(B, L, 2, H, D).permute(2, 0, 3, 1, 4)
        q = self.pool_q.view(1, H, 1, D).expand(B, H, 1, D)
        o = F.scaled_dot_product_attention(q, k, v).reshape(B, C)
        o = F.linear(o, *self.pool_proj)
        return F.linear(o, *self.head).float()
