"""PixAI Tagger v1.0 with int8 GEMMs and custom kernels, one CUDA graph per batch size.

Per transformer block:
  LN1 -> H1024 rotation -> int8 (per-token scale)  -> QKV GEMM (int32 acc) + bias + RoPE -> fp16
  flash attention (fp16 tensor cores)              -> fp16
  H1024 rotation -> int8 (per-token)               -> proj GEMM + bias + residual (fp16 stream)
  LN2 -> H1024 rotation -> int8 (per-token)        -> fc1 GEMM + bias + GELU -> fp16
  block-H128 rotation -> int8 (per-token)          -> fc2 GEMM + bias + residual
Weights carry the same rotations (W' = W H), quantized per output channel. Stem and head run in fp16.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .kernels import kernels
from .reference import DEPTH, GLOBAL, LW, C, L, PixAITorch

EPI_QKV_ROPE, EPI_RESIDUAL, EPI_GELU_F16 = 0, 1, 2  # must match csrc/gemm.cuh
GEMM_STAGES = (2, 3)  # pipeline depths autotuned per GEMM
Q_SCALE = 0.125 * math.log2(math.e)  # folded into q by the QKV epilogue


def hadamard(n: int, device: str) -> torch.Tensor:
    """Normalized Sylvester Hadamard matrix (symmetric, orthogonal); matches the in-kernel FWHT order."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(device=device, dtype=torch.float32)


def quantize_rows_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8: w ~= q * scale[:, None]."""
    scale = w.abs().amax(1).clamp_min(1e-12) / 127.0
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.float().contiguous()


# Quantized linears -> Hadamard block size of their input rotation.
QUANTIZED = {"attn.qkv": 1024, "attn.proj": 1024, "mlp.fc1": 1024, "mlp.fc2": 128}
FORMAT = "foidfinder-pixai-int8-v1"


def convert_checkpoint(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Official fp32 state dict -> engine format: each block linear becomes `<name>.qweight` (int8, rotated by W H),
    `<name>.wscale` (fp32 per output channel) and `<name>.bias`; everything else is copied unchanged."""
    out = {k: v for k, v in raw.items() if not any(f".{n}." in k for n in QUANTIZED)}
    rot = {}
    for i in range(DEPTH):
        for name, n in QUANTIZED.items():
            p = f"blocks.{i}.{name}"
            w = raw[p + ".weight"].float()
            if w.shape[1] not in rot:
                rot[w.shape[1]] = torch.block_diag(*[hadamard(n, str(w.device))] * (w.shape[1] // n))
            out[p + ".qweight"], out[p + ".wscale"] = quantize_rows_int8(w @ rot[w.shape[1]])
            out[p + ".bias"] = raw[p + ".bias"].float().contiguous()
    return out


class PixAICuda:
    def __init__(self, model_dir: str | Path, *, device: str = "cuda"):
        """model_dir: the official checkpoint (converted in memory) or the output of `foidfinder convert`."""
        self.k = kernels()
        self.device = device
        raw = load_file(str(Path(model_dir) / "model.safetensors"), device=device)
        if "blocks.0.attn.qkv.qweight" not in raw:
            raw = convert_checkpoint(raw)
        # Supplies the fp16 stem/head and RoPE tables; blocks come from self.layers.
        self.ref = PixAITorch(model_dir, dtype=torch.float16, device=device, raw=raw, with_blocks=False)

        def quantized(p: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return raw[p + ".qweight"], raw[p + ".wscale"], raw[p + ".bias"]

        def norm(p: str) -> tuple[torch.Tensor, torch.Tensor]:
            return raw[p + ".weight"].float().contiguous(), raw[p + ".bias"].float().contiguous()

        self.layers = [
            {
                "n1": norm(f"blocks.{i}.norm1"),
                "n2": norm(f"blocks.{i}.norm2"),
                **{name.split(".")[1]: quantized(f"blocks.{i}.{name}") for name in QUANTIZED},
            }
            for i in range(DEPTH)
        ]
        torch.cuda.empty_cache()
        self.rope_win = tuple(t.contiguous() for t in self.ref.rope_win)
        self.rope_glob = tuple(t.contiguous() for t in self.ref.rope_glob)
        self.stages: dict[tuple[str, int], int] = {}
        self._graphs: dict[int, tuple] = {}  # batch -> (graph, static input, static output, buffers)

    def _buffers(self, B: int) -> dict[str, torch.Tensor]:
        T, dev = B * L, self.device
        return {
            "x": torch.empty(T, C, device=dev, dtype=torch.half),
            "a8": torch.empty(T, C, device=dev, dtype=torch.int8),
            "rs": torch.empty(T, device=dev),
            "qkv": torch.empty(T, 3 * C, device=dev, dtype=torch.half),
            "o16": torch.empty(T, C, device=dev, dtype=torch.half),
            "h16": torch.empty(T, 4736, device=dev, dtype=torch.half),
            "h8": torch.empty(T, 4736, device=dev, dtype=torch.int8),
        }

    def _autotune(self, B: int, buf: dict[str, torch.Tensor]) -> None:
        """Pick the faster pipeline depth per GEMM for this batch size."""
        lw = self.layers[0]
        jobs = {
            "qkv": lambda st: self._gemm(buf["a8"], lw["qkv"], buf["rs"], EPI_QKV_ROPE, st, buf["qkv"], self.rope_win),
            "proj": lambda st: self._gemm(buf["a8"], lw["proj"], buf["rs"], EPI_RESIDUAL, st, buf["o16"]),
            "fc1": lambda st: self._gemm(buf["a8"], lw["fc1"], buf["rs"], EPI_GELU_F16, st, buf["h16"]),
            "fc2": lambda st: self._gemm(buf["h8"], lw["fc2"], buf["rs"], EPI_RESIDUAL, st, buf["o16"]),
        }
        buf["a8"].zero_()
        buf["h8"].zero_()
        buf["rs"].fill_(1.0)
        buf["o16"].zero_()
        for name, fn in jobs.items():
            best = None
            for st in GEMM_STAGES:
                for _ in range(3):
                    fn(st)
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                for _ in range(10):
                    fn(st)
                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end)
                if best is None or ms < best[0]:
                    best = (ms, st)
            self.stages[(name, B)] = best[1]

    def _gemm(self, a, lw, rs, epi, stages, out, rope=None):
        w, ws, b = lw
        cos, sin = rope if rope is not None else (None, None)
        self.k.gemm_s8(a, w, rs, ws, b, out, epi, stages, cos, sin, Q_SCALE)

    def _forward(self, img: torch.Tensor, buf: dict[str, torch.Tensor]) -> torch.Tensor:
        k, ref = self.k, self.ref
        B = img.shape[0]
        x, a8, rs = buf["x"], buf["a8"], buf["rs"]
        torch.matmul(ref.patchify(img), ref.patch_w.t(), out=x)
        x.view(B, L, C).add_(ref.pos)
        x.copy_(F.layer_norm(x, (C,), *ref.ln_pre, 1e-5))
        st = {name: self.stages[(name, B)] for name in ("qkv", "proj", "fc1", "fc2")}
        for i, lw in enumerate(self.layers):
            seq, rope = (L, self.rope_glob) if i in GLOBAL else (LW, self.rope_win)
            k.rot_q8_1024(x, *lw["n1"], a8, rs)
            self._gemm(a8, lw["qkv"], rs, EPI_QKV_ROPE, st["qkv"], buf["qkv"], rope)
            k.attention(buf["qkv"], buf["o16"], seq)
            k.rot_q8_1024(buf["o16"], None, None, a8, rs)
            self._gemm(a8, lw["proj"], rs, EPI_RESIDUAL, st["proj"], x)
            k.rot_q8_1024(x, *lw["n2"], a8, rs)
            self._gemm(a8, lw["fc1"], rs, EPI_GELU_F16, st["fc1"], buf["h16"])
            k.rot_q8_h128(buf["h16"], buf["h8"], rs)
            self._gemm(buf["h8"], lw["fc2"], rs, EPI_RESIDUAL, st["fc2"], x)
        return ref.pool_head(x.view(B, L, C))

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 1008, 1008, 3] normalized pixels (any float dtype) -> logits [B, 30877] fp32."""
        B = images.shape[0]
        if B not in self._graphs:
            buf = self._buffers(B)
            self._autotune(B, buf)
            static_in = torch.zeros(B, *images.shape[1:], device=self.device, dtype=torch.half)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):  # allocate outside graph capture
                    self._forward(static_in, buf)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self._forward(static_in, buf)
            self._graphs[B] = (graph, static_in, static_out, buf)
        graph, static_in, static_out, _ = self._graphs[B]
        static_in.copy_(images)
        graph.replay()
        return static_out.clone()
