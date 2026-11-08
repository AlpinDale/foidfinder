"""Fused Metal kernels for the Haruka MLX forward (memory-bound steps between the GEMMs)."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

# One thread per interleaved (even, odd) pair of q and of k. Rows are tokens of the [rows, 3*dim] QKV output;
# cs holds per-token (cos, sin) for each of the head_dim/2 pairs (row 0, CLS, is the identity).
_ROPE = mx.fast.metal_kernel(
    name="haruka_rope",
    input_names=["qkv", "cs"],
    output_names=["qk"],
    source="""
        const uint pair = thread_position_in_grid.x;  // 0 .. dim/2
        const uint row = thread_position_in_grid.y;   // token in batch*tokens
        const uint token = row % TOKENS;
        const uint band = pair % (HEAD_DIM / 2);
        const float2 r = float2(cs[(token * (HEAD_DIM / 2) + band) * 2], cs[(token * (HEAD_DIM / 2) + band) * 2 + 1]);
        const device vec<T, 2>* src = reinterpret_cast<const device vec<T, 2>*>(qkv + row * 3 * DIM);
        device vec<T, 2>* dst = reinterpret_cast<device vec<T, 2>*>(qk + row * 2 * DIM);
        const float2 q = float2(src[pair]);
        const float2 k = float2(src[DIM / 2 + pair]);
        dst[pair] = vec<T, 2>(T(q.x * r.x - q.y * r.y), T(q.y * r.x + q.x * r.y));
        dst[DIM / 2 + pair] = vec<T, 2>(T(k.x * r.x - k.y * r.y), T(k.y * r.x + k.x * r.y));
    """,
)

# One threadgroup of 256 threads per row: h = silu(gate) * val over 2048 features, then LayerNorm, in fp32 registers.
_SWIGLU_LN = mx.fast.metal_kernel(
    name="haruka_swiglu_ln",
    input_names=["gx", "w", "b"],
    output_names=["out"],
    source="""
        constexpr uint THREADS = 256, PER = HIDDEN / THREADS;
        const uint row = threadgroup_position_in_grid.x;
        const uint t = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
        threadgroup float part[THREADS / 32];
        const device T* g = gx + row * 2 * HIDDEN + t * PER;
        const device T* v = g + HIDDEN;
        float h[PER];
        float s = 0.f;
        for (uint i = 0; i < PER; ++i) {
            const float gi = float(g[i]);
            h[i] = gi / (1.f + metal::exp(-gi)) * float(v[i]);
            s += h[i];
        }
        s = simd_sum(s);
        if (lane == 0) part[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mean = 0.f;
        for (uint i = 0; i < THREADS / 32; ++i) mean += part[i];
        mean /= HIDDEN;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q = 0.f;
        for (uint i = 0; i < PER; ++i) { h[i] -= mean; q += h[i] * h[i]; }
        q = simd_sum(q);
        if (lane == 0) part[sg] = q;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float var = 0.f;
        for (uint i = 0; i < THREADS / 32; ++i) var += part[i];
        const float rstd = metal::rsqrt(var / HIDDEN + 1e-6f);
        device T* o = out + row * HIDDEN + t * PER;
        for (uint i = 0; i < PER; ++i) {
            const uint c = t * PER + i;
            o[i] = T(h[i] * rstd * float(w[c]) + float(b[c]));
        }
    """,
)


# Pillow's 8-bit separable resampling (Resample.c) with its fixed-point coefficients. Each pass writes the region
# [top, top + h) x [left, left + w) of its output and fills the rest with white, so the last pass produces the padded
# square directly. Bit-exact with Image.resize(BICUBIC), including Pillow's choice of pass order.
_RESAMPLE_SOURCE = """
    const uint c = thread_position_in_grid.x, x = thread_position_in_grid.y, y = thread_position_in_grid.z;
    const int out_w = geom[0], top = geom[1], left = geom[2], h = geom[3], w = geom[4], ksize = geom[5];
    const int src_w = geom[6], row0 = geom[7];
    const int yy = int(y) - top, xx = int(x) - left;
    uint8_t v = 255;
    if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
        const int i_out = AXIS == 0 ? xx : yy;
        const int first = bounds[i_out * 2], taps = bounds[i_out * 2 + 1];
        const auto in = AXIS == 0 ? src + ((row0 + yy) * src_w + first) * 3 + c : src + ((row0 + first) * src_w + xx) * 3 + c;
        const int stride = AXIS == 0 ? 3 : src_w * 3;
        int ss = 1 << 21;
        for (int i = 0; i < taps; ++i) ss += int(in[i * stride]) * kk[i_out * ksize + i];
        v = uint8_t(clamp(ss >> 22, 0, 255));
    }
    out[(y * out_w + x) * 3 + c] = v;
"""
_RESAMPLE = mx.fast.metal_kernel(
    name="haruka_resample",
    input_names=["src", "kk", "bounds", "geom"],
    output_names=["out"],
    source=_RESAMPLE_SOURCE,
)


def rope_qk(qkv: mx.array, cs: mx.array, tokens: int, dim: int, head_dim: int) -> mx.array:
    """qkv [rows, 3*dim] fp16 -> rotated [rows, 2*dim] (q then k) in the same row layout."""
    rows = qkv.shape[0]
    return _ROPE(
        inputs=[qkv, cs],
        template=[("T", qkv.dtype), ("TOKENS", tokens), ("DIM", dim), ("HEAD_DIM", head_dim)],
        grid=(dim // 2, rows, 1),
        threadgroup=(min(dim // 2, 256), 1, 1),
        output_shapes=[(rows, 2 * dim)],
        output_dtypes=[qkv.dtype],
    )[0]


def swiglu_layer_norm(gx: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """gx [rows, 2*hidden] (gate | value) -> LayerNorm(silu(gate) * value) [rows, hidden]."""
    rows, hidden = gx.shape[0], gx.shape[1] // 2
    return _SWIGLU_LN(
        inputs=[gx, w, b],
        template=[("T", gx.dtype), ("HIDDEN", hidden)],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[gx.dtype],
    )[0]


def _bicubic_coeffs(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Pillow's precompute_coeffs + normalize_coeffs_8bpc for BICUBIC (a = -0.5): fixed-point taps [out, ksize]
    and (first tap, tap count) per output. Same float64 operations in the same order as Resample.c."""
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 2.0 * filterscale
    ksize = math.ceil(support) * 2 + 1
    center = (np.arange(out_size) + 0.5) * scale
    xmin = np.maximum(np.trunc(center - support + 0.5), 0).astype(np.int64)
    xmax = np.minimum(np.trunc(center + support + 0.5).astype(np.int64), in_size) - xmin
    taps = np.arange(ksize)
    x = np.abs((taps[None, :] + xmin[:, None] - center[:, None] + 0.5) * (1.0 / filterscale))
    w = np.where(x < 1.0, (1.5 * x - 2.5) * x * x + 1, np.where(x < 2.0, (((x - 5) * x + 8) * x - 4) * -0.5, 0.0))
    w = np.where(taps[None, :] < xmax[:, None], w, 0.0)
    ww = np.cumsum(w, axis=1)[:, -1:]  # sequential sum, as the C loop
    w = np.divide(w, ww, out=w, where=ww != 0.0)
    kk = np.trunc(np.where(w < 0, -0.5 + w * (1 << 22), 0.5 + w * (1 << 22))).astype(np.int32)
    return kk, np.stack([xmin, xmax], axis=1).astype(np.int32)


def _resample_pass(
    src: mx.array,
    axis: int,
    kk: np.ndarray,
    bounds: np.ndarray,
    out_hw: tuple[int, int],
    region: tuple[int, int, int, int],
    row0: int = 0,
) -> mx.array:
    """One Pillow pass along axis (0: width, 1: height) into region (top, left, h, w) of an out_hw canvas."""
    out_h, out_w = out_hw
    top, left, h, w = region
    geom = np.array([out_w, top, left, h, w, kk.shape[1], src.shape[1], row0], np.int32)
    return _RESAMPLE(
        inputs=[src, mx.array(kk), mx.array(bounds), mx.array(geom)],
        template=[("AXIS", axis)],
        grid=(3, out_w, out_h),
        threadgroup=(3, min(out_w, 64), 1),
        output_shapes=[(out_h, out_w, 3)],
        output_dtypes=[mx.uint8],
    )[0]


def resize_pad(rgb: np.ndarray, new_w: int, new_h: int, size: int) -> mx.array:
    """uint8 [H, W, 3] -> uint8 [size, size, 3]: Pillow BICUBIC resize to (new_w, new_h), centered on white."""
    h, w = rgb.shape[:2]
    kh, bh = _bicubic_coeffs(w, new_w)
    kv, bv = _bicubic_coeffs(h, new_h)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    src = mx.array(rgb)
    if (h - new_h) > 0 and (h - new_h) > (w - new_w) * 2:  # Pillow: vertical first when height shrinks much more
        tmp = _resample_pass(src, 1, kv, bv, (new_h, w), (0, 0, new_h, w))
        return _resample_pass(tmp, 0, kh, bh, (size, size), (top, left, new_h, new_w))
    row0 = int(bv[0, 0])  # only the source rows the vertical pass reads
    rows = int(bv[-1, 0] + bv[-1, 1]) - row0
    bv = bv - np.array([row0, 0], np.int32)
    tmp = _resample_pass(src, 0, kh, bh, (rows, new_w), (0, 0, rows, new_w), row0)
    return _resample_pass(tmp, 1, kv, bv, (size, size), (top, left, new_h, new_w))
