#pragma once
// Flash attention, head_dim 64, on the QKV GEMM output: qkv [num_seqs * seq_len, 3072] fp16 with RoPE applied
// and q scaled by softmax_scale * log2(e). One CTA = WARPS x 16 query rows of one head.
// QK^T and PV accumulate in f16 within a 64-key block (2x the f32 rate on Ada); running output and normalizer are f32.
#include "common.cuh"

namespace attn {
constexpr int D = 64, CDIM = 1024, ROW = 3 * CDIM, BKV = 64;
__device__ __forceinline__ uint32_t swz128(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
template <int WARPS>
constexpr int smem_bytes() { return (WARPS * 16 + 4 * BKV) * 128; }  // Q + double-buffered K and V
}  // namespace attn

template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32) attention_kernel(const __half* __restrict__ qkv, __half* __restrict__ out,
                                                               int seq_len) {
  using namespace attn;
  constexpr int BQ = WARPS * 16, THREADS = WARPS * 32;
  static_assert(smem_bytes<WARPS>() <= 48 * 1024, "static shared memory limit");
  // Static smem measured faster than dynamic here.
  __shared__ __align__(128) uint8_t smem[smem_bytes<WARPS>()];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int qb = blockIdx.x, h = blockIdx.y, seq = blockIdx.z;
  const size_t tok0 = (size_t)seq * seq_len;
  const __half* base = qkv + tok0 * ROW + h * D;
  const int nblocks = seq_len / BKV;
  const uint32_t q_s = smem_u32(smem);
  // Address arithmetic, not a pointer array, which would spill to local memory.
  auto k_s = [&](int b) { return q_s + (BQ + b * BKV) * 128; };
  auto v_s = [&](int b) { return q_s + (BQ + (2 + b) * BKV) * 128; };

  auto load_rows = [&](uint32_t dst, const __half* src0, int row0, auto rows) {
#pragma unroll
    for (int i = tid; i < decltype(rows)::value * 8; i += THREADS) {
      const int r = i >> 3, c = i & 7;
      cp_async16(dst + swz128(r, c), src0 + (size_t)(row0 + r) * ROW + c * 8, true);
    }
  };
  load_rows(q_s, base, qb * BQ, std::integral_constant<int, BQ>{});
  load_rows(k_s(0), base + CDIM, 0, std::integral_constant<int, BKV>{});
  load_rows(v_s(0), base + 2 * CDIM, 0, std::integral_constant<int, BKV>{});
  cp_async_commit();

  uint32_t qf[4][4];
  float o[8][4];
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;

  for (int j = 0; j < nblocks; ++j) {
    if (j + 1 < nblocks) {
      load_rows(k_s((j + 1) & 1), base + CDIM, (j + 1) * BKV, std::integral_constant<int, BKV>{});
      load_rows(v_s((j + 1) & 1), base + 2 * CDIM, (j + 1) * BKV, std::integral_constant<int, BKV>{});
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    if (j == 0) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) ldsm_x4(qf[kk], q_s + swz128(warp * 16 + (lane & 15), kk * 2 + (lane >> 4)));
    }
    const uint32_t kb = k_s(j & 1), vb = v_s(j & 1);

    // S = Q K^T (16 x 64 per warp)
    uint32_t s16[8][2];
#pragma unroll
    for (int i = 0; i < 8; ++i) s16[i][0] = s16[i][1] = 0u;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t t[4];
        ldsm_x4(t, kb + swz128(np * 16 + ((lane >> 4) << 3) + (lane & 7), kk * 2 + ((lane >> 3) & 1)));
        mma_f16_f16(s16[2 * np], qf[kk], t[0], t[1]);
        mma_f16_f16(s16[2 * np + 1], qf[kk], t[2], t[3]);
      }
    // Online softmax in base 2 on packed f16 pairs. s16[i][hh] holds row (lane >> 2) + 8 * hh, columns
    // 8i + 2 * (lane & 3) + {0, 1}, which is already the PV mma's A-operand layout. The running max is rounded
    // to fp16 so P, the normalizer and the rescale factors share one reference.
    float corr[2];
    uint32_t pf[4][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      __half2 mx2 = *reinterpret_cast<const __half2*>(&s16[0][hh]);
#pragma unroll
      for (int i = 1; i < 8; ++i) mx2 = __hmax2(mx2, *reinterpret_cast<const __half2*>(&s16[i][hh]));
      float mx = fmaxf(__low2float(mx2), __high2float(mx2));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const __half m_h = __float2half_rn(fmaxf(m_run[hh], mx));
      const float m_new = __half2float(m_h);
      corr[hh] = ex2(m_run[hh] - m_new);
      m_run[hh] = m_new;
      const __half2 m2 = __half2half2(m_h);
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const __half2 d = __hsub2(*reinterpret_cast<const __half2*>(&s16[i][hh]), m2);
        uint32_t p;
        asm("ex2.approx.f16x2 %0, %1;" : "=r"(p) : "r"(*reinterpret_cast<const uint32_t*>(&d)));
        const float2 pf2 = unpack_half2(p);
        sum += pf2.x + pf2.y;
        pf[i >> 1][(i & 1) * 2 + hh] = p;
      }
      l_run[hh] = l_run[hh] * corr[hh] + sum;
    }
    uint32_t o16[8][2];
#pragma unroll
    for (int i = 0; i < 8; ++i) o16[i][0] = o16[i][1] = 0u;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t t[4];
        ldsm_x4_t(t, vb + swz128(kk * 16 + (lane & 15), dp * 2 + (lane >> 4)));
        mma_f16_f16(o16[2 * dp], pf[kk], t[0], t[1]);
        mma_f16_f16(o16[2 * dp + 1], pf[kk], t[2], t[3]);
      }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float2 a = unpack_half2(o16[i][0]), b = unpack_half2(o16[i][1]);
      o[i][0] = o[i][0] * corr[0] + a.x;
      o[i][1] = o[i][1] * corr[0] + a.y;
      o[i][2] = o[i][2] * corr[1] + b.x;
      o[i][3] = o[i][3] * corr[1] + b.y;
    }
    __syncthreads();
  }

  // Normalize, stage in the Q area, store full 128-byte rows.
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    float l = l_run[hh];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    const float inv = 1.f / l;
    const int r = warp * 16 + (lane >> 2) + hh * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = i * 8 + (lane & 3) * 2;
      *reinterpret_cast<__half2*>(smem + r * 128 + c * 2) = __floats2half2_rn(o[i][2 * hh] * inv, o[i][2 * hh + 1] * inv);
    }
  }
  __syncthreads();
  for (int i = tid; i < BQ * 8; i += THREADS) {
    const int r = i >> 3, c = i & 7;
    *reinterpret_cast<uint4*>(out + (tok0 + qb * BQ + r) * CDIM + h * D + c * 8) =
        *reinterpret_cast<const uint4*>(smem + r * 128 + c * 16);
  }
}
