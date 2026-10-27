#pragma once
// Y[m, n] = sum_k A[m, k] * W[n, k] * row_scale[m] * col_scale[n] + bias[n], int8 A and W (K contiguous),
// int32 accumulation, fused fp16 epilogues.
// 128x128 tiles so two CTAs fit per SM. These GEMMs are bound by L2 bandwidth on AD102, and the second CTA
// overlaps each tile's prologue/epilogue with the other's mainloop. Larger tiles measured slower.
#include "common.cuh"

enum Epi : int {
  EPI_QKV_ROPE = 0,  // fp16 out; RoPE on the q/k sections, q additionally scaled by q_scale
  EPI_RESIDUAL = 1,  // fp16 residual += y
  EPI_GELU_F16 = 2,  // fp16 out = gelu(y)
};

struct GemmArgs {
  const int8_t* A;
  const int8_t* W;
  int M, N, K;
  const float* row_scale;  // [M]
  const float* col_scale;  // [N]
  const float* bias;       // [N]
  __half* out;             // [M, N] output, or the residual stream for EPI_RESIDUAL
  const float* rope_cos;   // [rope_len, 32]
  const float* rope_sin;
  int rope_len;            // token m uses RoPE row m % rope_len (576 windowed, 5184 global)
  float q_scale;
};

template <int STAGES_>
struct GemmCfg {
  static constexpr int BM = 128, BN = 128, BK = 64, WM = 64, WN = 64, STAGES = STAGES_;
  static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN, THREADS = WARPS_M * WARPS_N * 32;
  static constexpr int MT = WM / 16, NT = WN / 8;
  static constexpr int PIPE_SMEM = STAGES * (BM + BN) * BK;
  static constexpr int OUT_STRIDE = BN + 8;  // halves; +16B keeps fragment writes conflict-free
  static constexpr int EPI_SMEM = BM * OUT_STRIDE * 2;
  static constexpr int SMEM = PIPE_SMEM > EPI_SMEM ? PIPE_SMEM : EPI_SMEM;
  static_assert(2 * SMEM <= 99 * 1024, "two CTAs must fit per SM");
};

// 64-byte smem rows of 4 x 16B chunks, XOR-swizzled so ldmatrix and cp.async are conflict-free.
__device__ __forceinline__ uint32_t swz64(int row, int chunk) { return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }

template <class C, int EPI>
__global__ void __launch_bounds__(C::THREADS, 2) gemm_s8_kernel(const GemmArgs args) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / C::WARPS_N, wn = warp % C::WARPS_N;
  const int tiles_n = (args.N + C::BN - 1) / C::BN;
  const int bn = blockIdx.x % tiles_n, bm = blockIdx.x / tiles_n;
  const int m0 = bm * C::BM, n0 = bn * C::BN;
  const int KT = args.K / C::BK;
  uint8_t* sA = smem;
  uint8_t* sB = smem + C::STAGES * C::BM * C::BK;

  auto load_stage = [&](int stage, int kt) {
    const int kb = kt * C::BK;
    const uint32_t a = smem_u32(sA + stage * C::BM * C::BK);
#pragma unroll
    for (int i = tid; i < C::BM * 4; i += C::THREADS) {
      const int r = i >> 2, c = i & 3, m = m0 + r;
      cp_async16(a + swz64(r, c), args.A + (size_t)min(m, args.M - 1) * args.K + kb + c * 16, m < args.M);
    }
    const uint32_t b = smem_u32(sB + stage * C::BN * C::BK);
#pragma unroll
    for (int i = tid; i < C::BN * 4; i += C::THREADS) {
      const int r = i >> 2, c = i & 3, n = n0 + r;
      cp_async16(b + swz64(r, c), args.W + (size_t)min(n, args.N - 1) * args.K + kb + c * 16, n < args.N);
    }
  };

  int acc[C::MT][C::NT][4];
#pragma unroll
  for (int i = 0; i < C::MT; ++i)
#pragma unroll
    for (int j = 0; j < C::NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;

  auto compute_stage = [&](int stage) {
    const uint32_t a_base = smem_u32(sA + stage * C::BM * C::BK);
    const uint32_t b_base = smem_u32(sB + stage * C::BN * C::BK);
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      uint32_t bf[C::NT][2];
#pragma unroll
      for (int np = 0; np < C::NT / 2; ++np) {
        const int r = wn * C::WN + np * 16 + ((lane >> 4) << 3) + (lane & 7);
        uint32_t t[4];
        ldsm_x4(t, b_base + swz64(r, ks * 2 + ((lane >> 3) & 1)));
        bf[2 * np][0] = t[0];
        bf[2 * np][1] = t[1];
        bf[2 * np + 1][0] = t[2];
        bf[2 * np + 1][1] = t[3];
      }
#pragma unroll
      for (int mt = 0; mt < C::MT; ++mt) {
        uint32_t af[4];
        ldsm_x4(af, a_base + swz64(wm * C::WM + mt * 16 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
        for (int nt = 0; nt < C::NT; ++nt) mma_s8_s32(acc[mt][nt], af, bf[nt][0], bf[nt][1]);
      }
    }
  };

#pragma unroll
  for (int s = 0; s < C::STAGES - 1; ++s) {
    if (s < KT) load_stage(s, s);
    cp_async_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<C::STAGES - 2>();
    __syncthreads();
    const int nk = kt + C::STAGES - 1;
    if (nk < KT) load_stage(nk % C::STAGES, nk);
    cp_async_commit();
    compute_stage(kt % C::STAGES);
  }
  cp_async_wait<0>();
  __syncthreads();

  // Epilogue in registers, then staged through smem for coalesced 16B stores.
  __half* o16 = reinterpret_cast<__half*>(smem);
#pragma unroll
  for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = wm * C::WM + mt * 16 + (lane >> 2) + h * 8;
      const int m = min(m0 + r, args.M - 1);
      const float rs = args.row_scale[m];
      const int rope_row = EPI == EPI_QKV_ROPE ? (m % args.rope_len) * 32 : 0;
#pragma unroll
      for (int nt = 0; nt < C::NT; ++nt) {
        const int c = wn * C::WN + nt * 8 + (lane & 3) * 2;
        const int n = min(n0 + c, args.N - 2);
        const float2 cs = *reinterpret_cast<const float2*>(args.col_scale + n);
        const float2 bi = *reinterpret_cast<const float2*>(args.bias + n);
        float y0 = static_cast<float>(acc[mt][nt][2 * h]) * rs * cs.x + bi.x;
        float y1 = static_cast<float>(acc[mt][nt][2 * h + 1]) * rs * cs.y + bi.y;
        if constexpr (EPI == EPI_GELU_F16) {
          y0 = gelu(y0);
          y1 = gelu(y1);
        }
        if constexpr (EPI == EPI_QKV_ROPE) {
          const int section = n >> 10;  // 0 q, 1 k, 2 v (C = 1024); uniform across a tile
          if (section < 2) {
            const int pr = rope_row + ((n & 63) >> 1);
            const float co = args.rope_cos[pr], si = args.rope_sin[pr];
            const float r0 = y0 * co - y1 * si, r1 = y0 * si + y1 * co;
            const float qs = section == 0 ? args.q_scale : 1.f;
            y0 = r0 * qs;
            y1 = r1 * qs;
          }
        }
        *reinterpret_cast<__half2*>(o16 + r * C::OUT_STRIDE + c) = __floats2half2_rn(y0, y1);
      }
    }
  __syncthreads();
  constexpr int CPR = C::BN / 8;
#pragma unroll 4
  for (int i = tid; i < C::BM * CPR; i += C::THREADS) {
    const int r = i / CPR, cc = i % CPR, m = m0 + r, n = n0 + cc * 8;
    if (m >= args.M || n >= args.N) continue;
    uint4 v = *reinterpret_cast<const uint4*>(o16 + r * C::OUT_STRIDE + cc * 8);
    uint4* dst = reinterpret_cast<uint4*>(args.out + (size_t)m * args.N + n);
    if constexpr (EPI == EPI_RESIDUAL) {
      const uint4 x = *dst;
      __half2* pv = reinterpret_cast<__half2*>(&v);
      const __half2* px = reinterpret_cast<const __half2*>(&x);
#pragma unroll
      for (int j = 0; j < 4; ++j) pv[j] = __hadd2(pv[j], px[j]);
    }
    *dst = v;
  }
}
