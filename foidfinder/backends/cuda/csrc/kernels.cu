#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "attention.cuh"
#include "gemm.cuh"

// fp16 rows of 1024 -> [LayerNorm if NORM] -> 1024-point Hadamard rotation -> int8, value = code * row_scale.
// The rotation spreads activation outliers over all channels; the weights carry the same rotation.
// One warp per row; lane owns columns j*256 + lane*8 + e. FWHT: column bits 0-2 and 8-9 in registers,
// bits 3-7 across lanes.
__device__ __forceinline__ void fwht1024_warp(float (&v)[32], int lane) {
#pragma unroll
  for (int h = 1; h < 8; h <<= 1)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (!(e & h)) {
          const float a = v[j * 8 + e], b = v[j * 8 + e + h];
          v[j * 8 + e] = a + b;
          v[j * 8 + e + h] = a - b;
        }
#pragma unroll
  for (int h = 1; h < 4; h <<= 1)
#pragma unroll
    for (int j = 0; j < 4; ++j)
      if (!(j & h))
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float a = v[j * 8 + e], b = v[(j + h) * 8 + e];
          v[j * 8 + e] = a + b;
          v[(j + h) * 8 + e] = a - b;
        }
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
    const bool upper = lane & m;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const float o = __shfl_xor_sync(0xffffffffu, v[i], m);
      v[i] = upper ? o - v[i] : v[i] + o;
    }
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] *= 0.03125f;  // 1/sqrt(1024)
}

template <bool NORM>
__global__ void __launch_bounds__(256) rot_q8_1024_kernel(const __half* __restrict__ x, const float* __restrict__ gamma,
                                                          const float* __restrict__ beta, int T,
                                                          int8_t* __restrict__ out, float* __restrict__ row_scale) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= T) return;
  const __half* xr = x + (size_t)row * 1024;
  float v[32];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4 u = *reinterpret_cast<const uint4*>(xr + j * 256 + lane * 8);
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f = __half22float2(h[e]);
      v[j * 8 + 2 * e] = f.x;
      v[j * 8 + 2 * e + 1] = f.y;
    }
  }
  if constexpr (NORM) {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) sum += v[i];
    const float mean = warp_sum(sum) * (1.f / 1024.f);
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      v[i] -= mean;
      sq += v[i] * v[i];
    }
    const float rstd = rsqrtf(warp_sum(sq) * (1.f / 1024.f) + 1e-5f);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int c = j * 256 + lane * 8;
      const float4 g0 = *reinterpret_cast<const float4*>(gamma + c), g1 = *reinterpret_cast<const float4*>(gamma + c + 4);
      const float4 b0 = *reinterpret_cast<const float4*>(beta + c), b1 = *reinterpret_cast<const float4*>(beta + c + 4);
      const float g[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
      const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) v[j * 8 + e] = v[j * 8 + e] * rstd * g[e] + b[e];
    }
  }
  fwht1024_warp(v, lane);
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
  amax = warp_max(amax);
  const float scale = fmaxf(amax, 1e-12f) / 127.f, inv = 1.f / scale;
  int8_t* orow = out + (size_t)row * 1024;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint2 pk;
    uint16_t* p16 = reinterpret_cast<uint16_t*>(&pk);
#pragma unroll
    for (int e = 0; e < 4; ++e) p16[e] = q8x2(v[j * 8 + 2 * e] * inv, v[j * 8 + 2 * e + 1] * inv);
    *reinterpret_cast<uint2*>(orow + j * 256 + lane * 8) = pk;
  }
  if (lane == 0) row_scale[row] = scale;
}

// fp16 rows of C (multiple of 128) -> block-diagonal 128-point Hadamard rotation -> int8 with a per-row scale.
// One CTA per row; the rotated row is staged in smem so the scale sees the whole row's amax.
__global__ void __launch_bounds__(128) rot_q8_h128_kernel(const __half* __restrict__ x, int C,
                                                          int8_t* __restrict__ out, float* __restrict__ row_scale) {
  extern __shared__ float buf[];
  __shared__ float red[4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const size_t row = blockIdx.x;
  const __half* xr = x + row * C;
  float amax = 0.f;
  for (int blk = warp; blk < C / 128; blk += 4) {
    // lane owns columns blk*128 + lane*4 + e: column bits 0-1 in registers, bits 2-6 across lanes
    const uint2 u = *reinterpret_cast<const uint2*>(xr + blk * 128 + lane * 4);
    const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    const float a0 = f0.x + f0.y, a1 = f0.x - f0.y, a2 = f1.x + f1.y, a3 = f1.x - f1.y;  // bit 0
    float v[4] = {a0 + a2, a1 + a3, a0 - a2, a1 - a3};                                   // bit 1
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
      const bool upper = lane & m;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float o = __shfl_xor_sync(0xffffffffu, v[i], m);
        v[i] = upper ? o - v[i] : v[i] + o;
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[i] *= 0.08838834764831845f;  // 1/sqrt(128)
      amax = fmaxf(amax, fabsf(v[i]));
    }
    *reinterpret_cast<float4*>(buf + blk * 128 + lane * 4) = make_float4(v[0], v[1], v[2], v[3]);
  }
  amax = warp_max(amax);
  if (lane == 0) red[warp] = amax;
  __syncthreads();
  amax = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  const float scale = fmaxf(amax, 1e-12f) / 127.f, inv = 1.f / scale;
  int8_t* orow = out + row * C;
  for (int c = tid * 8; c < C; c += 128 * 8) {
    const float4 a = *reinterpret_cast<const float4*>(buf + c), b = *reinterpret_cast<const float4*>(buf + c + 4);
    uint2 pk;
    uint16_t* p16 = reinterpret_cast<uint16_t*>(&pk);
    p16[0] = q8x2(a.x * inv, a.y * inv);
    p16[1] = q8x2(a.z * inv, a.w * inv);
    p16[2] = q8x2(b.x * inv, b.y * inv);
    p16[3] = q8x2(b.z * inv, b.w * inv);
    *reinterpret_cast<uint2*>(orow + c) = pk;
  }
  if (tid == 0) row_scale[row] = scale;
}

template <class Kernel>
static void allow_smem(Kernel kernel, int bytes) {
  TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess);
}

template <int STAGES, int EPI>
static void launch_gemm(const GemmArgs& a, cudaStream_t stream) {
  using C = GemmCfg<STAGES>;
  static const bool configured = (allow_smem(gemm_s8_kernel<C, EPI>, C::SMEM), true);
  (void)configured;
  const int tiles = ((a.N + C::BN - 1) / C::BN) * ((a.M + C::BM - 1) / C::BM);
  gemm_s8_kernel<C, EPI><<<tiles, C::THREADS, C::SMEM, stream>>>(a);
}

template <int STAGES>
static void dispatch_epi(const GemmArgs& a, int epi, cudaStream_t s) {
  switch (epi) {
    case EPI_QKV_ROPE: return launch_gemm<STAGES, EPI_QKV_ROPE>(a, s);
    case EPI_RESIDUAL: return launch_gemm<STAGES, EPI_RESIDUAL>(a, s);
    case EPI_GELU_F16: return launch_gemm<STAGES, EPI_GELU_F16>(a, s);
  }
  TORCH_CHECK(false, "bad epilogue ", epi);
}

// A [M, K] int8, W [N, K] int8 -> epilogue into out [M, N] fp16. stages: 2 or 3.
void gemm_s8(at::Tensor A, at::Tensor W, at::Tensor row_scale, at::Tensor col_scale, at::Tensor bias,
             at::Tensor out, int64_t epi, int64_t stages, std::optional<at::Tensor> rope_cos,
             std::optional<at::Tensor> rope_sin, double q_scale) {
  TORCH_CHECK(A.scalar_type() == at::kChar && W.scalar_type() == at::kChar && A.is_contiguous() && W.is_contiguous());
  TORCH_CHECK(out.scalar_type() == at::kHalf && out.is_contiguous());
  GemmArgs a{};
  a.A = A.data_ptr<int8_t>();
  a.W = W.data_ptr<int8_t>();
  a.M = A.size(0);
  a.K = A.size(1);
  a.N = W.size(0);
  TORCH_CHECK(W.size(1) == a.K && a.K % 64 == 0 && a.N % 16 == 0 && out.size(0) == a.M && out.size(1) == a.N);
  TORCH_CHECK(row_scale.numel() == a.M && col_scale.numel() == a.N && bias.numel() == a.N);
  a.row_scale = row_scale.data_ptr<float>();
  a.col_scale = col_scale.data_ptr<float>();
  a.bias = bias.data_ptr<float>();
  a.out = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  if (epi == EPI_QKV_ROPE) {
    TORCH_CHECK(rope_cos && rope_sin && a.N == 3072, "QKV epilogue needs RoPE tables and N = 3 * 1024");
    a.rope_cos = rope_cos->data_ptr<float>();
    a.rope_sin = rope_sin->data_ptr<float>();
    a.rope_len = rope_cos->size(0);
  }
  a.q_scale = static_cast<float>(q_scale);
  const at::cuda::CUDAGuard guard(A.device());
  auto s = at::cuda::getCurrentCUDAStream();
  if (stages == 3) dispatch_epi<3>(a, static_cast<int>(epi), s);
  else if (stages == 2) dispatch_epi<2>(a, static_cast<int>(epi), s);
  else TORCH_CHECK(false, "stages must be 2 or 3");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// x [T, 1024] fp16 -> Hadamard-rotated int8 codes [T, 1024] + row_scale [T]; LayerNorm first if gamma is given.
void rot_q8_1024(at::Tensor x, std::optional<at::Tensor> gamma, std::optional<at::Tensor> beta,
                 at::Tensor out, at::Tensor row_scale) {
  TORCH_CHECK(x.size(1) == 1024 && x.is_contiguous() && x.scalar_type() == at::kHalf && out.scalar_type() == at::kChar);
  const int T = x.size(0);
  const at::cuda::CUDAGuard guard(x.device());
  auto s = at::cuda::getCurrentCUDAStream();
  auto px = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  const dim3 grid((T + 7) / 8);
  if (gamma)
    rot_q8_1024_kernel<true><<<grid, 256, 0, s>>>(px, gamma->data_ptr<float>(), beta->data_ptr<float>(), T,
                                                  out.data_ptr<int8_t>(), row_scale.data_ptr<float>());
  else
    rot_q8_1024_kernel<false><<<grid, 256, 0, s>>>(px, nullptr, nullptr, T, out.data_ptr<int8_t>(),
                                                   row_scale.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// x [T, C] fp16 (C multiple of 128) -> block-Hadamard-128 rotated int8 codes [T, C] + row_scale [T].
void rot_q8_h128(at::Tensor x, at::Tensor out, at::Tensor row_scale) {
  const int T = x.size(0), C = x.size(1);
  TORCH_CHECK(C % 128 == 0 && x.is_contiguous() && x.scalar_type() == at::kHalf && out.scalar_type() == at::kChar);
  const at::cuda::CUDAGuard guard(x.device());
  const int smem = C * sizeof(float);
  static int configured_bytes = 0;
  if (smem > configured_bytes) {
    allow_smem(rot_q8_h128_kernel, smem);
    configured_bytes = smem;
  }
  rot_q8_h128_kernel<<<T, 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), C, out.data_ptr<int8_t>(), row_scale.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// qkv [T, 3072] fp16 (q pre-scaled, q/k rotated) -> out [T, 1024] fp16; 96 query rows per CTA
// divide both 576 and 5184.
void attention(at::Tensor qkv, at::Tensor out, int64_t seq_len) {
  constexpr int WARPS = 6, BQ = WARPS * 16;
  TORCH_CHECK(qkv.size(1) == 3072 && qkv.size(0) % seq_len == 0 && seq_len % BQ == 0 && seq_len % attn::BKV == 0);
  TORCH_CHECK(qkv.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf && out.size(1) == 1024);
  const at::cuda::CUDAGuard guard(qkv.device());
  const dim3 grid(seq_len / BQ, 16, qkv.size(0) / seq_len);
  attention_kernel<WARPS><<<grid, WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(qkv.data_ptr<at::Half>()), reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      seq_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(foidfinder, m) {
  m.def(
      "gemm_s8(Tensor A, Tensor W, Tensor row_scale, Tensor col_scale, Tensor bias, Tensor(a!) out, int epi, "
      "int stages, Tensor? rope_cos, Tensor? rope_sin, float q_scale) -> ()");
  m.def("rot_q8_1024(Tensor x, Tensor? gamma, Tensor? beta, Tensor(a!) out, Tensor(b!) row_scale) -> ()");
  m.def("rot_q8_h128(Tensor x, Tensor(a!) out, Tensor(b!) row_scale) -> ()");
  m.def("attention(Tensor qkv, Tensor(a!) out, int seq_len) -> ()");
}

TORCH_LIBRARY_IMPL(foidfinder, CUDA, m) {
  m.impl("gemm_s8", &gemm_s8);
  m.impl("rot_q8_1024", &rot_q8_1024);
  m.impl("rot_q8_h128", &rot_q8_h128);
  m.impl("attention", &attention);
}
