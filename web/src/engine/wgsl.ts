// WGSL for the Haruka forward. Shapes are compile-time constants so loops unroll and indexing folds.

export interface Features {
  /** Subgroup ops usable for 16-lane row reductions (subgroups enabled and the minimum subgroup size >= 16). */
  subgroups16: boolean;
}

const header = (f: Features) => `enable f16;${f.subgroups16 ? "\nenable subgroups;" : ""}\n`;

export interface Tile {
  BM: number; // rows per workgroup
  BN: number; // columns per workgroup
  BK: number; // K step
  TM: number; // rows per thread (strided by BM / TM)
  TN: number; // columns per thread, in groups of 4 (strided by BN / TN * 4)
}

/** Output handling after C = A W^T (+ bias), per 4-column group. */
export type Epilogue =
  | { kind: "bias" } // out f16 [M, N]
  | { kind: "residual" } // out f32 [M, N] += C + bias
  | { kind: "qkv"; heads: number; dim: number } // RoPE on q, k; out f16 [3, heads, M, 64]; binding 4: cos/sin
  | { kind: "patch" }; // out f32 [M + 1, N], row r + 1 = C + bias[r] (bias is [M, N])

/**
 * C[M, N] = A[M, K] · WT[K, N] (A f16 row-major, WT the weight transposed), then the epilogue.
 *
 * Both tiles are staged as plain vec4 copies in workgroup memory: A as [m][k / 4], WT as [k][n / 4]. Each thread
 * owns TM contiguous rows and TN contiguous columns; per 4 k steps it reads one A vec4 per row and TN / 4 WT vec4s
 * per k. Products accumulate in f16 for `flushK` k steps, then into f32 accumulators: f16 arithmetic runs at
 * about twice the rate here because it halves workgroup-memory traffic, and flushing every 8 keeps the deviation
 * from fp32 at the level of fp16 inference with f32 accumulation. The next K tile's global loads are issued
 * before the current tile is consumed. K must be a multiple of BK and N of BN; rows past M are masked.
 */
export function matmul(M: number, N: number, K: number, t: Tile, ep: Epilogue, flushK = 8): string {
  const { BM, BN, BK, TM, TN } = t;
  const TX = BN / TN;
  const TY = BM / TM;
  const threads = TX * TY;
  const BKV = BK / 4;
  const BNV = BN / 4;
  const GB = TN / 4;
  if (K % BK || N % BN || BK % 4 || TN % 4 || BM % TM || BN % TN) throw new Error(`bad tile for ${M}x${N}x${K}`);
  const aIters = (BM * BKV) / threads;
  const bIters = (BK * BNV) / threads;
  if (!Number.isInteger(aIters) || !Number.isInteger(bIters)) throw new Error("tile loads must divide threads");

  const outDecl =
    ep.kind === "bias" || ep.kind === "qkv"
      ? "var<storage, read_write> C: array<vec4<f16>>;"
      : "var<storage, read_write> C: array<vec4<f32>>;";
  const extra = ep.kind === "qkv" ? "@group(0) @binding(4) var<storage, read> cs: array<vec2<f32>>;" : "";
  const biasAt = ep.kind === "patch" ? "bias[(r * N + c) / 4u]" : "bias[c / 4u]";

  let decls = "";
  for (let i = 0; i < TM; i++) for (let j = 0; j < GB; j++) decls += `var acc${i}_${j} = vec4<f32>(0.0);\n  `;
  let tileInit = "";
  let tileFlush = "";
  for (let i = 0; i < TM; i++)
    for (let j = 0; j < GB; j++) {
      tileInit += `var h${i}_${j} = vec4<f16>(0.0);\n    `;
      tileFlush += `acc${i}_${j} += vec4<f32>(h${i}_${j});\n    `;
    }
  if (flushK % 4 || BK % flushK) throw new Error("flushK must divide BK in steps of 4");
  let inner = "";
  for (let k4 = 0; k4 < BKV; k4++) {
    if (k4 > 0 && (k4 * 4) % flushK === 0) inner += `${tileFlush}${tileInit.replaceAll("var ", "")}`;
    inner += "{\n      ";
    for (let i = 0; i < TM; i++) inner += `let a${i} = As[abase + ${i * BKV + k4}u];\n      `;
    for (let e = 0; e < 4; e++) {
      for (let j = 0; j < GB; j++) inner += `let b${e}_${j} = Bs[bbase + ${(k4 * 4 + e) * BNV + j}u];\n      `;
      for (let i = 0; i < TM; i++)
        for (let j = 0; j < GB; j++)
          inner += `h${i}_${j} = fma(vec4<f16>(a${i}[${e}]), b${e}_${j}, h${i}_${j});\n      `;
    }
    inner += "}\n      ";
  }
  let store = "";
  for (let i = 0; i < TM; i++) {
    store += `{ let r = m0 + ty * ${TM}u + ${i}u; if (r < M) {\n`;
    for (let j = 0; j < GB; j++)
      store += `  { let c = n0 + (tx * ${GB}u + ${j}u) * 4u; let v = acc${i}_${j} + vec4<f32>(${biasAt});\n${epilogue(ep)} }\n`;
    store += "} }\n  ";
  }
  let regDecls = "";
  let loadRegs = "";
  let storeRegs = "";
  for (let s = 0; s < aIters; s++) {
    regDecls += `var ra${s}: vec4<f16>;\n  `;
    loadRegs += `{ let i = t + ${s * threads}u; let r = i / BKV; ra${s} = vec4<f16>(0.0); if (m0 + r < M) { ra${s} = A[(m0 + r) * (K / 4u) + k0 / 4u + i % BKV]; } }\n    `;
    storeRegs += `As[t + ${s * threads}u] = ra${s};\n    `;
  }
  for (let s = 0; s < bIters; s++) {
    regDecls += `var rb${s}: vec4<f16>;\n  `;
    loadRegs += `{ let i = t + ${s * threads}u; rb${s} = WT[(k0 + i / BNV) * (N / 4u) + n0 / 4u + i % BNV]; }\n    `;
    storeRegs += `Bs[t + ${s * threads}u] = rb${s};\n    `;
  }

  return `enable f16;
const M = ${M}u; const N = ${N}u; const K = ${K}u;
const BK = ${BK}u; const BKV = ${BKV}u; const BNV = ${BNV}u;
@group(0) @binding(0) var<storage, read> A: array<vec4<f16>>;   // [M, K]
@group(0) @binding(1) var<storage, read> WT: array<vec4<f16>>;  // [K, N]
@group(0) @binding(2) var<storage, read> bias: array<vec4<f16>>;
@group(0) @binding(3) ${outDecl}
${extra}
var<workgroup> As: array<vec4<f16>, ${BM * BKV}>;  // [m][k / 4]
var<workgroup> Bs: array<vec4<f16>, ${BK * BNV}>;  // [k][n / 4]

@compute @workgroup_size(${threads})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let m0 = wg.y * ${BM}u;
  let n0 = wg.x * ${BN}u;
  let tx = t % ${TX}u;
  let ty = t / ${TX}u;
  let abase = ty * ${TM * BKV}u;
  let bbase = tx * ${GB}u;
  ${decls}
  ${regDecls}
  var k0 = 0u;
  ${loadRegs}
  loop {
    ${storeRegs}
    workgroupBarrier();
    k0 += BK;
    if (k0 < K) {
      ${loadRegs}
    }
    ${tileInit}
    ${inner}
    ${tileFlush}
    workgroupBarrier();
    if (k0 >= K) { break; }
  }
  ${store}
}
`;
}

function epilogue(ep: Epilogue): string {
  switch (ep.kind) {
    case "bias":
      return "C[(r * N + c) / 4u] = vec4<f16>(v);";
    case "residual":
      return "C[(r * N + c) / 4u] += v;";
    case "patch":
      return "C[((r + 1u) * N + c) / 4u] = v;";
    case "qkv": {
      const { heads, dim } = ep;
      return `let part = c / ${dim}u;
let within = c % ${dim}u;
let head = within / 64u;
let d = within % 64u;
var o = v;
if (part < 2u) {
  let r0 = cs[r * 32u + d / 2u];
  let r1 = cs[r * 32u + d / 2u + 1u];
  o = vec4<f32>(v.x * r0.x - v.y * r0.y, v.y * r0.x + v.x * r0.y, v.z * r1.x - v.w * r1.y, v.w * r1.x + v.z * r1.y);
}
C[((part * ${heads}u + head) * M + r) * 16u + d / 4u] = vec4<f16>(o);`;
    }
  }
}

/** Reduction of one value per thread over a row's 16 consecutive lanes (lane = tx in 0..15). */
function rowReduce(f: Features, op: "max" | "add"): string {
  const combine = op === "max" ? "max(v, o)" : "v + o";
  if (f.subgroups16) {
    return `fn reduce_${op}(x: f32, slot: u32) -> f32 {
  var v = x;
  for (var m = 1u; m < 16u; m <<= 1u) { let o = subgroupShuffleXor(v, m); v = ${combine}; }
  return v;
}`;
  }
  return `fn reduce_${op}(x: f32, slot: u32) -> f32 {
  red[slot] = x;
  workgroupBarrier();
  let base = slot & ~15u;
  var v = red[base];
  for (var i = 1u; i < 16u; i++) { let o = red[base + i]; v = ${combine}; }
  workgroupBarrier();
  return v;
}`;
}

/**
 * Flash attention, head_dim 64, scale already folded into q. qkv f16 [3, H, M, 64] (q, k rotated) -> out f16 [M, H * 64].
 * Workgroup: 64 queries of one head, 256 threads as 16 x 16; each thread owns 4 query rows x (4 keys | 4 dims).
 */
export function attention(M: number, H: number, f: Features): string {
  return `${header(f)}
const M = ${M}u; const H = ${H}u;
@group(0) @binding(0) var<storage, read> qkv: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f16>>;
var<workgroup> Qs: array<vec4<f16>, ${64 * 17}>;   // [q][d / 4], row stride 17 against bank conflicts
var<workgroup> KPs: array<vec4<f16>, ${64 * 17}>;  // K tile [k][d / 4], then P tile [q][k / 4]
var<workgroup> Vs: array<vec4<f16>, ${64 * 16}>;   // [k][d / 4]
${f.subgroups16 ? "" : "var<workgroup> red: array<f32, 1024>;"}
${rowReduce(f, "max")}
${rowReduce(f, "add")}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let h = wg.y;
  let q0 = wg.x * 64u;
  let tx = t % 16u;
  let ty = t / 16u;
  let qbase = h * M;
  let kbase = (H + h) * M;
  let vbase = (2u * H + h) * M;
  for (var s = 0u; s < 4u; s++) {
    let i = t + s * 256u;
    let r = i / 16u;
    let c = i % 16u;
    var v = vec4<f16>(0.0);
    if (q0 + r < M) { v = qkv[(qbase + q0 + r) * 16u + c]; }
    Qs[r * 17u + c] = v;
  }
  var m = array<f32, 4>(-1e30, -1e30, -1e30, -1e30);
  var l = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  var o = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
  for (var k0 = 0u; k0 < M; k0 += 64u) {
    workgroupBarrier();
    for (var s = 0u; s < 4u; s++) {
      let i = t + s * 256u;
      let r = i / 16u;
      let c = i % 16u;
      var kv = vec4<f16>(0.0);
      var vv = vec4<f16>(0.0);
      if (k0 + r < M) {
        kv = qkv[(kbase + k0 + r) * 16u + c];
        vv = qkv[(vbase + k0 + r) * 16u + c];
      }
      KPs[r * 17u + c] = kv;
      Vs[r * 16u + c] = vv;
    }
    workgroupBarrier();
    var sc: array<array<f32, 4>, 4>;
    for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { sc[i][j] = 0.0; } }
    for (var c = 0u; c < 16u; c++) {
      let qa = vec4<f32>(Qs[(ty * 4u) * 17u + c]);
      let qb = vec4<f32>(Qs[(ty * 4u + 1u) * 17u + c]);
      let qc = vec4<f32>(Qs[(ty * 4u + 2u) * 17u + c]);
      let qd = vec4<f32>(Qs[(ty * 4u + 3u) * 17u + c]);
      for (var j = 0u; j < 4u; j++) {
        let kj = vec4<f32>(KPs[(tx * 4u + j) * 17u + c]);
        sc[0][j] += dot(qa, kj);
        sc[1][j] += dot(qb, kj);
        sc[2][j] += dot(qc, kj);
        sc[3][j] += dot(qd, kj);
      }
    }
    var p: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i++) {
      for (var j = 0u; j < 4u; j++) { if (k0 + tx * 4u + j >= M) { sc[i][j] = -1e30; } }
      let local = max(max(sc[i][0], sc[i][1]), max(sc[i][2], sc[i][3]));
      let mx = reduce_max(local, (ty * 4u + i) * 16u + tx);
      let mn = max(m[i], mx);
      let alpha = exp(m[i] - mn);
      p[i] = exp(vec4<f32>(sc[i][0], sc[i][1], sc[i][2], sc[i][3]) - vec4<f32>(mn));
      let rs = reduce_add(p[i].x + p[i].y + p[i].z + p[i].w, (ty * 4u + i) * 16u + tx);
      l[i] = l[i] * alpha + rs;
      o[i] *= alpha;
      m[i] = mn;
    }
    workgroupBarrier();
    for (var i = 0u; i < 4u; i++) { KPs[(ty * 4u + i) * 17u + tx] = vec4<f16>(p[i]); }
    workgroupBarrier();
    for (var k4 = 0u; k4 < 16u; k4++) {
      let pa = vec4<f32>(KPs[(ty * 4u) * 17u + k4]);
      let pb = vec4<f32>(KPs[(ty * 4u + 1u) * 17u + k4]);
      let pc = vec4<f32>(KPs[(ty * 4u + 2u) * 17u + k4]);
      let pd = vec4<f32>(KPs[(ty * 4u + 3u) * 17u + k4]);
      for (var e = 0u; e < 4u; e++) {
        let v = vec4<f32>(Vs[(k4 * 4u + e) * 16u + tx]);
        o[0] = fma(vec4<f32>(pa[e]), v, o[0]);
        o[1] = fma(vec4<f32>(pb[e]), v, o[1]);
        o[2] = fma(vec4<f32>(pc[e]), v, o[2]);
        o[3] = fma(vec4<f32>(pd[e]), v, o[3]);
      }
    }
  }
  for (var i = 0u; i < 4u; i++) {
    let r = q0 + ty * 4u + i;
    if (r < M) { out[r * ${H * 16}u + h * 16u + tx] = vec4<f16>(o[i] / l[i]); }
  }
}
`;
}

/** LayerNorm rows of a [rows, D] buffer (f32 or f16 in, f16 or f32 out), one 64-thread workgroup per row. */
export function layerNorm(D: number, eps: number, input: "f32" | "f16" = "f32", output: "f16" | "f32" = "f16"): string {
  const per = D / 4 / 64;
  if (!Number.isInteger(per)) throw new Error("D must be a multiple of 256");
  return `enable f16;
@group(0) @binding(0) var<storage, read> x: array<vec4<${input}>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> b: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> y: array<vec4<${output}>>;
var<workgroup> red: array<f32, 64>;
fn total(v: f32, t: u32) -> f32 {
  red[t] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (t < s) { red[t] += red[t + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let row = wg.x + wg.y * 65535u;
  let base = row * ${D / 4}u;
  var v: array<vec4<f32>, ${per}>;
  var s = 0.0;
  for (var i = 0u; i < ${per}u; i++) { v[i] = vec4<f32>(x[base + t + i * 64u]); s += v[i].x + v[i].y + v[i].z + v[i].w; }
  let mean = total(s, t) / ${D}.0;
  var q = 0.0;
  for (var i = 0u; i < ${per}u; i++) { v[i] -= vec4<f32>(mean); q += dot(v[i], v[i]); }
  let rstd = inverseSqrt(total(q, t) / ${D}.0 + ${eps});
  for (var i = 0u; i < ${per}u; i++) {
    let c = t + i * 64u;
    y[base + c] = vec4<${output}>(v[i] * rstd * vec4<f32>(w[c]) + vec4<f32>(b[c]));
  }
}
`;
}

/** h = LayerNorm(silu(gate) * value) over hidden features; gx f16 [rows, 2 * hidden] -> f16 [rows, hidden]. */
export function swigluNorm(hidden: number, eps: number): string {
  const per = hidden / 4 / 256;
  return `enable f16;
@group(0) @binding(0) var<storage, read> gx: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> b: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> y: array<vec4<f16>>;
var<workgroup> red: array<f32, 256>;
fn total(v: f32, t: u32) -> f32 {
  red[t] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) { red[t] += red[t + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let row = wg.x;
  let src = row * ${hidden / 2}u;   // vec4 index of the row's gate half
  var h: array<vec4<f32>, ${per}>;
  var s = 0.0;
  for (var i = 0u; i < ${per}u; i++) {
    let c = t + i * 256u;
    let g = vec4<f32>(gx[src + c]);
    let v = vec4<f32>(gx[src + ${hidden / 4}u + c]);
    h[i] = g / (vec4<f32>(1.0) + exp(-g)) * v;
    s += h[i].x + h[i].y + h[i].z + h[i].w;
  }
  let mean = total(s, t) / ${hidden}.0;
  var q = 0.0;
  for (var i = 0u; i < ${per}u; i++) { h[i] -= vec4<f32>(mean); q += dot(h[i], h[i]); }
  let rstd = inverseSqrt(total(q, t) / ${hidden}.0 + ${eps});
  for (var i = 0u; i < ${per}u; i++) {
    let c = t + i * 256u;
    y[row * ${hidden / 4}u + c] = vec4<f16>(h[i] * rstd * vec4<f32>(w[c]) + vec4<f32>(b[c]));
  }
}
`;
}

/**
 * Slot attention logits for one query set: attn[h][q][n] = softmax_q(l) * sigmoid(l + bias[q]),
 * l = q_h · k_h over head_dim 48 (scale is in the keys). kv f16 [N, 2D]; queries f32 [Q, D]. One thread per (n, h).
 */
export function slotLogits(N: number, D: number, Hp: number, Q: number): string {
  const hd = D / Hp;
  return `enable f16;
@group(0) @binding(0) var<storage, read> kv: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> qbias: array<f32>;
@group(0) @binding(3) var<storage, read_write> attn: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  let h = gid.y;
  if (n >= ${N}u) { return; }
  var k: array<vec4<f32>, ${hd / 4}>;
  for (var i = 0u; i < ${hd / 4}u; i++) { k[i] = vec4<f32>(kv[n * ${(2 * D) / 4}u + h * ${hd / 4}u + i]); }
  var lg: array<f32, ${Q}>;
  var mx = -1e30;
  for (var q = 0u; q < ${Q}u; q++) {
    var s = 0.0;
    for (var i = 0u; i < ${hd / 4}u; i++) { s += dot(k[i], queries[q * ${D / 4}u + h * ${hd / 4}u + i]); }
    lg[q] = s;
    mx = max(mx, s);
  }
  var den = 0.0;
  for (var q = 0u; q < ${Q}u; q++) { den += exp(lg[q] - mx); }
  for (var q = 0u; q < ${Q}u; q++) {
    let a = exp(lg[q] - mx) / den / (1.0 + exp(-(lg[q] + qbias[q])));
    attn[(h * ${Q}u + q) * ${N}u + n] = a;
  }
}
`;
}

/** z[q][h*hd + d] = sum_n attn[h][q][n] v[n][h*hd + d]; mass[h][q] = sum_n attn[h][q][n]. One thread per (d, q, h). */
export function slotGather(N: number, D: number, Hp: number, Q: number): string {
  const hd = D / Hp;
  return `enable f16;
@group(0) @binding(0) var<storage, read> kv: array<f16>;
@group(0) @binding(1) var<storage, read> attn: array<f32>;
@group(0) @binding(2) var<storage, read_write> z: array<f32>;
@group(0) @binding(3) var<storage, read_write> mass: array<f32>;
@compute @workgroup_size(${hd})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) d: u32) {
  let q = wg.x;
  let h = wg.y;
  let a = (h * ${Q}u + q) * ${N}u;
  var s = 0.0;
  var tot = 0.0;
  for (var n = 0u; n < ${N}u; n++) {
    let w = attn[a + n];
    s = fma(w, f32(kv[n * ${2 * D}u + ${D}u + h * ${hd}u + d]), s);
    tot += w;
  }
  z[q * ${D}u + h * ${hd}u + d] = s;
  if (d == 0u) { mass[h * ${Q}u + q] = tot; }
}
`;
}

const rowNormFn = (D: number, threads: number) => `
var<workgroup> red: array<f32, ${threads}>;
fn total(v: f32, t: u32) -> f32 {
  red[t] = v;
  workgroupBarrier();
  for (var s = ${threads / 2}u; s > 0u; s >>= 1u) {
    if (t < s) { red[t] += red[t + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
// LayerNorm of the ${D}-vector held as ${D / threads} values per thread (indices t + i * ${threads}).
fn norm(v: ptr<function, array<f32, ${D / threads}>>, w: u32, t: u32) {
  var s = 0.0;
  for (var i = 0u; i < ${D / threads}u; i++) { s += (*v)[i]; }
  let mean = total(s, t) / ${D}.0;
  var q = 0.0;
  for (var i = 0u; i < ${D / threads}u; i++) { (*v)[i] -= mean; q += (*v)[i] * (*v)[i]; }
  let rstd = inverseSqrt(total(q, t) / ${D}.0 + 1e-6);
  for (var i = 0u; i < ${D / threads}u; i++) {
    let c = t + i * ${threads}u;
    (*v)[i] = (*v)[i] * rstd * f32(nw[w * ${D}u + c]) + f32(nb[w * ${D}u + c]);
  }
}
`;

/**
 * Between the two slot attentions: z_coarse = LN_trunk(z), z_parent = z_coarse repeated per child,
 * q_leaf = LN_child(z_parent + q_child). Norm weights packed as [trunk, child_q, leaf] in nw / nb.
 * One workgroup per leaf query.
 */
export function poolMid(D: number, C: number, Q: number): string {
  const threads = 256;
  return `enable f16;
@group(0) @binding(0) var<storage, read> z: array<f32>;          // [C, D]
@group(0) @binding(1) var<storage, read> qchild: array<f16>;     // [Q, D]
@group(0) @binding(2) var<storage, read> nw: array<f16>;
@group(0) @binding(3) var<storage, read> nb: array<f16>;
@group(0) @binding(4) var<storage, read_write> parent: array<f32>; // [Q, D]
@group(0) @binding(5) var<storage, read_write> qleaf: array<f32>;  // [Q, D]
${rowNormFn(D, threads)}
@compute @workgroup_size(${threads})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let q = wg.x;
  let c = q / ${Q / C}u;
  var v: array<f32, ${D / threads}>;
  for (var i = 0u; i < ${D / threads}u; i++) { v[i] = z[c * ${D}u + t + i * ${threads}u]; }
  norm(&v, 0u, t);
  for (var i = 0u; i < ${D / threads}u; i++) {
    let j = t + i * ${threads}u;
    parent[q * ${D}u + j] = v[i];
    v[i] += f32(qchild[q * ${D}u + j]);
  }
  norm(&v, 1u, t);
  for (var i = 0u; i < ${D / threads}u; i++) { qleaf[q * ${D}u + t + i * ${threads}u] = v[i]; }
}
`;
}

/**
 * Leaf features: z_tree = LN_leaf(parent + z_child) * mass / (mass + 4), mass = mean over heads of the patch mass,
 * leaf[q][e] = z_tree[q] · leafW[q][:, e] + leafB[q][e]. One workgroup per leaf query; out f32 [Q * E].
 */
export function poolTail(D: number, Hp: number, Q: number, E: number): string {
  const threads = 256;
  const parts = threads / E;
  return `enable f16;
@group(0) @binding(0) var<storage, read> parent: array<f32>;
@group(0) @binding(1) var<storage, read> zc: array<f32>;
@group(0) @binding(2) var<storage, read> mass: array<f32>;      // [Hp, Q]
@group(0) @binding(3) var<storage, read> nw: array<f16>;
@group(0) @binding(4) var<storage, read> nb: array<f16>;
@group(0) @binding(5) var<storage, read> lw: array<f16>;        // [Q, D, E]
@group(0) @binding(6) var<storage, read> lb: array<f16>;        // [Q, E]
@group(0) @binding(7) var<storage, read_write> leaf: array<f32>; // [Q * E]
${rowNormFn(D, threads)}
var<workgroup> zt: array<f32, ${D}>;
var<workgroup> part: array<f32, ${threads}>;
@compute @workgroup_size(${threads})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let q = wg.x;
  var v: array<f32, ${D / threads}>;
  for (var i = 0u; i < ${D / threads}u; i++) {
    let j = q * ${D}u + t + i * ${threads}u;
    v[i] = parent[j] + zc[j];
  }
  norm(&v, 2u, t);
  var m = 0.0;
  for (var h = 0u; h < ${Hp}u; h++) { m += mass[h * ${Q}u + q]; }
  m /= ${Hp}.0;
  let gate = m / (m + 4.0);
  for (var i = 0u; i < ${D / threads}u; i++) { zt[t + i * ${threads}u] = v[i] * gate; }
  workgroupBarrier();
  let e = t % ${E}u;
  let p = t / ${E}u;
  var s = 0.0;
  for (var d = p * ${D / parts}u; d < (p + 1u) * ${D / parts}u; d++) {
    s = fma(zt[d], f32(lw[(q * ${D}u + d) * ${E}u + e]), s);
  }
  part[t] = s;
  workgroupBarrier();
  if (p == 0u) {
    var tot = f32(lb[q * ${E}u + e]);
    for (var i = 0u; i < ${parts}u; i++) { tot += part[i * ${E}u + e]; }
    leaf[q * ${E}u + e] = tot;
  }
}
`;
}

/** probs[r] = sigmoid(W[r] · x + b[r]) for rows [row0, row0 + rows) of one weight chunk; 8 rows per workgroup. */
export function headGemv(rows: number, K: number, row0: number): string {
  const kv = K / 4;
  const per = kv / 32;
  if (!Number.isInteger(per)) throw new Error("K / 4 must be a multiple of 32");
  return `enable f16;
@group(0) @binding(0) var<storage, read> W: array<vec4<f16>>;    // this chunk, [rows, K]
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> b: array<f16>;           // all rows
@group(0) @binding(3) var<storage, read_write> probs: array<f32>; // all rows
var<workgroup> xs: array<vec4<f32>, ${kv}>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  for (var i = t; i < ${kv}u; i += 256u) { xs[i] = x[i]; }
  workgroupBarrier();
  let lane = t % 32u;
  let r = (wg.x + wg.y * 65535u) * 8u + t / 32u;
  var s = 0.0;
  if (r < ${rows}u) {
    for (var i = 0u; i < ${per}u; i++) {
      let c = lane + i * 32u;
      s += dot(vec4<f32>(W[r * ${kv}u + c]), xs[c]);
    }
  }
  red[t] = s;
  workgroupBarrier();
  for (var st = 16u; st > 0u; st >>= 1u) {
    if (lane < st) { red[t] += red[t + st]; }
    workgroupBarrier();
  }
  if (lane == 0u && r < ${rows}u) {
    let g = ${row0}u + r;
    probs[g] = 1.0 / (1.0 + exp(-(red[t] + f32(b[g]))));
  }
}
`;
}

/** Weight reconstruction: f16[i] = zigzag^-1(planes) * step[row]; planes are bytes packed in u32 words. */
export function dequant(cols: number, planes: number): string {
  const decls = Array.from({ length: planes }, (_, i) => `@group(0) @binding(${i}) var<storage, read> p${i}: array<u32>;`).join("\n");
  let z = "";
  for (let i = 0; i < planes; i++) z += ` | (((p${i}[w] >> (8u * e)) & 0xffu) << ${8 * i}u)`;
  return `enable f16;
${decls}
@group(0) @binding(${planes}) var<storage, read> step: array<f16>;
@group(0) @binding(${planes + 1}) var<storage, read_write> out: array<vec4<f16>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let w = gid.x + gid.y * nwg.x * 256u;   // one u32 word = 4 weights
  if (w >= arrayLength(&out)) { return; }
  let s = f32(step[(w * 4u) / ${cols}u]);
  var v: vec4<f32>;
  for (var e = 0u; e < 4u; e++) {
    let z = 0u${z};
    let code = i32(z >> 1u) ^ -i32(z & 1u);
    v[e] = f32(code) * s;
  }
  out[w] = vec4<f16>(v);
}
`;
}

/** As dequant, but writes the matrix transposed: out f16 [cols, rows] (for matmul weights). One thread per 4 rows. */
export function dequantT(rows: number, cols: number, planes: number): string {
  if (rows % 4) throw new Error("rows must be a multiple of 4");
  const decls = Array.from({ length: planes }, (_, i) => `@group(0) @binding(${i}) var<storage, read> p${i}: array<u32>;`).join("\n");
  let z = "";
  for (let i = 0; i < planes; i++) z += ` | (((p${i}[j / 4u] >> (8u * (j % 4u))) & 0xffu) << ${8 * i}u)`;
  return `enable f16;
${decls}
@group(0) @binding(${planes}) var<storage, read> step: array<f16>;
@group(0) @binding(${planes + 1}) var<storage, read_write> out: array<vec4<f16>>;  // [cols, rows / 4]
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let w = gid.x + gid.y * nwg.x * 256u;
  if (w >= ${(rows * cols) / 4}u) { return; }
  let c = w / ${rows / 4}u;
  let r0 = (w % ${rows / 4}u) * 4u;
  var v: vec4<f32>;
  for (var e = 0u; e < 4u; e++) {
    let j = (r0 + e) * ${cols}u + c;
    let z = 0u${z};
    v[e] = f32(i32(z >> 1u) ^ -i32(z & 1u)) * f32(step[r0 + e]);
  }
  out[w] = vec4<f16>(v);
}
`;
}
