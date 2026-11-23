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

/**
 * C[M, N] = A[M, K] · WT[K, N] (A f16 row-major, WT the weight transposed), then the epilogue.
 *
 * Products accumulate in f16 for `flushK` k steps, then into f32 accumulators to halve workgroup-memory
 * traffic while staying within fp16/f32-accum accuracy. K must divide BK and N must divide BN.
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
  if (flushK % 4 || BK % flushK) throw new Error("flushK must divide BK in steps of 4");

  const outDecl =
    ep.kind === "bias" || ep.kind === "qkv"
      ? "var<storage, read_write> C: array<vec4<f16>>;"
      : "var<storage, read_write> C: array<vec4<f32>>;";
  const extra = ep.kind === "qkv" ? "@group(0) @binding(4) var<storage, read> cs: array<vec2<f32>>;" : "";
  const biasAt = ep.kind === "patch" ? "bias[(r * N + c) / 4u]" : "bias[c / 4u]";

  let decls = "";
  let tileInit = "";
  let tileFlush = "";
  for (let i = 0; i < TM; i++) {
    for (let j = 0; j < GB; j++) {
      decls += `var acc${i}_${j} = vec4<f32>(0.0);\n  `;
      tileInit += `var h${i}_${j} = vec4<f16>(0.0);\n    `;
      tileFlush += `acc${i}_${j} += vec4<f32>(h${i}_${j});\n    `;
    }
  }

  let inner = "";
  for (let k4 = 0; k4 < BKV; k4++) {
    if (k4 > 0 && (k4 * 4) % flushK === 0) inner += `${tileFlush}${tileInit.replaceAll("var ", "")}`;
    inner += "{\n      ";
    for (let i = 0; i < TM; i++) inner += `let a${i} = As[abase + ${i * BKV + k4}u];\n      `;
    for (let e = 0; e < 4; e++) {
      for (let j = 0; j < GB; j++) inner += `let b${e}_${j} = Bs[bbase + ${(k4 * 4 + e) * BNV + j}u];\n      `;
      for (let i = 0; i < TM; i++) {
        for (let j = 0; j < GB; j++) {
          inner += `h${i}_${j} = fma(vec4<f16>(a${i}[${e}]), b${e}_${j}, h${i}_${j});\n      `;
        }
      }
    }
    inner += "}\n      ";
  }

  let store = "";
  for (let i = 0; i < TM; i++) {
    store += `{ let r = m0 + ty * ${TM}u + ${i}u; if (r < M) {\n`;
    for (let j = 0; j < GB; j++) {
      store += `  { let c = n0 + (tx * ${GB}u + ${j}u) * 4u; let v = acc${i}_${j} + vec4<f32>(${biasAt});\n${epilogue(ep)} }\n`;
    }
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
