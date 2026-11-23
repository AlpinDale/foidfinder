import type { DeviceFeatures } from "../../../webgpu/device";

const header = (f: DeviceFeatures) => `enable f16;${f.subgroups16 ? "\nenable subgroups;" : ""}\n`;

/** Reduction of one value per thread over a row's 16 consecutive lanes (lane = tx in 0..15). */
function rowReduce(f: DeviceFeatures, op: "max" | "add"): string {
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
export function attention(M: number, H: number, f: DeviceFeatures): string {
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
