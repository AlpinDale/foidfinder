/**
 * Slot attention logits for one query set: attn[h][q][n] = softmax_q(l) * sigmoid(l + bias[q]),
 * l = q_h · k_h over head_dim (scale is in the keys). kv f16 [N, 2D]; queries f32 [Q, D]. One thread per (n, h).
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
