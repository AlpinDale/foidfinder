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
