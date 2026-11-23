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
