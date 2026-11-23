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
