import { at } from "./at";

// The model card's preprocessing on the GPU, bit-exact with foidfinder's Pillow path: composite onto white,
// letterboxed BICUBIC resize to SIZE x SIZE on white, written straight into the patch-embedding layout
// (one row per 14 x 14 patch, (y, x, channel) order, f16, padded to the matmul's K).

export interface Coeffs {
  kk: Int32Array; // [out, ksize] fixed-point taps (22 fraction bits)
  bounds: Int32Array; // [out, 2] (first tap, tap count)
  ksize: number;
}

/** Pillow's precompute_coeffs + normalize_coeffs_8bpc for BICUBIC (a = -0.5), with the same float64 operations. */
export function bicubicCoeffs(inSize: number, outSize: number): Coeffs {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1.0);
  const support = 2.0 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const kk = new Int32Array(outSize * ksize);
  const bounds = new Int32Array(outSize * 2);
  const w = new Float64Array(ksize);
  for (let i = 0; i < outSize; i++) {
    const center = (i + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    let ww = 0.0;
    for (let t = 0; t < ksize; t++) {
      let v = 0.0;
      if (t < xmax) {
        const x = Math.abs((t + xmin - center + 0.5) * (1.0 / filterscale));
        v = x < 1.0 ? (1.5 * x - 2.5) * x * x + 1 : x < 2.0 ? (((x - 5) * x + 8) * x - 4) * -0.5 : 0.0;
      }
      w[t] = v;
      ww += v;
    }
    for (let t = 0; t < ksize; t++) {
      const v = ww !== 0.0 ? at(w, t) / ww : at(w, t);
      kk[i * ksize + t] = Math.trunc(v < 0 ? -0.5 + v * (1 << 22) : 0.5 + v * (1 << 22));
    }
    bounds[i * 2] = xmin;
    bounds[i * 2 + 1] = xmax;
  }
  return { kk, bounds, ksize };
}

function identity(n: number): Coeffs {
  const bounds = new Int32Array(n * 2);
  for (let i = 0; i < n; i++) {
    bounds[i * 2] = i;
    bounds[i * 2 + 1] = 1;
  }
  return { kk: new Int32Array(n).fill(1 << 22), bounds, ksize: 1 };
}

// Pillow's alpha_composite of an RGBA pixel over opaque white, in its integer arithmetic.
const COMPOSITE = `
fn over_white(p: vec4<u32>) -> vec3<i32> {
  if (p.a == 0u) { return vec3<i32>(255); }
  let t = 128u * (p.rgb * p.a + vec3<u32>(255u * (255u - p.a))) + vec3<u32>(16384u);
  return vec3<i32>((((t >> vec3<u32>(8u)) + t) >> vec3<u32>(8u)) >> vec3<u32>(7u));
}`;

/** One Pillow resampling pass along `axis` (0: width, 1: height) into the region of an out_w x out_h canvas. */
function resampleShader(axis: 0 | 1, fromTexture: boolean, toPatches: boolean, patch: number, grid: number, kpad: number) {
  const src = fromTexture
    ? `@group(0) @binding(0) var src: texture_2d<f32>;
${COMPOSITE}
fn fetch(x: i32, y: i32) -> vec3<i32> {
  return over_white(vec4<u32>(round(textureLoad(src, vec2<i32>(x, y), 0) * 255.0)));
}`
    : `@group(0) @binding(0) var<storage, read> src: array<u32>;
fn fetch(x: i32, y: i32) -> vec3<i32> {
  let v = src[u32(y) * g.src_w + u32(x)];
  return vec3<i32>(i32(v & 255u), i32((v >> 8u) & 255u), i32((v >> 16u) & 255u));
}`;
  const dst = toPatches
    ? `@group(0) @binding(4) var<storage, read_write> dst: array<f16>;
fn put(x: u32, y: u32, v: vec3<i32>) {
  let row = (y / ${patch}u) * ${grid}u + x / ${patch}u;
  let col = ((y % ${patch}u) * ${patch}u + x % ${patch}u) * 3u;
  let o = row * ${kpad}u + col;
  dst[o] = f16(v.r);
  dst[o + 1u] = f16(v.g);
  dst[o + 2u] = f16(v.b);
}`
    : `@group(0) @binding(4) var<storage, read_write> dst: array<u32>;
fn put(x: u32, y: u32, v: vec3<i32>) {
  dst[y * g.out_w + x] = u32(v.r) | (u32(v.g) << 8u) | (u32(v.b) << 16u);
}`;
  const tap =
    axis === 0 ? "fetch(first + i, i32(g.row0) + yy)" : "fetch(xx, i32(g.row0) + first + i)";
  return `enable f16;
struct Geom { out_w: u32, out_h: u32, top: i32, left: i32, h: i32, w: i32, ksize: u32, src_w: u32, row0: u32 };
@group(0) @binding(1) var<storage, read> kk: array<i32>;
@group(0) @binding(2) var<storage, read> bounds: array<i32>;
@group(0) @binding(3) var<uniform> g: Geom;
${src}
${dst}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= g.out_w || id.y >= g.out_h) { return; }
  let yy = i32(id.y) - g.top;
  let xx = i32(id.x) - g.left;
  var v = vec3<i32>(255);
  if (yy >= 0 && yy < g.h && xx >= 0 && xx < g.w) {
    let o = ${axis === 0 ? "xx" : "yy"};
    let first = bounds[o * 2];
    let taps = bounds[o * 2 + 1];
    var ss = vec3<i32>(1 << 21);
    for (var i = 0; i < taps; i++) { ss += ${tap} * kk[o * i32(g.ksize) + i]; }
    v = clamp(ss >> vec3<u32>(22u), vec3<i32>(0), vec3<i32>(255));
  }
  put(id.x, id.y, v);
}
`;
}

interface Pass {
  axis: 0 | 1;
  coeffs: Coeffs;
  outW: number;
  outH: number;
  region: [top: number, left: number, h: number, w: number];
  srcW: number;
  row0: number;
}

export class Preprocessor {
  private pipelines = new Map<string, GPUComputePipeline>();

  constructor(
    private device: GPUDevice,
    private size: number,
    private patch: number,
    private kpad: number,
    /** f16 [patches, kpad], columns past patch * patch * 3 stay zero. */
    private patches: GPUBuffer,
  ) {}

  private pipeline(axis: 0 | 1, fromTexture: boolean, toPatches: boolean): GPUComputePipeline {
    const key = `${axis}${+fromTexture}${+toPatches}`;
    let p = this.pipelines.get(key);
    if (!p) {
      const code = resampleShader(axis, fromTexture, toPatches, this.patch, this.size / this.patch, this.kpad);
      p = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  /** The passes foidfinder's resize_pad runs for a w x h image (Pillow's pass order and row trimming). */
  plan(w: number, h: number): Pass[] {
    const s = this.size;
    if (w === s && h === s) {
      return [{ axis: 0, coeffs: identity(s), outW: s, outH: s, region: [0, 0, s, s], srcW: w, row0: 0 }];
    }
    const r = Math.min(s / w, s / h);
    const nw = Math.max(1, Math.trunc(w * r));
    const nh = Math.max(1, Math.trunc(h * r));
    const top = Math.trunc((s - nh) / 2);
    const left = Math.trunc((s - nw) / 2);
    const ch = bicubicCoeffs(w, nw);
    const cv = bicubicCoeffs(h, nh);
    if (h - nh > 0 && h - nh > (w - nw) * 2) {
      return [
        { axis: 1, coeffs: cv, outW: w, outH: nh, region: [0, 0, nh, w], srcW: w, row0: 0 },
        { axis: 0, coeffs: ch, outW: s, outH: s, region: [top, left, nh, nw], srcW: w, row0: 0 },
      ];
    }
    const row0 = at(cv.bounds, 0);
    const last = (nh - 1) * 2;
    const rows = at(cv.bounds, last) + at(cv.bounds, last + 1) - row0;
    const shifted = { ...cv, bounds: cv.bounds.map((v, i) => (i % 2 === 0 ? v - row0 : v)) };
    return [
      { axis: 0, coeffs: ch, outW: nw, outH: rows, region: [0, 0, rows, nw], srcW: w, row0 },
      { axis: 1, coeffs: shifted, outW: s, outH: s, region: [top, left, nh, nw], srcW: nw, row0: 0 },
    ];
  }

  /** Records the passes for `image` into `encoder`; returns buffers to destroy after submission. */
  encode(encoder: GPUCommandEncoder, image: ImageBitmap): { destroy: () => void } {
    const d = this.device;
    const texture = d.createTexture({
      size: [image.width, image.height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    d.queue.copyExternalImageToTexture({ source: image }, { texture, premultipliedAlpha: false }, [image.width, image.height]);
    const owned: (GPUBuffer | GPUTexture)[] = [texture];
    const storage = (data: Int32Array) => {
      const b = d.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(b, 0, data);
      owned.push(b);
      return b;
    };
    const passes = this.plan(image.width, image.height);
    let src: GPUTexture | GPUBuffer = texture;
    passes.forEach((p, n) => {
      const last = n === passes.length - 1;
      const dst = last
        ? this.patches
        : d.createBuffer({ size: p.outW * p.outH * 4, usage: GPUBufferUsage.STORAGE });
      if (!last) owned.push(dst as GPUBuffer);
      const geom = new Int32Array([p.outW, p.outH, ...p.region, p.coeffs.ksize, p.srcW, p.row0]);
      const ub = d.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(ub, 0, geom);
      owned.push(ub);
      const pipeline = this.pipeline(p.axis, n === 0, last);
      const bind = d.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: src instanceof GPUTexture ? src.createView() : { buffer: src } },
          { binding: 1, resource: { buffer: storage(p.coeffs.kk) } },
          { binding: 2, resource: { buffer: storage(p.coeffs.bounds) } },
          { binding: 3, resource: { buffer: ub } },
          { binding: 4, resource: { buffer: dst } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(p.outW / 16), Math.ceil(p.outH / 16));
      pass.end();
      src = dst;
    });
    return { destroy: () => owned.forEach((o) => o.destroy()) };
  }
}
