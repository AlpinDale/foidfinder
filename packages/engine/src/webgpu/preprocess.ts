import { planResizePad } from "./bicubic";

// Bit-exact GPU image preprocessing: composite RGBA onto white, letterboxed Pillow BICUBIC resize to
// size x size on white, written directly into the patch-embedding layout (f16 [patches, kpad]).

const COMPOSITE = `
fn over_white(p: vec4<u32>) -> vec3<i32> {
  if (p.a == 0u) { return vec3<i32>(255); }
  let t = 128u * (p.rgb * p.a + vec3<u32>(255u * (255u - p.a))) + vec3<u32>(16384u);
  return vec3<i32>((((t >> vec3<u32>(8u)) + t) >> vec3<u32>(8u)) >> vec3<u32>(7u));
}`;

function resampleShader(axis: 0 | 1, fromTexture: boolean, toPatches: boolean, patch: number, grid: number, kpad: number): string {
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
  const tap = axis === 0 ? "fetch(first + i, i32(g.row0) + yy)" : "fetch(xx, i32(g.row0) + first + i)";
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

export class PatchPreprocessor {
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

  /** Records the preprocessing passes for `image` into `encoder`; returns a callback to release temporary buffers. */
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
    const passes = planResizePad(image.width, image.height, this.size);
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
