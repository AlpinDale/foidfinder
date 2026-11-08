// Haruka Tagger on WebGPU: the forward of foidfinder/backends/mlx/haruka.py over the compressed checkpoint.

import { at } from "./at";
import { Preprocessor } from "./preprocess";
import * as K from "./wgsl";
import { gunzip, halves, parseSafetensors, toFloat32, type Tensor } from "./weights";

export interface ModelConfig {
  model: {
    img_size: number;
    patch_size: number;
    dim: number;
    heads: number;
    depth: number;
    mlp_hidden: number;
    pool_heads: number;
    coarse_queries: number;
    leaf_queries: number;
    leaf_dim: number;
    rope_ref_grid: number;
  };
  quantization: { format: string; layers: Record<string, { shape: [number, number]; planes: number }> };
  tags: string[];
  categories: string[];
  thresholds: Record<string, number>;
}

const EPS = 1e-6;
const TILE: K.Tile = { BM: 32, BN: 64, BK: 16, TM: 4, TN: 4 };
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

interface Op {
  label: string;
  pipeline: GPUComputePipeline;
  bind: GPUBindGroup;
  x: number;
  y: number;
}

export interface RunResult {
  probs: Float32Array;
  /** GPU time per kernel label, summed over repeats (profile runs only). */
  profile?: Map<string, number>;
}

/** WebGPU adapter + device with what the engine needs; throws a readable error when unsupported. */
export async function requestDevice(): Promise<{ device: GPUDevice; features: K.Features; description: string }> {
  if (!("gpu" in navigator)) throw new Error("This browser has no WebGPU, which foidfinder needs to run the model");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU is turned on, but the browser offers no GPU to it");
  if (!adapter.features.has("shader-f16")) {
    throw new Error("This GPU or browser lacks 16-bit floats in WebGPU shaders (shader-f16), which the model needs");
  }
  const info = adapter.info as GPUAdapterInfo & { subgroupMinSize?: number };
  const subgroups16 = adapter.features.has("subgroups") && (info.subgroupMinSize ?? 0) >= 16;
  const required: GPUFeatureName[] = ["shader-f16"];
  if (subgroups16) required.push("subgroups" as GPUFeatureName);
  if (adapter.features.has("timestamp-query")) required.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: required,
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxStorageBuffersPerShaderStage: Math.max(8, Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage)),
      maxTextureDimension2D: adapter.limits.maxTextureDimension2D,
    },
  });
  const description = [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "WebGPU";
  return { device, features: { subgroups16 }, description };
}

function floatToHalf(v: number): number {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, v, true);
  const x = view.getUint32(0, true);
  const sign = (x >> 16) & 0x8000;
  const e = ((x >> 23) & 0xff) - 127 + 15;
  const m = x & 0x7fffff;
  if (e <= 0) return sign; // tiny values do not occur here
  if (e >= 31) return sign | 0x7c00;
  let h = sign | (e << 10) | (m >> 13);
  const rest = m & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && h & 1)) h += 1;
  return h;
}

export class Haruka {
  readonly config: ModelConfig;
  readonly tagCount: number;
  private device: GPUDevice;
  private features: K.Features;
  private pipelines = new Map<string, Promise<GPUComputePipeline>>();
  private buffers: GPUBuffer[] = [];
  private ops: Op[] = [];
  private pre!: Preprocessor;
  private patches!: GPUBuffer;
  private resid!: GPUBuffer;
  private cls!: GPUBuffer;
  private probs!: GPUBuffer;
  private readback!: GPUBuffer;
  private kpad: number;
  private constructor(device: GPUDevice, features: K.Features, config: ModelConfig) {
    this.device = device;
    this.features = features;
    this.config = config;
    this.tagCount = config.tags.length;
    const { patch_size: p } = config.model;
    this.kpad = Math.ceil((p * p * 3) / TILE.BK) * TILE.BK;
  }

  static async load(
    device: GPUDevice,
    features: K.Features,
    config: ModelConfig,
    weights: ArrayBuffer,
    onProgress?: (done: number, total: number) => void,
  ): Promise<Haruka> {
    if (config.quantization?.format !== "foidfinder-haruka-rowstep1") throw new Error("Unknown model format");
    const m = new Haruka(device, features, config);
    await m.build(parseSafetensors(weights), onProgress);
    return m;
  }

  // ------------------------------------------------------------------ buffers

  private buffer(size: number, usage = STORAGE): GPUBuffer {
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
    this.buffers.push(b);
    return b;
  }

  private upload(data: ArrayBufferView): GPUBuffer {
    const b = this.buffer(data.byteLength);
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    if (bytes.byteLength % 4) {
      const padded = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
      padded.set(bytes);
      this.device.queue.writeBuffer(b, 0, padded);
    } else this.device.queue.writeBuffer(b, 0, bytes);
    return b;
  }

  private pipeline(code: string): Promise<GPUComputePipeline> {
    let p = this.pipelines.get(code);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      this.pipelines.set(code, p);
    }
    return p;
  }

  private async op(label: string, code: string, buffers: GPUBuffer[], x: number, y = 1, offsets?: number[]) {
    const pipeline = await this.pipeline(code);
    const bind = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({
        binding,
        resource: { buffer, offset: offsets?.[binding] ?? 0 },
      })),
    });
    return { label, pipeline, bind, x, y } satisfies Op;
  }

  /** Dispatches a one-off kernel (weight reconstruction) and submits it. */
  private async runOnce(code: string, buffers: GPUBuffer[], workgroups: number, offsets?: number[]) {
    const x = Math.min(workgroups, 32768);
    const o = await this.op("once", code, buffers, x, Math.ceil(workgroups / x), offsets);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(o.pipeline);
    pass.setBindGroup(0, o.bind);
    pass.dispatchWorkgroups(o.x, o.y);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  // ------------------------------------------------------------------ weights

  private async build(t: Map<string, Tensor>, onProgress?: (done: number, total: number) => void) {
    const cfg = this.config.model;
    const D = cfg.dim;
    const H = cfg.heads;
    const HID = cfg.mlp_hidden;
    const grid = cfg.img_size / cfg.patch_size;
    const N = grid * grid + 1;
    const Hp = cfg.pool_heads;
    const C = cfg.coarse_queries;
    const Q = cfg.leaf_queries;
    const E = cfg.leaf_dim;
    const T = this.tagCount;
    const f = this.features;
    const layers = (name: string) => {
      const l = this.config.quantization.layers[name];
      if (!l) throw new Error(`missing compressed matrix ${name}`);
      return l;
    };
    const get = (name: string) => {
      const v = t.get(name);
      if (!v) throw new Error(`missing tensor ${name}`);
      return v;
    };
    const f16 = (name: string) => this.upload(halves(get(name)));
    const f32 = (name: string) => this.upload(toFloat32(get(name)));

    // Compressed matrices: gunzip the byte planes, then rebuild f16 weights on the GPU.
    const names = Object.keys(this.config.quantization.layers);
    let done = 0;
    const planesOf = async (name: string) =>
      Promise.all(Array.from({ length: layers(name).planes }, (_, i) => gunzip(get(`${name}.plane${i}`).bytes)));
    const weightT = new Map<string, GPUBuffer>();
    const decodeT = async (name: string) => {
      const [rows, cols] = layers(name).shape;
      const planes = (await planesOf(name)).map((p) => this.upload(p));
      const out = this.buffer(rows * cols * 2);
      await this.runOnce(K.dequantT(rows, cols, planes.length), [...planes, f16(`${name}.step`), out], Math.ceil((rows * cols) / 4 / 256));
      planes.forEach((p) => p.destroy());
      weightT.set(name, out);
      onProgress?.(++done, names.length);
    };
    for (const name of names) if (name !== "head.weight") await decodeT(name);
    const weight = (name: string) => {
      const w = weightT.get(name);
      if (!w) throw new Error(`missing compressed matrix ${name}`);
      return w;
    };

    // Classifier: row-major, split into chunks that fit one storage binding.
    const [hRows, hCols] = layers("head.weight").shape;
    const limit = this.device.limits.maxStorageBufferBindingSize;
    const chunkRows = Math.min(hRows, Math.floor(limit / (hCols * 2) / 128) * 128);
    const headPlanes = await planesOf("head.weight");
    const headStep = f16("head.weight.step");
    const headChunks: { buf: GPUBuffer; row0: number; rows: number }[] = [];
    for (let row0 = 0; row0 < hRows; row0 += chunkRows) {
      const rows = Math.min(chunkRows, hRows - row0);
      const planes = headPlanes.map((p) => this.upload(p.subarray(row0 * hCols, (row0 + rows) * hCols)));
      const buf = this.buffer(rows * hCols * 2);
      const offsets = [...planes.map(() => 0), row0 * 2, 0];
      await this.runOnce(K.dequant(hCols, planes.length), [...planes, headStep, buf], Math.ceil((rows * hCols) / 4 / 256), offsets);
      planes.forEach((p) => p.destroy());
      headChunks.push({ buf, row0, rows });
    }
    onProgress?.(++done, names.length);

    // Patch embedding: [D, p*p*3] -> transposed and zero-padded to [kpad, D].
    const pw = halves(get("patch_w"));
    const pk = cfg.patch_size * cfg.patch_size * 3;
    const pwT = new Uint16Array(this.kpad * D);
    pw.forEach((v, i) => (pwT[(i % pk) * D + Math.floor(i / pk)] = v));
    const patchWT = this.upload(pwT);

    // RoPE (cos, sin) per token and band, as rope_tables(): axial, positions rescaled to the pretrain grid.
    const hd = D / H;
    const nb = hd / 4;
    const cs = new Float32Array(N * (hd / 2) * 2);
    for (let n = 0; n < N; n++) {
      for (let b = 0; b < hd / 2; b++) {
        let ang = 0;
        if (n > 0) {
          const gy = Math.floor((n - 1) / grid);
          const gx = (n - 1) % grid;
          const band = Math.fround(1 / Math.fround(10000 ** Math.fround((b % nb) / nb)));
          const pos = Math.fround(Math.fround((b < nb ? gy : gx) / grid) * cfg.rope_ref_grid);
          ang = Math.fround(pos * band);
        }
        cs[(n * (hd / 2) + b) * 2] = Math.cos(ang);
        cs[(n * (hd / 2) + b) * 2 + 1] = Math.sin(ang);
      }
    }
    const csBuf = this.upload(cs);

    // Activations.
    this.patches = this.buffer((N - 1) * this.kpad * 2);
    this.resid = this.buffer(N * D * 4);
    this.cls = f32("cls");
    const xn = this.buffer(N * D * 2);
    const qkvh = this.buffer(N * 3 * D * 2);
    const att = this.buffer(N * D * 2);
    const gx = this.buffer(N * 2 * HID * 2);
    const hm = this.buffer(N * HID * 2);
    const xf = this.buffer(N * D * 4);
    const kv = this.buffer(N * 2 * D * 2);
    const attn1 = this.buffer(Hp * C * N * 4);
    const z1 = this.buffer(C * D * 4);
    const mass1 = this.buffer(Hp * C * 4);
    const parent = this.buffer(Q * D * 4);
    const qleaf = this.buffer(Q * D * 4);
    const attn2 = this.buffer(Hp * Q * N * 4);
    const z2 = this.buffer(Q * D * 4);
    const mass2 = this.buffer(Hp * Q * 4);
    const leaf = this.buffer(Q * E * 4);
    this.probs = this.buffer(T * 4);
    this.readback = this.buffer(T * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    this.pre = new Preprocessor(this.device, cfg.img_size, cfg.patch_size, this.kpad, this.patches);

    const rowsM = Math.ceil(N / TILE.BM);
    const ops: Promise<Op>[] = [];
    ops.push(
      this.op("patch", K.matmul(N - 1, D, this.kpad, TILE, { kind: "patch" }), [this.patches, patchWT, f16("patch_b"), this.resid], D / TILE.BN, Math.ceil((N - 1) / TILE.BM)),
    );
    const ln = K.layerNorm(D, EPS);
    for (let i = 0; i < cfg.depth; i++) {
      const b = `blocks.${i}.`;
      ops.push(this.op("ln", ln, [this.resid, f16(b + "norm1.weight"), f16(b + "norm1.bias"), xn], N));
      ops.push(
        this.op("qkv", K.matmul(N, 3 * D, D, TILE, { kind: "qkv", heads: H, dim: D }), [xn, weight(b + "qkv.weight"), f16(b + "qkv.bias"), qkvh, csBuf], (3 * D) / TILE.BN, rowsM),
      );
      ops.push(this.op("attention", K.attention(N, H, f), [qkvh, att], Math.ceil(N / 64), H));
      ops.push(
        this.op("proj", K.matmul(N, D, D, TILE, { kind: "residual" }), [att, weight(b + "proj.weight"), f16(b + "proj.bias"), this.resid], D / TILE.BN, rowsM),
      );
      ops.push(this.op("ln", ln, [this.resid, f16(b + "norm2.weight"), f16(b + "norm2.bias"), xn], N));
      ops.push(
        this.op("fc1", K.matmul(N, 2 * HID, D, TILE, { kind: "bias" }), [xn, weight(b + "fc1.weight"), f16(b + "fc1.bias"), gx], (2 * HID) / TILE.BN, rowsM),
      );
      ops.push(this.op("swiglu", K.swigluNorm(HID, EPS), [gx, f16(b + "mlp_norm.weight"), f16(b + "mlp_norm.bias"), hm], N));
      ops.push(
        this.op("fc2", K.matmul(N, D, HID, TILE, { kind: "residual" }), [hm, weight(b + "fc2.weight"), f16(b + "fc2.bias"), this.resid], D / TILE.BN, rowsM),
      );
    }
    ops.push(this.op("ln", K.layerNorm(D, EPS, "f32", "f32"), [this.resid, f16("norm.weight"), f16("norm.bias"), xf], N));
    ops.push(this.op("ln", K.layerNorm(D, EPS, "f32", "f16"), [xf, f16("pool.norm.weight"), f16("pool.norm.bias"), xn], N));
    ops.push(
      this.op("pool", K.matmul(N, 2 * D, D, TILE, { kind: "bias" }), [xn, weight("pool.kv.weight"), f16("pool.kv.bias"), kv], (2 * D) / TILE.BN, rowsM),
    );
    const norms = (kind: "weight" | "bias") => {
      const parts = ["trunk_norm", "child_q_norm", "leaf_norm"].map((n) => halves(get(`pool.${n}.${kind}`)));
      const all = new Uint16Array(parts.length * D);
      parts.forEach((p, i) => all.set(p, i * D));
      return this.upload(all);
    };
    const nw = norms("weight");
    const nbias = norms("bias");
    ops.push(this.op("pool", K.slotLogits(N, D, Hp, C), [kv, f32("pool.q_coarse"), f32("pool.q_coarse_bias"), attn1], Math.ceil(N / 64), Hp));
    ops.push(this.op("pool", K.slotGather(N, D, Hp, C), [kv, attn1, z1, mass1], C, Hp));
    ops.push(this.op("pool", K.poolMid(D, C, Q), [z1, f16("pool.q_child"), nw, nbias, parent, qleaf], Q));
    ops.push(this.op("pool", K.slotLogits(N, D, Hp, Q), [kv, qleaf, f32("pool.q_child_bias"), attn2], Math.ceil(N / 64), Hp));
    ops.push(this.op("pool", K.slotGather(N, D, Hp, Q), [kv, attn2, z2, mass2], Q, Hp));
    ops.push(
      this.op("pool", K.poolTail(D, Hp, Q, E), [parent, z2, mass2, nw, nbias, f16("pool.leaf.weight"), f16("pool.leaf.bias"), leaf], Q),
    );
    const headBias = f16("head.bias");
    for (const c of headChunks) {
      ops.push(this.op("head", K.headGemv(c.rows, hCols, c.row0), [c.buf, leaf, headBias, this.probs], Math.ceil(c.rows / 8)));
    }
    this.ops = await Promise.all(ops);
    await this.device.queue.onSubmittedWorkDone();
  }

  // ------------------------------------------------------------------ inference

  private encodeForward(enc: GPUCommandEncoder, timestamps?: GPUQuerySet) {
    enc.copyBufferToBuffer(this.cls, 0, this.resid, 0, this.config.model.dim * 4);
    if (!timestamps) {
      const pass = enc.beginComputePass();
      for (const o of this.ops) {
        pass.setPipeline(o.pipeline);
        pass.setBindGroup(0, o.bind);
        pass.dispatchWorkgroups(o.x, o.y);
      }
      pass.end();
    } else {
      this.ops.forEach((o, i) => {
        const pass = enc.beginComputePass({
          timestampWrites: { querySet: timestamps, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 },
        });
        pass.setPipeline(o.pipeline);
        pass.setBindGroup(0, o.bind);
        pass.dispatchWorkgroups(o.x, o.y);
        pass.end();
      });
    }
    enc.copyBufferToBuffer(this.probs, 0, this.readback, 0, this.tagCount * 4);
  }

  private async finish(enc: GPUCommandEncoder, cleanup?: () => void): Promise<Float32Array> {
    this.device.queue.submit([enc.finish()]);
    cleanup?.();
    await this.readback.mapAsync(GPUMapMode.READ);
    const probs = new Float32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();
    return probs;
  }

  /** Tag probabilities for a decoded image (unpremultiplied, as createImageBitmap with premultiplyAlpha "none"). */
  async run(image: ImageBitmap): Promise<Float32Array> {
    const enc = this.device.createCommandEncoder();
    const pre = this.pre.encode(enc, image);
    this.encodeForward(enc);
    return this.finish(enc, pre.destroy);
  }

  /** Tag probabilities for already preprocessed pixels (uint8 [size, size, 3]), as foidfinder's preprocess(). */
  async runPixels(pixels: Uint8Array): Promise<Float32Array> {
    const { img_size: s, patch_size: p } = this.config.model;
    const grid = s / p;
    const a = new Uint16Array(grid * grid * this.kpad);
    pixels.forEach((v, i) => {
      const c = i % 3;
      const x = Math.floor(i / 3) % s;
      const y = Math.floor(i / 3 / s);
      const row = Math.floor(y / p) * grid + Math.floor(x / p);
      a[row * this.kpad + ((y % p) * p + (x % p)) * 3 + c] = floatToHalf(v);
    });
    this.device.queue.writeBuffer(this.patches, 0, a);
    const enc = this.device.createCommandEncoder();
    this.encodeForward(enc);
    return this.finish(enc);
  }

  /** Per-kernel GPU time (ms) over `repeats` forwards of the current input, plus the wall time per forward. */
  async profile(repeats: number): Promise<{ kernels: Map<string, number>; wallMs: number }> {
    const n = this.ops.length;
    const kernels = new Map<string, number>();
    if (this.device.features.has("timestamp-query")) {
      const qs = this.device.createQuerySet({ type: "timestamp", count: 2 * n });
      const resolve = this.device.createBuffer({ size: 16 * n, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const read = this.device.createBuffer({ size: 16 * n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      for (let r = 0; r < repeats; r++) {
        const enc = this.device.createCommandEncoder();
        this.encodeForward(enc, qs);
        enc.resolveQuerySet(qs, 0, 2 * n, resolve, 0);
        enc.copyBufferToBuffer(resolve, 0, read, 0, 16 * n);
        await this.finish(enc);
        await read.mapAsync(GPUMapMode.READ);
        const ts = new BigInt64Array(read.getMappedRange().slice(0));
        read.unmap();
        this.ops.forEach((o, i) => {
          kernels.set(o.label, (kernels.get(o.label) ?? 0) + Number(at(ts, 2 * i + 1) - at(ts, 2 * i)) / 1e6 / repeats);
        });
      }
      qs.destroy();
      resolve.destroy();
      read.destroy();
    }
    const t0 = performance.now();
    for (let r = 0; r < repeats; r++) {
      const enc = this.device.createCommandEncoder();
      this.encodeForward(enc);
      await this.finish(enc);
    }
    return { kernels, wallMs: (performance.now() - t0) / repeats };
  }

  destroy() {
    this.buffers.forEach((b) => b.destroy());
  }
}
