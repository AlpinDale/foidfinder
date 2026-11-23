import { gunzip, halves, toFloat32, type Tensor } from "../../webgpu/weights";
import type { HarukaConfig } from "./config";
import { dequant, dequantT } from "./wgsl";

export interface ComputeOp {
  pipeline: GPUComputePipeline;
  bind: GPUBindGroup;
  x: number;
  y: number;
}

export interface HeadChunk {
  buf: GPUBuffer;
  row0: number;
  rows: number;
}

export interface HarukaWeights {
  weights: Map<string, GPUBuffer>;
  headChunks: HeadChunk[];
  headCols: number;
  patchWT: GPUBuffer;
  ropeCosSin: GPUBuffer;
  poolNormWeights: GPUBuffer;
  poolNormBiases: GPUBuffer;
  f16: (name: string) => GPUBuffer;
  f32: (name: string) => GPUBuffer;
}

/** Tracks GPU buffers and cached compute pipelines for a single loaded model instance. */
export class GpuArena {
  private pipelines = new Map<string, Promise<GPUComputePipeline>>();
  private buffers: GPUBuffer[] = [];

  constructor(readonly device: GPUDevice) {}

  buffer(size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC): GPUBuffer {
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
    this.buffers.push(b);
    return b;
  }

  upload(data: ArrayBufferView): GPUBuffer {
    const b = this.buffer(data.byteLength);
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    if (bytes.byteLength % 4) {
      const padded = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
      padded.set(bytes);
      this.device.queue.writeBuffer(b, 0, padded);
    } else {
      this.device.queue.writeBuffer(b, 0, bytes);
    }
    return b;
  }

  pipeline(code: string): Promise<GPUComputePipeline> {
    let p = this.pipelines.get(code);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      this.pipelines.set(code, p);
    }
    return p;
  }

  async op(code: string, buffers: GPUBuffer[], x: number, y = 1, offsets?: number[]): Promise<ComputeOp> {
    const pipeline = await this.pipeline(code);
    const bind = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({
        binding,
        resource: { buffer, offset: offsets?.[binding] ?? 0 },
      })),
    });
    return { pipeline, bind, x, y };
  }

  /** Dispatches a one-off compute kernel (such as weight reconstruction) and submits it. */
  async runOnce(code: string, buffers: GPUBuffer[], workgroups: number, offsets?: number[]): Promise<void> {
    const x = Math.min(workgroups, 32768);
    const o = await this.op(code, buffers, x, Math.ceil(workgroups / x), offsets);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(o.pipeline);
    pass.setBindGroup(0, o.bind);
    pass.dispatchWorkgroups(o.x, o.y);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  destroy(): void {
    this.buffers.forEach((b) => b.destroy());
    this.buffers.length = 0;
  }
}

/** Axial EVA02 RoPE (cos, sin) table [N, head_dim / 2, 2] with identity rotation on CLS token 0. */
function buildRopeTable(grid: number, refGrid: number, headDim: number): Float32Array {
  const N = grid * grid + 1;
  const nb = headDim / 4;
  const cs = new Float32Array(N * (headDim / 2) * 2);
  for (let n = 0; n < N; n++) {
    for (let b = 0; b < headDim / 2; b++) {
      let ang = 0;
      if (n > 0) {
        const gy = Math.floor((n - 1) / grid);
        const gx = (n - 1) % grid;
        const band = Math.fround(1 / Math.fround(10000 ** Math.fround((b % nb) / nb)));
        const pos = Math.fround(Math.fround((b < nb ? gy : gx) / grid) * refGrid);
        ang = Math.fround(pos * band);
      }
      cs[(n * (headDim / 2) + b) * 2] = Math.cos(ang);
      cs[(n * (headDim / 2) + b) * 2 + 1] = Math.sin(ang);
    }
  }
  return cs;
}

/** Decompresses byte-plane quantized weights on the GPU and uploads all static Haruka tensors. */
export async function loadHarukaWeights(
  arena: GpuArena,
  config: HarukaConfig,
  tensors: Map<string, Tensor>,
  kpad: number,
): Promise<HarukaWeights> {
  const cfg = config.model;
  const D = cfg.dim;
  const grid = cfg.img_size / cfg.patch_size;

  const get = (name: string): Tensor => {
    const v = tensors.get(name);
    if (!v) throw new Error(`missing tensor ${name}`);
    return v;
  };
  const layerSpec = (name: string) => {
    const l = config.quantization.layers[name];
    if (!l) throw new Error(`missing compressed matrix ${name}`);
    return l;
  };
  const f16 = (name: string) => arena.upload(halves(get(name)));
  const f32 = (name: string) => arena.upload(toFloat32(get(name)));
  const planesOf = (name: string) =>
    Promise.all(Array.from({ length: layerSpec(name).planes }, (_, i) => gunzip(get(`${name}.plane${i}`).bytes)));

  // Reconstruct transposed backbone + pool KV weights on the GPU.
  const weights = new Map<string, GPUBuffer>();
  for (const name of Object.keys(config.quantization.layers)) {
    if (name === "head.weight") continue;
    const [rows, cols] = layerSpec(name).shape;
    const planes = (await planesOf(name)).map((p) => arena.upload(p));
    const out = arena.buffer(rows * cols * 2);
    await arena.runOnce(dequantT(rows, cols, planes.length), [...planes, f16(`${name}.step`), out], Math.ceil((rows * cols) / 4 / 256));
    planes.forEach((p) => p.destroy());
    weights.set(name, out);
  }

  // Classifier head: row-major, split into chunks that fit maxStorageBufferBindingSize.
  const [hRows, hCols] = layerSpec("head.weight").shape;
  const limit = arena.device.limits.maxStorageBufferBindingSize;
  const chunkRows = Math.min(hRows, Math.floor(limit / (hCols * 2) / 128) * 128);
  const headPlanes = await planesOf("head.weight");
  const headStep = f16("head.weight.step");
  const headChunks: HeadChunk[] = [];
  for (let row0 = 0; row0 < hRows; row0 += chunkRows) {
    const rows = Math.min(chunkRows, hRows - row0);
    const planes = headPlanes.map((p) => arena.upload(p.subarray(row0 * hCols, (row0 + rows) * hCols)));
    const buf = arena.buffer(rows * hCols * 2);
    const offsets = [...planes.map(() => 0), row0 * 2, 0];
    await arena.runOnce(dequant(hCols, planes.length), [...planes, headStep, buf], Math.ceil((rows * hCols) / 4 / 256), offsets);
    planes.forEach((p) => p.destroy());
    headChunks.push({ buf, row0, rows });
  }

  // Patch embedding: [D, p*p*3] -> transposed and zero-padded to [kpad, D].
  const pw = halves(get("patch_w"));
  const pk = cfg.patch_size * cfg.patch_size * 3;
  const pwT = new Uint16Array(kpad * D);
  pw.forEach((v, i) => (pwT[(i % pk) * D + Math.floor(i / pk)] = v));

  const packPoolNorms = (kind: "weight" | "bias") => {
    const parts = ["trunk_norm", "child_q_norm", "leaf_norm"].map((n) => halves(get(`pool.${n}.${kind}`)));
    const all = new Uint16Array(parts.length * D);
    parts.forEach((p, i) => all.set(p, i * D));
    return arena.upload(all);
  };

  return {
    weights,
    headChunks,
    headCols: hCols,
    patchWT: arena.upload(pwT),
    ropeCosSin: arena.upload(buildRopeTable(grid, cfg.rope_ref_grid, D / cfg.heads)),
    poolNormWeights: packPoolNorms("weight"),
    poolNormBiases: packPoolNorms("bias"),
    f16,
    f32,
  };
}
