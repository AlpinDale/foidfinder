import { TagIndex } from "../../tagging";
import type { DeviceFeatures } from "../../webgpu/device";
import { PatchPreprocessor } from "../../webgpu/preprocess";
import { parseSafetensors } from "../../webgpu/weights";
import type { LoadedModel } from "../types";
import { HARUKA_ARCHITECTURE, HARUKA_QUANT_FORMAT, type HarukaConfig } from "./config";
import { GpuArena, loadHarukaWeights, type ComputeOp } from "./loader";
import * as K from "./wgsl";

const EPS = 1e-6;
const TILE: K.Tile = { BM: 32, BN: 64, BK: 16, TM: 4, TN: 4 };

/** Haruka Tagger (EVA02-B/14 + 18-query tree pool) compiled as a static WebGPU compute graph. */
export class HarukaModel implements LoadedModel {
  readonly id = HARUKA_ARCHITECTURE;
  readonly name: string;
  readonly tagCount: number;
  readonly index: TagIndex;

  private arena: GpuArena;
  private ops: ComputeOp[] = [];
  private pre!: PatchPreprocessor;
  private resid!: GPUBuffer;
  private cls!: GPUBuffer;
  private probs!: GPUBuffer;
  private readback!: GPUBuffer;
  private dim: number;

  private constructor(device: GPUDevice, config: HarukaConfig) {
    this.arena = new GpuArena(device);
    this.name = config.model.name ?? "Haruka Tagger";
    this.tagCount = config.tags.length;
    this.dim = config.model.dim;
    this.index = new TagIndex(config.tags, config.categories, config.thresholds);
  }

  static async load(device: GPUDevice, features: DeviceFeatures, config: HarukaConfig, weights: ArrayBuffer): Promise<HarukaModel> {
    if (config.quantization?.format !== HARUKA_QUANT_FORMAT) throw new Error("Unknown Haruka model format");
    const model = new HarukaModel(device, config);
    await model.build(features, config, weights);
    return model;
  }

  private async build(features: DeviceFeatures, config: HarukaConfig, weightsBuf: ArrayBuffer): Promise<void> {
    const cfg = config.model;
    const { dim: D, heads: H, mlp_hidden: HID, pool_heads: Hp, coarse_queries: C, leaf_queries: Q, leaf_dim: E } = cfg;
    const grid = cfg.img_size / cfg.patch_size;
    const N = grid * grid + 1;
    const kpad = Math.ceil((cfg.patch_size * cfg.patch_size * 3) / TILE.BK) * TILE.BK;
    const a = this.arena;

    const w = await loadHarukaWeights(a, config, parseSafetensors(weightsBuf), kpad);
    const weight = (name: string) => {
      const buf = w.weights.get(name);
      if (!buf) throw new Error(`missing compressed matrix ${name}`);
      return buf;
    };

    // Activation buffers.
    const patches = a.buffer((N - 1) * kpad * 2);
    this.resid = a.buffer(N * D * 4);
    this.cls = w.f32("cls");
    const xn = a.buffer(N * D * 2);
    const qkvh = a.buffer(N * 3 * D * 2);
    const att = a.buffer(N * D * 2);
    const gx = a.buffer(N * 2 * HID * 2);
    const hm = a.buffer(N * HID * 2);
    const xf = a.buffer(N * D * 4);
    const kv = a.buffer(N * 2 * D * 2);
    const attn1 = a.buffer(Hp * C * N * 4);
    const z1 = a.buffer(C * D * 4);
    const mass1 = a.buffer(Hp * C * 4);
    const parent = a.buffer(Q * D * 4);
    const qleaf = a.buffer(Q * D * 4);
    const attn2 = a.buffer(Hp * Q * N * 4);
    const z2 = a.buffer(Q * D * 4);
    const mass2 = a.buffer(Hp * Q * 4);
    const leaf = a.buffer(Q * E * 4);
    this.probs = a.buffer(this.tagCount * 4);
    this.readback = a.buffer(this.tagCount * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    this.pre = new PatchPreprocessor(a.device, cfg.img_size, cfg.patch_size, kpad, patches);

    const rowsM = Math.ceil(N / TILE.BM);
    const ops: Promise<ComputeOp>[] = [];
    ops.push(a.op(K.matmul(N - 1, D, kpad, TILE, { kind: "patch" }), [patches, w.patchWT, w.f16("patch_b"), this.resid], D / TILE.BN, Math.ceil((N - 1) / TILE.BM)));

    const ln = K.layerNorm(D, EPS);
    for (let i = 0; i < cfg.depth; i++) {
      const b = `blocks.${i}.`;
      ops.push(a.op(ln, [this.resid, w.f16(b + "norm1.weight"), w.f16(b + "norm1.bias"), xn], N));
      ops.push(a.op(K.matmul(N, 3 * D, D, TILE, { kind: "qkv", heads: H, dim: D }), [xn, weight(b + "qkv.weight"), w.f16(b + "qkv.bias"), qkvh, w.ropeCosSin], (3 * D) / TILE.BN, rowsM));
      ops.push(a.op(K.attention(N, H, features), [qkvh, att], Math.ceil(N / 64), H));
      ops.push(a.op(K.matmul(N, D, D, TILE, { kind: "residual" }), [att, weight(b + "proj.weight"), w.f16(b + "proj.bias"), this.resid], D / TILE.BN, rowsM));
      ops.push(a.op(ln, [this.resid, w.f16(b + "norm2.weight"), w.f16(b + "norm2.bias"), xn], N));
      ops.push(a.op(K.matmul(N, 2 * HID, D, TILE, { kind: "bias" }), [xn, weight(b + "fc1.weight"), w.f16(b + "fc1.bias"), gx], (2 * HID) / TILE.BN, rowsM));
      ops.push(a.op(K.swigluNorm(HID, EPS), [gx, w.f16(b + "mlp_norm.weight"), w.f16(b + "mlp_norm.bias"), hm], N));
      ops.push(a.op(K.matmul(N, D, HID, TILE, { kind: "residual" }), [hm, weight(b + "fc2.weight"), w.f16(b + "fc2.bias"), this.resid], D / TILE.BN, rowsM));
    }

    // Tree attention pool + classifier head.
    ops.push(a.op(K.layerNorm(D, EPS, "f32", "f32"), [this.resid, w.f16("norm.weight"), w.f16("norm.bias"), xf], N));
    ops.push(a.op(K.layerNorm(D, EPS, "f32", "f16"), [xf, w.f16("pool.norm.weight"), w.f16("pool.norm.bias"), xn], N));
    ops.push(a.op(K.matmul(N, 2 * D, D, TILE, { kind: "bias" }), [xn, weight("pool.kv.weight"), w.f16("pool.kv.bias"), kv], (2 * D) / TILE.BN, rowsM));
    ops.push(a.op(K.slotLogits(N, D, Hp, C), [kv, w.f32("pool.q_coarse"), w.f32("pool.q_coarse_bias"), attn1], Math.ceil(N / 64), Hp));
    ops.push(a.op(K.slotGather(N, D, Hp, C), [kv, attn1, z1, mass1], C, Hp));
    ops.push(a.op(K.poolMid(D, C, Q), [z1, w.f16("pool.q_child"), w.poolNormWeights, w.poolNormBiases, parent, qleaf], Q));
    ops.push(a.op(K.slotLogits(N, D, Hp, Q), [kv, qleaf, w.f32("pool.q_child_bias"), attn2], Math.ceil(N / 64), Hp));
    ops.push(a.op(K.slotGather(N, D, Hp, Q), [kv, attn2, z2, mass2], Q, Hp));
    ops.push(a.op(K.poolTail(D, Hp, Q, E), [parent, z2, mass2, w.poolNormWeights, w.poolNormBiases, w.f16("pool.leaf.weight"), w.f16("pool.leaf.bias"), leaf], Q));

    const headBias = w.f16("head.bias");
    for (const c of w.headChunks) {
      ops.push(a.op(K.headGemv(c.rows, w.headCols, c.row0), [c.buf, leaf, headBias, this.probs], Math.ceil(c.rows / 8)));
    }

    this.ops = await Promise.all(ops);
    await a.device.queue.onSubmittedWorkDone();
  }

  /** Tag probabilities for a decoded image (unpremultiplied, as createImageBitmap with premultiplyAlpha "none"). */
  async run(image: ImageBitmap): Promise<Float32Array> {
    const enc = this.arena.device.createCommandEncoder();
    const pre = this.pre.encode(enc, image);
    enc.copyBufferToBuffer(this.cls, 0, this.resid, 0, this.dim * 4);
    const pass = enc.beginComputePass();
    for (const o of this.ops) {
      pass.setPipeline(o.pipeline);
      pass.setBindGroup(0, o.bind);
      pass.dispatchWorkgroups(o.x, o.y);
    }
    pass.end();
    enc.copyBufferToBuffer(this.probs, 0, this.readback, 0, this.tagCount * 4);
    this.arena.device.queue.submit([enc.finish()]);
    pre.destroy();
    await this.readback.mapAsync(GPUMapMode.READ);
    const probs = new Float32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();
    return probs;
  }

  destroy(): void {
    this.arena.destroy();
  }
}
