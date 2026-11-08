// Dev page (bench.html): load the WebGPU engine, check it against the MLX reference in /dev-ref (written by
// foidfinder's reference dumper), and time it. Driven from the console or browser automation via window.bench.

import { at } from "./engine/at";
import { Haruka, requestDevice, type ModelConfig } from "./engine/haruka";
import { matmul, type Epilogue, type Tile } from "./engine/wgsl";
import { fetchCached, MODEL_BASE } from "./engine/weights";

let model: Haruka | null = null;
let config: ModelConfig | null = null;

const log = (msg: string) => {
  const el = document.getElementById("log");
  if (el) el.textContent += msg + "\n";
  console.log(msg);
};

/** `subgroups: false` runs the attention reductions through workgroup memory, as on GPUs without 16-lane subgroups. */
async function load(base = MODEL_BASE, subgroups = true) {
  const t0 = performance.now();
  const { device, features, description } = await requestDevice();
  device.addEventListener("uncapturederror", (e) => log(`GPU error: ${(e as GPUUncapturedErrorEvent).error.message}`));
  void device.lost.then((i) => log(`device lost: ${i.message}`));
  config = JSON.parse(new TextDecoder().decode(await fetchCached(`${base}/config.json`))) as ModelConfig;
  const weights = await fetchCached(`${base}/model.safetensors`);
  const t1 = performance.now();
  model?.destroy();
  model = await Haruka.load(device, { subgroups16: features.subgroups16 && subgroups }, config, weights);
  const t2 = performance.now();
  const r = { description, features, subgroups, fetchMs: Math.round(t1 - t0), buildMs: Math.round(t2 - t1) };
  log(JSON.stringify(r));
  return r;
}

function compare(got: Float32Array, ref: Float32Array, cfg: ModelConfig) {
  let maxDp = 0;
  let kl = 0;
  const flips: Record<string, number> = {};
  for (let j = 0; j < ref.length; j++) {
    const p = Math.min(Math.max(at(ref, j), 1e-7), 1 - 1e-7);
    const q = Math.min(Math.max(at(got, j), 1e-7), 1 - 1e-7);
    maxDp = Math.max(maxDp, Math.abs(p - q));
    kl += p * Math.log(p / q) + (1 - p) * Math.log((1 - p) / (1 - q));
    const c = at(cfg.categories, j);
    const th = cfg.thresholds[c] ?? NaN;
    if (c !== "rating" && at(ref, j) >= th !== at(got, j) >= th) flips[c] = (flips[c] ?? 0) + 1;
  }
  return { maxDp, kl, flips };
}

async function validate() {
  if (!model || !config) throw new Error("load first");
  const man = (await (await fetch("/dev-ref/manifest.json")).json()) as { n: number; files: string[]; tags: number };
  const px = new Uint8Array(await (await fetch("/dev-ref/pixels.u8")).arrayBuffer());
  const ref = new Float32Array(await (await fetch("/dev-ref/probs.f32")).arrayBuffer());
  const S = config.model.img_size ** 2 * 3;
  const out = [];
  for (let i = 0; i < man.n; i++) {
    const r = ref.subarray(i * man.tags, (i + 1) * man.tags);
    const fromPixels = await model.runPixels(px.subarray(i * S, (i + 1) * S));
    const blob = await (await fetch(`/dev-ref/${man.files[i]}`)).blob();
    const bitmap = await createImageBitmap(blob, { premultiplyAlpha: "none", colorSpaceConversion: "none" });
    const fromFile = await model.run(bitmap);
    let preDiff = 0;
    fromFile.forEach((v, j) => (preDiff = Math.max(preDiff, Math.abs(v - at(fromPixels, j)))));
    out.push({ file: man.files[i], ...compare(fromPixels, r, config), filePathVsPixels: preDiff });
  }
  log(JSON.stringify(out));
  return out;
}

async function profile(repeats = 10) {
  if (!model) throw new Error("load first");
  const { kernels, wallMs } = await model.profile(repeats);
  const total = [...kernels.values()].reduce((a, b) => a + b, 0);
  const r = { wallMs, gpuMs: total, kernels: Object.fromEntries([...kernels].map(([k, v]) => [k, Math.round(v * 100) / 100])) };
  log(JSON.stringify(r));
  return r;
}

/**
 * TFLOPS of each tile on random data for one matmul shape (C = A WT). Each tile is timed twice and the faster run
 * kept, because the GPU clock ramps up over the first dispatches.
 */
async function tune(M: number, N: number, K: number, tiles: Tile[], kind: Epilogue["kind"] = "bias", reps = 20) {
  const { device } = await requestDevice();
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const rnd = (n: number) => {
    const h = new Uint16Array(n);
    for (let i = 0; i < n; i++) h[i] = 0x3000 + Math.floor(Math.random() * 0x800); // small positive halves
    const b = device.createBuffer({ size: n * 2, usage });
    device.queue.writeBuffer(b, 0, h);
    return b;
  };
  const A = rnd(M * K);
  const W = rnd(K * N);
  const bias = rnd(N);
  const C = device.createBuffer({ size: M * N * 4, usage });
  const qs = device.createQuerySet({ type: "timestamp", count: 2 });
  const res = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const out: Record<string, number> = {};
  for (const t of tiles) {
    const key = `${t.BM}x${t.BN}x${t.BK} ${t.TM}x${t.TN}`;
    try {
      const ep: Epilogue = kind === "residual" ? { kind } : { kind: "bias" };
      const module = device.createShaderModule({ code: matmul(M, N, K, t, ep) });
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [A, W, bias, C].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      let best = Infinity;
      for (let run = 0; run < 2; run++) {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        for (let r = 0; r < reps; r++) pass.dispatchWorkgroups(N / t.BN, Math.ceil(M / t.BM));
        pass.end();
        enc.resolveQuerySet(qs, 0, 2, res, 0);
        enc.copyBufferToBuffer(res, 0, read, 0, 16);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const ts = new BigInt64Array(read.getMappedRange().slice(0));
        read.unmap();
        best = Math.min(best, Number(at(ts, 1) - at(ts, 0)) / 1e6 / reps);
      }
      out[key] = Math.round(((2 * M * N * K) / best / 1e9) * 100) / 100;
    } catch (e) {
      out[key] = -1;
      log(`${key}: ${String(e)}`);
    }
  }
  device.destroy();
  log(`${M}x${N}x${K}: ${JSON.stringify(out)}`);
  return out;
}

declare global {
  interface Window {
    bench: { load: typeof load; validate: typeof validate; profile: typeof profile; tune: typeof tune };
  }
}
window.bench = { load, validate, profile, tune };
