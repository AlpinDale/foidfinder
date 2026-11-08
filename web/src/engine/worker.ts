// Runs the WebGPU engine off the main thread: loads the model once, then tags images one at a time.

import { MAX_PIXELS, type TagResult } from "../tagger";
import { Haruka, requestDevice, type ModelConfig } from "./haruka";
import { TagIndex } from "./tagging";
import { fetchCached } from "./weights";

export type WorkerRequest = { type: "init"; base: string } | { type: "tag"; id: number; file: Blob };
export type WorkerReply =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready" }
  | { type: "failed"; message: string }
  | { type: "result"; id: number; result: TagResult }
  | { type: "error"; id: number; code: string; message: string };

const ctx = self as unknown as { postMessage(m: WorkerReply): void; onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null };

let model: Haruka | null = null;
let index: TagIndex | null = null;
let maxSide = 8192;
let queue: Promise<void> = Promise.resolve();

class TagError extends Error {
  constructor(
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

async function init(base: string) {
  try {
    const { device, features } = await requestDevice();
    maxSide = device.limits.maxTextureDimension2D;
    const config = JSON.parse(new TextDecoder().decode(await fetchCached(`${base}/config.json`))) as ModelConfig;
    const weights = await fetchCached(`${base}/model.safetensors`, (loaded, total) => ctx.postMessage({ type: "progress", loaded, total }));
    model = await Haruka.load(device, features, config, weights);
    index = new TagIndex(config.tags, config.categories, config.thresholds);
    void device.lost.then((i) => {
      model = null;
      ctx.postMessage({ type: "failed", message: `GPU device lost: ${i.message}` });
    });
    ctx.postMessage({ type: "ready" });
  } catch (e) {
    ctx.postMessage({ type: "failed", message: e instanceof Error ? e.message : String(e) });
  }
}

async function decode(file: Blob): Promise<{ image: ImageBitmap; ms: number }> {
  const t0 = performance.now();
  try {
    const image = await createImageBitmap(file, { premultiplyAlpha: "none", colorSpaceConversion: "none" });
    return { image, ms: performance.now() - t0 };
  } catch {
    throw new TagError("unreadable_image", "Unsupported or damaged image");
  }
}

async function tag(decoded: Promise<{ image: ImageBitmap; ms: number }>): Promise<TagResult> {
  const { image, ms: decodeMs } = await decoded;
  try {
    if (!model || !index) throw new TagError("unavailable", "The model is not loaded");
    if (image.width * image.height > MAX_PIXELS) throw new TagError("too_many_pixels", `Larger than ${MAX_PIXELS / 1e6} megapixels`);
    if (Math.max(image.width, image.height) > maxSide) throw new TagError("too_many_pixels", `Wider or taller than ${maxSide} pixels`);
    const t1 = performance.now();
    const probs = await model.run(image);
    const inference = performance.now() - t1;
    return {
      ...index.result(probs),
      image: { width: image.width, height: image.height },
      timings_ms: { decode: decodeMs, inference, total: decodeMs + inference },
    };
  } finally {
    image.close();
  }
}

ctx.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    void init(msg.base);
    return;
  }
  // Decoding starts at once, so it overlaps the previous image's GPU work; the GPU runs one image at a time.
  const decoded = decode(msg.file);
  decoded.catch(() => {}); // handled when its turn comes
  queue = queue.then(async () => {
    try {
      ctx.postMessage({ type: "result", id: msg.id, result: await tag(decoded) });
    } catch (err) {
      const code = err instanceof TagError ? err.code : "inference_failed";
      ctx.postMessage({ type: "error", id: msg.id, code, message: err instanceof Error ? err.message : String(err) });
    }
  });
};
