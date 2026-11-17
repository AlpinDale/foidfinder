// Runs the in-browser tagger (LocalTagger in a Web Worker) and fetches/crops images for identification.
// Used by the Chrome offscreen document and directly by the Firefox background event page.

import { LocalTagger } from "../engine/client";
import { MODEL_BASE, MODEL_CACHE } from "../engine/weights";
import { MAX_BYTES, TaggerError } from "../tagger";
import type { Failure, IdentifyReply, ImageSource } from "./messages";

/** Unused this long, the model is released to free GPU memory. */
const IDLE_MS = 5 * 60_000;
const PROGRESS_INTERVAL_MS = 200;

class FetchFailed extends Error {}

function failure(code: string, message: string): IdentifyReply {
  const error: Failure = { code, message };
  return { ok: false, error };
}

async function load(image: ImageSource): Promise<Blob> {
  if (image.kind === "screenshot") {
    const { x, y, width, height } = image.rect;
    const shot = await (await fetch(image.dataUrl)).blob();
    const bitmap = await createImageBitmap(shot, x, y, width, height);
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    canvas.getContext("2d")?.drawImage(bitmap, 0, 0);
    bitmap.close();
    return canvas.convertToBlob({ type: "image/png" });
  }
  let res: Response;
  try {
    // No cookies: cross-origin reads allowed by CORS ("*") must not carry credentials.
    res = await fetch(image.url, { credentials: "omit" });
  } catch {
    throw new FetchFailed("The image could not be downloaded");
  }
  if (!res.ok) throw new FetchFailed(`The image server refused it (HTTP ${res.status})`);
  // Hotlink protection often answers with a web page instead of the image.
  const type = res.headers.get("Content-Type") ?? "";
  if (/^(text|application\/(json|xhtml|xml))/.test(type)) throw new FetchFailed("The image server sent a page instead of the image");
  return res.blob();
}

export interface RunnerCallbacks {
  onProgress: (loaded: number, total: number) => void;
  onIdle: () => void;
}

export interface TaggerRunner {
  identify(image: ImageSource): Promise<IdentifyReply>;
  isCached(): Promise<boolean>;
  warm(): Promise<boolean>;
  dispose(): void;
}

export function createRunner(callbacks: RunnerCallbacks): TaggerRunner {
  let tagger: LocalTagger | null = null;
  let busy = 0;
  let idleTimer: ReturnType<typeof setTimeout> | undefined;
  let lastProgress = 0;

  function engine(): LocalTagger {
    if (!tagger) {
      const t = new LocalTagger((loaded, total) => {
        const now = performance.now();
        if (loaded < total && now - lastProgress < PROGRESS_INTERVAL_MS) return;
        lastProgress = now;
        callbacks.onProgress(loaded, total);
      });
      // A failed load is not cached: the next image tries again (the reason may have been a dropped download).
      t.ready.catch(() => {
        if (tagger === t) tagger = null;
      });
      tagger = t;
    }
    return tagger;
  }

  async function identify(image: ImageSource): Promise<IdentifyReply> {
    globalThis.clearTimeout(idleTimer);
    idleTimer = undefined;
    busy += 1;
    try {
      const blob = await load(image);
      if (blob.size > MAX_BYTES) return failure("too_large", "The image is larger than 32 MB");
      return { ok: true, result: await engine().tag(blob) };
    } catch (err) {
      if (err instanceof FetchFailed) return failure("fetch_failed", err.message);
      if (err instanceof TaggerError) return failure(err.code, err.message);
      return failure("inference_failed", err instanceof Error ? err.message : String(err));
    } finally {
      busy -= 1;
      if (busy === 0) idleTimer = globalThis.setTimeout(() => callbacks.onIdle(), IDLE_MS);
    }
  }

  async function isCached(): Promise<boolean> {
    if (tagger) return true;
    if (typeof caches === "undefined") return false;
    const base = import.meta.env.VITE_MODEL_BASE ?? MODEL_BASE;
    const cache = await caches.open(MODEL_CACHE);
    return Boolean(await cache.match(`${base}/model.safetensors`));
  }

  async function warm(): Promise<boolean> {
    try {
      await engine().ready;
      return true;
    } catch {
      return false;
    }
  }

  function dispose() {
    if (busy > 0) return;
    globalThis.clearTimeout(idleTimer);
    idleTimer = undefined;
    tagger?.terminate();
    tagger = null;
  }

  return { identify, isCached, warm, dispose };
}
