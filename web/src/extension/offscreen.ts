// The extension's offscreen document: keeps the model loaded in the engine's Web Worker and identifies images for
// the service worker, which cannot start workers itself.

import { LocalTagger } from "../engine/client";
import { MAX_BYTES, TaggerError } from "../tagger";
import type { Failure, IdentifyReply, ImageSource, ToBackground, ToOffscreen } from "./messages";

/** Unused this long, the model is released (the service worker closes this document). */
const IDLE_MS = 5 * 60_000;
const PROGRESS_INTERVAL_MS = 200;

let tagger: LocalTagger | null = null;
let busy = 0;
let idleTimer: number | undefined;
let lastProgress = 0;

function send(msg: ToBackground) {
  void chrome.runtime.sendMessage(msg).catch(() => {});
}

function engine(): LocalTagger {
  if (!tagger) {
    const t = new LocalTagger((loaded, total) => {
      const now = performance.now();
      if (loaded < total && now - lastProgress < PROGRESS_INTERVAL_MS) return;
      lastProgress = now;
      send({ target: "background", type: "progress", loaded, total });
    });
    // A failed load is not cached: the next image tries again (the reason may have been a dropped download).
    t.ready.catch(() => {
      if (tagger === t) tagger = null;
    });
    tagger = t;
  }
  return tagger;
}

class FetchFailed extends Error {}

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
    res = await fetch(image.url, { credentials: "include" });
  } catch {
    throw new FetchFailed("The image could not be downloaded");
  }
  if (!res.ok) throw new FetchFailed(`The image server refused it (HTTP ${res.status})`);
  // Hotlink protection often answers with a web page instead of the image.
  const type = res.headers.get("Content-Type") ?? "";
  if (/^(text|application\/(json|xhtml|xml))/.test(type)) throw new FetchFailed("The image server sent a page instead of the image");
  return res.blob();
}

async function identify(image: ImageSource): Promise<IdentifyReply> {
  window.clearTimeout(idleTimer);
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
    if (busy === 0) idleTimer = window.setTimeout(() => send({ target: "background", type: "idle" }), IDLE_MS);
  }
}

function failure(code: string, message: string): IdentifyReply {
  const error: Failure = { code, message };
  return { ok: false, error };
}

chrome.runtime.onMessage.addListener((msg: ToOffscreen, _sender, sendResponse) => {
  if (msg?.target !== "offscreen") return false;
  void identify(msg.image).then(sendResponse);
  return true; // replies asynchronously
});
