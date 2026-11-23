/** Model weight fetching (Cache API), safetensors parsing, gzip decompression, and fp16 helpers. */

export const MODEL_CACHE = "foidfinder-models";

export type OnProgress = (loaded: number, total: number) => void;

/** Returns true when `url` is already stored in the Cache API. */
export async function isFileCached(url: string): Promise<boolean> {
  if (typeof caches === "undefined") return false;
  const cache = await caches.open(MODEL_CACHE);
  return Boolean(await cache.match(url));
}

/** Fetches `url` from the Cache API when present, otherwise downloads with progress and caches it. */
export async function fetchCached(url: string, onProgress?: OnProgress, stallTimeoutMs = 30_000): Promise<ArrayBuffer> {
  const cache = typeof caches === "undefined" ? null : await caches.open(MODEL_CACHE);
  const hit = await cache?.match(url);
  if (hit) {
    const buf = await hit.arrayBuffer();
    onProgress?.(buf.byteLength, buf.byteLength);
    return buf;
  }
  const controller = new AbortController();
  let stalled = false;
  let stallTimer: ReturnType<typeof setTimeout> | undefined;
  const resetStall = () => {
    clearTimeout(stallTimer);
    stallTimer = setTimeout(() => {
      stalled = true;
      controller.abort();
    }, stallTimeoutMs);
  };
  resetStall();
  let res: Response;
  try {
    res = await fetch(url, { signal: controller.signal, referrerPolicy: "no-referrer" });
  } catch (err) {
    clearTimeout(stallTimer);
    if (stalled) throw new Error("The model download timed out", { cause: err });
    throw err;
  }
  if (!res.ok || !res.body) {
    clearTimeout(stallTimer);
    throw new Error(`The model download failed (HTTP ${res.status})`);
  }
  const total = Number(res.headers.get("Content-Length")) || 0;
  const out = new Uint8Array(total);
  const parts: Uint8Array[] = [];
  let loaded = 0;
  const reader = res.body.getReader();
  try {
    for (;;) {
      resetStall();
      const { done, value } = await reader.read();
      if (done) break;
      if (total && loaded + value.byteLength <= total) out.set(value, loaded);
      else parts.push(value);
      loaded += value.byteLength;
      onProgress?.(loaded, total || loaded);
    }
  } catch (err) {
    if (stalled) throw new Error("The model download stalled", { cause: err });
    throw err;
  } finally {
    clearTimeout(stallTimer);
  }
  if (total && loaded !== total) throw new Error(`The model download was incomplete (${loaded} of ${total} bytes)`);
  let buf: ArrayBuffer;
  if (loaded === total) buf = out.buffer;
  else {
    buf = new ArrayBuffer(loaded);
    const all = new Uint8Array(buf);
    let o = 0;
    if (total) {
      all.set(out.subarray(0, Math.min(total, loaded)));
      o = Math.min(total, loaded);
    }
    for (const p of parts) {
      all.set(p, o);
      o += p.byteLength;
    }
  }
  await cache?.put(url, new Response(buf.slice(0), { headers: { "Content-Type": "application/octet-stream" } }));
  return buf;
}

export interface Tensor {
  dtype: string;
  shape: number[];
  bytes: Uint8Array;
}

export function parseSafetensors(buf: ArrayBuffer): Map<string, Tensor> {
  const view = new DataView(buf);
  const headerLen = Number(view.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, headerLen))) as Record<
    string,
    { dtype: string; shape: number[]; data_offsets: [number, number] }
  >;
  const start = 8 + headerLen;
  const out = new Map<string, Tensor>();
  for (const [name, t] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [a, b] = t.data_offsets;
    out.set(name, { dtype: t.dtype, shape: t.shape, bytes: new Uint8Array(buf, start + a, b - a) });
  }
  return out;
}

export async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** IEEE 754 binary16 -> float. */
export function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

export function halves(t: Tensor): Uint16Array {
  if (t.dtype !== "F16") throw new Error(`expected F16, got ${t.dtype}`);
  // Copy: the tensor's offset in the file need not be 2-byte aligned.
  return new Uint16Array(t.bytes.slice().buffer);
}

export function toFloat32(t: Tensor): Float32Array {
  const h = halves(t);
  const f = new Float32Array(h.length);
  h.forEach((v, i) => (f[i] = halfToFloat(v)));
  return f;
}
