// Client for the API in foidfinder/web.py.

export interface Tag {
  tag: string;
  score: number;
}

export type Category = "general" | "character" | "copyright" | "style" | "meta";

export interface TagResult {
  rating: Tag | null;
  tags: Record<Category, Tag[]>;
  candidates: { character: Tag[]; copyright: Tag[] };
  image: { width: number; height: number };
  timings_ms: { decode: number; inference: number; total: number };
}

export interface ServerInfo {
  model: string;
  backend: string;
  device: string;
  max_upload_bytes: number;
  max_pixels: number;
  formats: string[];
}

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  /** Retry-After seconds from a 503. */
  readonly retryAfter: number | null;

  constructor(message: string, code: string, status: number, retryAfter: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.retryAfter = retryAfter;
  }

  get retryable(): boolean {
    return this.code === "busy" || this.code === "network" || this.code === "timeout" || this.status >= 500;
  }
}

const TAG_TIMEOUT_MS = 120_000;

async function request<T>(path: string, init: RequestInit, timeoutMs: number): Promise<T> {
  const controller = new AbortController();
  const outer = init.signal;
  const onAbort = () => controller.abort(outer?.reason);
  outer?.addEventListener("abort", onAbort, { once: true });
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    let res: Response;
    try {
      res = await fetch(path, { ...init, signal: controller.signal });
    } catch (err) {
      if (outer?.aborted) throw err;
      if (timedOut) throw new ApiError("Server timed out", "timeout", 0);
      throw new ApiError("Can't reach the server", "network", 0);
    }
    const body: unknown = await res.json().catch(() => null);
    if (!res.ok) {
      const error = (body as { error?: { code?: string; message?: string } } | null)?.error;
      const retryAfter = Number(res.headers.get("Retry-After"));
      throw new ApiError(
        error?.message ?? `Server error (${res.status})`,
        error?.code ?? "server_error",
        res.status,
        Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter : null,
      );
    }
    if (body === null) throw new ApiError("Unreadable server response", "server_error", res.status);
    return body as T;
  } finally {
    clearTimeout(timer);
    outer?.removeEventListener("abort", onAbort);
  }
}

export function fetchInfo(signal?: AbortSignal): Promise<ServerInfo> {
  return request<ServerInfo>("/api/info", { signal }, 15_000);
}

export function tagImage(file: Blob, signal?: AbortSignal): Promise<TagResult> {
  return request<TagResult>(
    "/api/tag",
    { method: "POST", body: file, headers: { "Content-Type": file.type || "application/octet-stream" }, signal },
    TAG_TIMEOUT_MS,
  );
}
