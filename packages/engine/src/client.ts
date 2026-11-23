// The in-browser tagger: a Web Worker running a registered WebGPU model.

import { HARUKA_DEFAULT_BASE } from "./models/haruka/config";
import type { ModelSource } from "./models/types";
import { TaggerError, type Tagger, type TagResult } from "./types";
import { isFileCached, type OnProgress } from "./webgpu/weights";
import type { WorkerReply, WorkerRequest } from "./worker";

export function webgpuAvailable(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

/** Returns true when the model checkpoint for `source` is already stored in the Cache API. */
export function isModelCached(source?: ModelSource): Promise<boolean> {
  const envBase = import.meta.env?.VITE_MODEL_BASE as string | undefined;
  const baseUrl = (source?.baseUrl ?? envBase ?? HARUKA_DEFAULT_BASE).replace(/\/+$/, "");
  return isFileCached(`${baseUrl}/model.safetensors`);
}

export interface LocalTaggerOptions {
  onProgress?: OnProgress;
  /** Model source override (defaults to VITE_MODEL_BASE or the worker's default registered model). */
  source?: ModelSource;
}

export class LocalTagger implements Tagger {
  /** Resolves once the model is loaded; rejects with the reason when this browser cannot run it. */
  readonly ready: Promise<void>;
  private worker: Worker;
  private pending = new Map<number, { resolve: (r: TagResult) => void; reject: (e: unknown) => void }>();
  private nextId = 0;
  private failure: TaggerError | null = null;
  private rejectReady: (e: unknown) => void;

  constructor(options: LocalTaggerOptions = {}) {
    const envBase = import.meta.env?.VITE_MODEL_BASE as string | undefined;
    const source: ModelSource = {
      baseUrl: options.source?.baseUrl ?? envBase,
      modelId: options.source?.modelId,
    };
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    const ready = Promise.withResolvers<undefined>();
    this.ready = ready.promise;
    this.rejectReady = ready.reject;
    this.worker.onmessage = (e: MessageEvent<WorkerReply>) => {
      const msg = e.data;
      switch (msg.type) {
        case "progress":
          options.onProgress?.(msg.loaded, msg.total);
          break;
        case "ready":
          ready.resolve(undefined);
          break;
        case "failed":
          this.fail(msg.message);
          break;
        case "result":
          this.pending.get(msg.id)?.resolve(msg.result);
          this.pending.delete(msg.id);
          break;
        case "error":
          this.pending.get(msg.id)?.reject(new TaggerError(msg.message, msg.code));
          this.pending.delete(msg.id);
          break;
      }
    };
    // A worker that fails to load or throws outside a request never replies; fail immediately so callers don't hang.
    this.worker.onerror = (e: Event) => {
      e.preventDefault();
      this.fail(e instanceof ErrorEvent && e.message ? e.message : "The in-browser engine failed to start");
    };
    this.worker.onmessageerror = () => this.fail("Unreadable message from the in-browser engine");
    this.post({ type: "init", source });
  }

  private post(msg: WorkerRequest) {
    this.worker.postMessage(msg);
  }

  /** Marks the engine unusable: rejects `ready` (if still loading) and every request, now and later. */
  private fail(message: string) {
    if (this.failure) return;
    this.failure = new TaggerError(message, "unavailable");
    this.rejectReady(new Error(message));
    this.worker.terminate();
    this.pending.forEach((p) => p.reject(this.failure));
    this.pending.clear();
  }

  tag(file: Blob, signal?: AbortSignal): Promise<TagResult> {
    if (this.failure) return Promise.reject(this.failure);
    const id = this.nextId++;
    const { promise, resolve, reject } = Promise.withResolvers<TagResult>();
    // An aborted request still runs in the worker if already sent; its result is discarded.
    signal?.addEventListener("abort", () => {
      if (this.pending.delete(id)) reject(signal.reason);
    });
    this.pending.set(id, { resolve, reject });
    // The worker accepts images once the model is loaded; until then they queue here and fail() rejects them.
    this.ready.then(
      () => {
        if (this.pending.has(id)) this.post({ type: "tag", id, file });
      },
      () => {},
    );
    return promise;
  }

  terminate() {
    this.fail("Stopped");
  }
}
