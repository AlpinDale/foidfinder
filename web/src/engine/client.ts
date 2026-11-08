// The in-browser tagger: a Web Worker running the WebGPU engine.

import { TaggerError, type TagResult, type Tagger } from "../tagger";
import { MODEL_BASE } from "./weights";
import type { WorkerReply, WorkerRequest } from "./worker";

export function webgpuAvailable(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

export class LocalTagger implements Tagger {
  /** Resolves once the model is loaded; rejects with the reason when this browser cannot run it. */
  readonly ready: Promise<void>;
  private worker: Worker;
  private pending = new Map<number, { resolve: (r: TagResult) => void; reject: (e: unknown) => void }>();
  private nextId = 0;
  private failure: TaggerError | null = null;
  private rejectReady: (e: unknown) => void;

  constructor(onProgress: (loaded: number, total: number) => void, base: string = import.meta.env.VITE_MODEL_BASE ?? MODEL_BASE) {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    const ready = Promise.withResolvers<undefined>();
    this.ready = ready.promise;
    this.rejectReady = ready.reject;
    this.worker.onmessage = (e: MessageEvent<WorkerReply>) => {
      const msg = e.data;
      switch (msg.type) {
        case "progress":
          onProgress(msg.loaded, msg.total);
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
    // A worker that fails to load or throws outside a request never replies; without these, ready and every pending
    // request would wait forever and the page could not fall back to the server.
    this.worker.onerror = (e: Event) => {
      e.preventDefault();
      this.fail(e instanceof ErrorEvent && e.message ? e.message : "The in-browser engine failed to start");
    };
    this.worker.onmessageerror = () => this.fail("Unreadable message from the in-browser engine");
    this.post({ type: "init", base });
  }

  private post(msg: WorkerRequest) {
    this.worker.postMessage(msg);
  }

  /** The engine is unusable: reject `ready` (if still loading) and every request, now and later. */
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
    // An aborted request still runs in the worker; its result is dropped.
    signal?.addEventListener("abort", () => {
      if (this.pending.delete(id)) reject(signal.reason);
    });
    this.pending.set(id, { resolve, reject });
    this.post({ type: "tag", id, file });
    return promise;
  }

  terminate() {
    this.fail("Stopped");
  }
}
