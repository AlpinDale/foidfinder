import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { TaggerError } from "../tagger";
import { useEngine } from "../useEngine";
import { LocalTagger } from "./client";

/** jsdom has no Worker; this one records messages and lets a test raise its events. */
class StubWorker {
  static last: StubWorker | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  onmessageerror: ((e: MessageEvent) => void) | null = null;
  posted: unknown[] = [];
  terminated = false;

  constructor() {
    StubWorker.last = this;
  }

  postMessage(m: unknown) {
    this.posted.push(m);
  }

  terminate() {
    this.terminated = true;
  }
}

const worker = () => {
  if (!StubWorker.last) throw new Error("no worker created");
  return StubWorker.last;
};

beforeEach(() => {
  StubWorker.last = null;
  vi.stubGlobal("Worker", StubWorker);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("LocalTagger", () => {
  it("rejects ready and every request when the worker fails to start", async () => {
    const tagger = new LocalTagger(() => {}, "https://models.test");
    const queued = tagger.tag(new Blob());
    worker().onerror?.(new Event("error"));

    await expect(tagger.ready).rejects.toThrow("failed to start");
    await expect(queued).rejects.toBeInstanceOf(TaggerError);
    await expect(tagger.tag(new Blob())).rejects.toMatchObject({ code: "unavailable" });
    expect(worker().terminated).toBe(true);
  });

  it("fails requests in flight when the worker dies after loading", async () => {
    const tagger = new LocalTagger(() => {}, "https://models.test");
    worker().onmessage?.(new MessageEvent("message", { data: { type: "ready" } }));
    await expect(tagger.ready).resolves.toBeUndefined();

    const inFlight = tagger.tag(new Blob());
    worker().onerror?.(new ErrorEvent("error", { message: "device lost" }));
    await expect(inFlight).rejects.toThrow("device lost");
  });
});

describe("useEngine", () => {
  it("reports why the model cannot run when the engine fails to start", async () => {
    vi.stubGlobal("navigator", { ...navigator, gpu: {} });
    const { result } = renderHook(() => useEngine());
    await waitFor(() => expect(StubWorker.last).not.toBeNull());
    worker().onmessage?.(
      new MessageEvent("message", { data: { type: "failed", message: "This GPU or browser lacks 16-bit floats" } }),
    );

    await waitFor(() => expect(result.current.error).toBe("This GPU or browser lacks 16-bit floats"));
    expect(result.current.tagger).toBeNull();
  });

  it("explains that the browser has no WebGPU", () => {
    const { result } = renderHook(() => useEngine());
    expect(result.current.error).toMatch(/no WebGPU/);
    expect(StubWorker.last).toBeNull();
  });
});
