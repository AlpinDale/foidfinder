import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useEngine } from "./useEngine";

class StubWorker {
  static last: StubWorker | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  onmessageerror: ((e: MessageEvent) => void) | null = null;
  posted: unknown[] = [];

  constructor() {
    StubWorker.last = this;
  }

  postMessage(m: unknown) {
    this.posted.push(m);
  }

  terminate() {}
}

beforeEach(() => {
  StubWorker.last = null;
  vi.stubGlobal("Worker", StubWorker);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useEngine", () => {
  it("reports why the model cannot run when the engine fails to start", async () => {
    vi.stubGlobal("navigator", { ...navigator, gpu: {} });
    const { result } = renderHook(() => useEngine());
    await waitFor(() => expect(StubWorker.last).not.toBeNull());
    StubWorker.last?.onmessage?.(
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
