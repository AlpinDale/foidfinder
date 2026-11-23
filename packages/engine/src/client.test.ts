import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LocalTagger } from "./client";
import { TaggerError } from "./types";

/** jsdom has no Worker; this stub records messages and lets tests trigger its events. */
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
    const tagger = new LocalTagger({ source: { baseUrl: "https://models.test" } });
    const queued = tagger.tag(new Blob());
    worker().onerror?.(new Event("error"));

    await expect(tagger.ready).rejects.toThrow("failed to start");
    await expect(queued).rejects.toBeInstanceOf(TaggerError);
    await expect(tagger.tag(new Blob())).rejects.toMatchObject({ code: "unavailable" });
    expect(worker().terminated).toBe(true);
  });

  it("holds images sent before the model is loaded until it is", async () => {
    const tagger = new LocalTagger({ source: { baseUrl: "https://models.test" } });
    const file = new Blob(["x"]);
    const result = tagger.tag(file);
    await Promise.resolve();
    expect(worker().posted.map((m) => (m as { type: string }).type)).toEqual(["init"]);

    worker().onmessage?.(new MessageEvent("message", { data: { type: "ready", model: { id: "haruka-eva02-tree", name: "Haruka Tagger" } } }));
    await tagger.ready;
    await Promise.resolve();
    expect(worker().posted.at(-1)).toEqual({ type: "tag", id: 0, file });

    const answer = { rating: null, tags: {}, candidates: {}, image: { width: 1, height: 1 }, timings_ms: { decode: 0, inference: 0, total: 0 } };
    worker().onmessage?.(new MessageEvent("message", { data: { type: "result", id: 0, result: answer } }));
    await expect(result).resolves.toEqual(answer);
  });

  it("fails requests in flight when the worker dies after loading", async () => {
    const tagger = new LocalTagger({ source: { baseUrl: "https://models.test" } });
    worker().onmessage?.(new MessageEvent("message", { data: { type: "ready", model: { id: "haruka-eva02-tree", name: "Haruka Tagger" } } }));
    await expect(tagger.ready).resolves.toBeUndefined();

    const inFlight = tagger.tag(new Blob());
    worker().onerror?.(new ErrorEvent("error", { message: "device lost" }));
    await expect(inFlight).rejects.toThrow("device lost");
  });
});
