import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";

import { App } from "./App";
import { TaggerError, type TagResult } from "./tagger";

// The WebGPU engine, replaced by one whose loading and answers each test controls.
const engine = vi.hoisted(() => ({
  gpu: true,
  ready: Promise.resolve(),
  onProgress: null as ((loaded: number, total: number) => void) | null,
  answers: [] as (() => Promise<TagResult>)[],
  calls: 0,
}));

vi.mock("./engine/client", () => ({
  webgpuAvailable: () => engine.gpu,
  LocalTagger: class {
    ready: Promise<void>;
    constructor(onProgress: (loaded: number, total: number) => void) {
      engine.onProgress = onProgress;
      this.ready = engine.ready;
    }
    tag() {
      engine.calls += 1;
      const next = engine.answers.shift();
      if (!next) throw new Error("unexpected tag call");
      return next();
    }
    terminate() {}
  },
}));

const RESULT: TagResult = {
  rating: { tag: "rating:g", score: 0.9 },
  tags: {
    character: [{ tag: "kafka_(honkai:_star_rail)", score: 0.97 }],
    copyright: [{ tag: "honkai:_star_rail", score: 0.99 }],
    general: [{ tag: "1girl", score: 0.99 }],
  },
  candidates: { character: [{ tag: "himeko_(honkai:_star_rail)", score: 0.12 }], copyright: [] },
  image: { width: 800, height: 1200 },
  timings_ms: { decode: 3, inference: 20, total: 25 },
};

beforeEach(() => {
  engine.gpu = true;
  engine.ready = Promise.resolve();
  engine.answers = [];
  engine.calls = 0;
});

const png = (name = "a.png", size = 10) => new File([new Uint8Array(size)], name, { type: "image/png" });

async function upload(...files: File[]) {
  render(<App />);
  await userEvent.upload(screen.getByTestId("file-input"), files);
}

it("shows the identified character, series and closest guess", async () => {
  engine.answers.push(async () => RESULT);
  await upload(png());
  expect(await screen.findByRole("link", { name: "Kafka" })).toHaveAttribute(
    "href",
    "https://danbooru.donmai.us/posts?tags=kafka_(honkai%3A_star_rail)",
  );
  expect(screen.getByRole("link", { name: "Himeko" })).toBeInTheDocument();
  expect(screen.getByText("800 × 1200")).toBeInTheDocument();
  expect(screen.getByText("Rated General")).toBeInTheDocument();
});

it("shows Haruka's artist, year and rating", async () => {
  engine.answers.push(async () => ({
    ...RESULT,
    rating: { tag: "rating_s", score: 0.8 },
    tags: {
      character: [{ tag: "kafka_(honkai:_star_rail)", score: 0.97 }],
      copyright: [],
      artist: [{ tag: "mika_pikazo", score: 0.81 }],
      general: [{ tag: "1girl", score: 0.99 }],
      year: [{ tag: "era_2023_2025", score: 0.66 }],
    },
    candidates: { character: [], copyright: [], artist: [{ tag: "ask_(askzy)", score: 0.2 }] },
  }));
  await upload(png());
  expect(await screen.findByRole("link", { name: "Mika Pikazo" })).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Ask" })).toBeInTheDocument();
  expect(screen.getByText("era 2023 2025")).toBeInTheDocument();
  expect(screen.getByText("Rated Sensitive")).toBeInTheDocument();
});

it("rejects oversized and unsupported files without running the model", async () => {
  const big = png("big.png");
  Object.defineProperty(big, "size", { value: 33 << 20 });
  await upload(big, new File(["x"], "a.png", { type: "image/heic" }));
  expect(await screen.findAllByText("Larger than 32 MB")).not.toHaveLength(0);
  expect(engine.calls).toBe(0);
});

it("offers a retry after an inference failure and recovers", async () => {
  engine.answers.push(
    async () => {
      throw new TaggerError("GPU error", "inference_failed");
    },
    async () => RESULT,
  );
  await upload(png());
  const retry = await screen.findByRole("button", { name: "Retry" });
  expect(screen.getByRole("alert")).toHaveTextContent("GPU error");
  await userEvent.click(retry);
  expect(await screen.findByRole("link", { name: "Kafka" })).toBeInTheDocument();
});

it("shows unreadable images without a retry button", async () => {
  engine.answers.push(async () => {
    throw new TaggerError("Unsupported or damaged image", "unreadable_image");
  });
  await upload(png());
  expect(await screen.findByRole("alert")).toHaveTextContent("Unsupported or damaged image");
  expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
});

it("shows the download and runs queued images once the model is loaded", async () => {
  const loaded = Promise.withResolvers<undefined>();
  engine.ready = loaded.promise;
  engine.answers.push(async () => RESULT);
  await upload(png());
  act(() => engine.onProgress?.(50 << 20, 178 << 20));
  expect(screen.getByText(/Downloading the model, 50 MB of 178 MB/)).toBeInTheDocument();
  expect(engine.calls).toBe(0);
  await act(async () => loaded.resolve(undefined));
  expect(await screen.findByRole("link", { name: "Kafka" })).toBeInTheDocument();
});

it("says why when the browser cannot run the model", async () => {
  engine.gpu = false;
  render(<App />);
  expect(await screen.findByRole("alert")).toHaveTextContent("This browser has no WebGPU");
});
