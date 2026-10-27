import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { App } from "./App";

const INFO = {
  model: "PixAI Tagger v1.0",
  backend: "mlx",
  device: "test",
  max_upload_bytes: 1024,
  max_pixels: 64_000_000,
  formats: ["image/jpeg", "image/png"],
};

const RESULT = {
  rating: { tag: "rating:g", score: 0.9 },
  tags: {
    character: [{ tag: "kafka_(honkai:_star_rail)", score: 0.97 }],
    copyright: [{ tag: "honkai:_star_rail", score: 0.99 }],
    general: [{ tag: "1girl", score: 0.99 }],
    style: [],
    meta: [],
  },
  candidates: { character: [{ tag: "himeko_(honkai:_star_rail)", score: 0.12 }], copyright: [] },
  image: { width: 800, height: 1200 },
  timings_ms: { decode: 3, inference: 20, total: 25 },
};

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

let tagResponses: (() => Response | Promise<Response>)[];
let tagCalls: number;

beforeEach(() => {
  tagResponses = [];
  tagCalls = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (url === "/api/info") return json(INFO);
      tagCalls += 1;
      const next = tagResponses.shift();
      if (!next) throw new Error("unexpected /api/tag call");
      return next();
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

const png = (name = "a.png", size = 10) => new File([new Uint8Array(size)], name, { type: "image/png" });

async function upload(...files: File[]) {
  render(<App />);
  await waitFor(() => expect(fetch).toHaveBeenCalledWith("/api/info", expect.anything()));
  await userEvent.upload(screen.getByTestId("file-input"), files);
}

it("shows the identified character, series and closest guess", async () => {
  tagResponses.push(() => json(RESULT));
  await upload(png());
  expect(await screen.findByRole("link", { name: "Kafka" })).toHaveAttribute(
    "href",
    "https://danbooru.donmai.us/posts?tags=kafka_(honkai%3A_star_rail)",
  );
  expect(screen.getByRole("link", { name: "Himeko" })).toBeInTheDocument();
  expect(screen.getByText("800 × 1200")).toBeInTheDocument();
  expect(screen.getByText("Rated General")).toBeInTheDocument();
});

it("rejects oversized and unsupported files without uploading them", async () => {
  await upload(png("big.png", 2048), new File(["x"], "a.png", { type: "image/heic" }));
  expect(await screen.findAllByText("Larger than 1 KB")).not.toHaveLength(0);
  expect(tagCalls).toBe(0);
});

it("offers a retry after a network failure and recovers", async () => {
  tagResponses.push(() => Promise.reject(new TypeError("Failed to fetch")), () => json(RESULT));
  await upload(png());
  const retry = await screen.findByRole("button", { name: "Retry" });
  expect(screen.getByRole("alert")).toHaveTextContent("Can't reach the server");
  await userEvent.click(retry);
  expect(await screen.findByRole("link", { name: "Kafka" })).toBeInTheDocument();
});

it("shows server rejections without a retry button", async () => {
  tagResponses.push(() => json({ error: { code: "unsupported_format", message: "Unsupported image format" } }, 415));
  await upload(png());
  expect(await screen.findByRole("alert")).toHaveTextContent("Unsupported image format");
  expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
});

it("waits and retries by itself while the server is busy", async () => {
  tagResponses.push(() => json({ error: { code: "busy", message: "Server busy" } }, 503, { "Retry-After": "0.01" }));
  tagResponses.push(() => json(RESULT));
  await upload(png());
  expect(await screen.findByRole("link", { name: "Kafka" })).toBeInTheDocument();
  expect(tagCalls).toBe(2);
});
