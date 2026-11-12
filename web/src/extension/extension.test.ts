import { describe, expect, it } from "vitest";

import type { TagResult } from "../tagger";
import { cardBody } from "./card";
import { refererRule } from "./referer";

const RESULT: TagResult = {
  rating: { tag: "rating_g", score: 0.9 },
  tags: {
    character: [{ tag: "aizawa_ema", score: 0.9993 }],
    copyright: [{ tag: "vspo!", score: 0.97 }],
    artist: [],
    general: [{ tag: "1girl", score: 0.99 }],
  },
  candidates: { character: [], copyright: [], artist: [{ tag: "yuzuriha_(nx_e78)", score: 0.23 }] },
  image: { width: 800, height: 1200 },
  timings_ms: { decode: 5, inference: 60, total: 65 },
};

const text = (nodes: HTMLElement[]) => nodes.map((n) => n.textContent).join(" | ");

describe("cardBody", () => {
  it("lists the character and series with scores and Danbooru links", () => {
    const nodes = cardBody({ state: "result", result: RESULT, screenshot: false, upgrade: false });
    const link = nodes[0]?.querySelector("a");
    expect(link?.textContent).toBe("Aizawa Ema");
    expect(link?.href).toBe("https://danbooru.donmai.us/posts?tags=aizawa_ema");
    expect(link?.target).toBe("_blank");
    expect(text(nodes)).toContain("Series");
    expect(text(nodes)).toContain("Vspo!");
    expect(text(nodes)).toContain("On your GPU in 65 ms");
  });

  it("marks below-threshold guesses as possible", () => {
    const nodes = cardBody({ state: "result", result: RESULT, screenshot: false, upgrade: false });
    expect(text(nodes)).toContain("Artist, possibly");
    expect(text(nodes)).toContain("Yuzuriha");
  });

  it("says when no character matched and omits an empty artist row", () => {
    const result = { ...RESULT, tags: { ...RESULT.tags, character: [] }, candidates: { character: [], copyright: [], artist: [] } };
    const t = text(cardBody({ state: "result", result, screenshot: true, upgrade: false }));
    expect(t).toContain("Character");
    expect(t).toContain("No match");
    expect(t).not.toContain("Artist");
    expect(t).toContain("from a screenshot of the page");
  });

  it("treats tag names as text, not markup", () => {
    const result = { ...RESULT, tags: { ...RESULT.tags, character: [{ tag: "<img_src=x_onerror=alert(1)>", score: 0.9 }] } };
    const nodes = cardBody({ state: "result", result, screenshot: false, upgrade: false });
    expect(nodes[0]?.querySelector("img")).toBeNull();
    expect(nodes[0]?.querySelector("a")?.textContent).toBe("<img Src=x Onerror=alert(1)>");
  });

  it("offers full-quality downloads only when they would help", () => {
    let opened = 0;
    const onUpgrade = () => (opened += 1);
    expect(text(cardBody({ state: "result", result: RESULT, screenshot: false, upgrade: false }, onUpgrade))).not.toContain("Allow");
    const nodes = cardBody({ state: "error", message: "The image could not be downloaded", upgrade: true }, onUpgrade);
    const button = nodes.find((n) => n.tagName === "BUTTON");
    expect(button?.textContent).toBe("Allow full-quality downloads from all sites");
    button?.click();
    expect(opened).toBe(1);
  });

  it("shows download progress, then loading once the download is complete", () => {
    expect(text(cardBody({ state: "download", loaded: 50 << 20, total: 178 << 20 }))).toBe(
      "Downloading the model, 50 MB of 178 MB. This happens once. | ",
    );
    expect(text(cardBody({ state: "download", loaded: 178 << 20, total: 178 << 20 }))).toBe("Loading the model…");
  });
});

describe("refererRule", () => {
  it("sends the page's origin as Referer to the image's host, for the extension's own requests only", () => {
    const rule = refererRule(7, "https://i.pximg.net/img-original/1.png", "https://www.pixiv.net/artworks/1?x=y", "abc");
    expect(rule?.id).toBe(7);
    expect(rule?.action.requestHeaders?.[0]).toEqual({ header: "referer", operation: "set", value: "https://www.pixiv.net/" });
    expect(rule?.condition).toEqual({ requestDomains: ["i.pximg.net"], initiatorDomains: ["abc"], resourceTypes: ["xmlhttprequest"] });
  });

  it("does nothing for data: images or non-web pages", () => {
    expect(refererRule(1, "data:image/png;base64,AAAA", "https://example.com/", "abc")).toBeNull();
    expect(refererRule(1, "https://example.com/a.png", "file:///tmp/page.html", "abc")).toBeNull();
    expect(refererRule(1, "not a url", "https://example.com/", "abc")).toBeNull();
  });
});
