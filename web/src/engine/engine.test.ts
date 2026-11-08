import { describe, expect, it } from "vitest";

import { bicubicCoeffs } from "./preprocess";
import { TagIndex } from "./tagging";

// Pillow's fixed-point BICUBIC taps as computed by foidfinder.backends.mlx.kernels._bicubic_coeffs (bit-exact with
// Image.resize): downscale, upscale, identity, odd sizes. checksum = sum of kk[i] * (i + 1) over the flat array.
const PILLOW: Record<string, { ksize: number; checksum: number; boundsSum: number; row0: number[]; first: number[]; last: number[] }> = {
  "1706->448": {
    ksize: 17,
    checksum: 7154660883105,
    boundsSum: 385548,
    row0: [877555, 1161965, 1127194, 801131, 374005, 40320, -77659, -78841, -31349, -17, 0, 0, 0, 0, 0, 0, 0],
    first: [0, 10],
    last: [1696, 10],
  },
  "300->448": { ksize: 5, checksum: 2104521410048, boundsSum: 68093, row0: [4468335, -274031, 0, 0, 0], first: [0, 2], last: [298, 2] },
  "448->448": { ksize: 5, checksum: 2103590256640, boundsSum: 101469, row0: [4194304, 0, 0, 0, 0], first: [0, 3], last: [446, 2] },
  "1001->251": {
    ksize: 17,
    checksum: 2246024643251,
    boundsSum: 127620,
    row0: [829647, 1099148, 1097249, 825398, 438801, 99162, -56028, -83202, -48887, -6985, 0, 0, 0, 0, 0, 0, 0],
    first: [0, 10],
    last: [991, 10],
  },
  "7->3": {
    ksize: 11,
    checksum: 184549362,
    boundsSum: 20,
    row0: [1606689, 1843464, 896363, 0, -135300, -16913, 0, 0, 0, 0, 0],
    first: [0, 6],
    last: [1, 6],
  },
};

describe("bicubicCoeffs", () => {
  it.each(Object.entries(PILLOW))("matches Pillow for %s", (key, want) => {
    const [from, to] = key.split("->").map(Number) as [number, number];
    const c = bicubicCoeffs(from, to);
    expect(c.ksize).toBe(want.ksize);
    expect([...c.kk.subarray(0, c.ksize)]).toEqual(want.row0);
    expect(c.kk.reduce((s, v, i) => s + v * (i + 1), 0)).toBe(want.checksum);
    expect(c.bounds.reduce((s, v) => s + v, 0)).toBe(want.boundsSum);
    expect([...c.bounds.subarray(0, 2)]).toEqual(want.first);
    expect([...c.bounds.subarray(-2)]).toEqual(want.last);
  });
});

describe("TagIndex", () => {
  const tags = ["general_a", "general_b", "char_a", "char_b", "char_c", "char_d", "char_e", "y2020", "y2021", "y2022", "safe", "explicit"];
  const categories = ["general", "general", "character", "character", "character", "character", "character", "year", "year", "year", "rating", "rating"];
  const thresholds = { general: 0.35, character: 0.35, year: 0.25, rating: 0.5 };
  const index = new TagIndex(tags, categories, thresholds);
  const probs = Float32Array.from([0.9, 0.35, 0.2, 0.6, 0.3, 0.3, 0.04, 0.3, 0.5, 0.4, 0.2, 0.7]);
  const result = index.result(probs);

  it("accepts tags at or above the threshold (float32 0.35 included, as NumPy compares), best first", () => {
    expect(result.tags.general?.map((t) => t.tag)).toEqual(["general_a", "general_b"]);
    expect(result.tags.character?.map((t) => t.tag)).toEqual(["char_b"]);
  });

  it("keeps only the best two years", () => {
    expect(result.tags.year?.map((t) => t.tag)).toEqual(["y2021", "y2022"]);
  });

  it("returns up to three below-threshold identity candidates above the floor, ties in tag order", () => {
    expect(result.candidates.character?.map((t) => t.tag)).toEqual(["char_c", "char_d", "char_a"]);
    expect(result.candidates.general).toBeUndefined();
  });

  it("reports the most likely rating", () => {
    expect(result.rating?.tag).toBe("explicit");
    expect(result.tags.rating).toBeUndefined();
  });

  it("rounds scores to four decimals", () => {
    expect(index.result(Float32Array.from([0.987654, ...probs.subarray(1)])).tags.general?.[0]?.score).toBe(0.9877);
  });
});
