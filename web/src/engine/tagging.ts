// foidfinder/tagging.py for probabilities computed in the browser.

import type { Tag, TagResult } from "../tagger";
import { at } from "./at";

const CANDIDATES = 3;
// Probabilities are float32; like NumPy, compare them with the cutoffs at float32 precision.
const CANDIDATE_FLOOR = Math.fround(0.05);
const IDENTITY: Record<string, true> = { character: true, copyright: true, artist: true };
const LIMIT: Record<string, number> = { year: 2 };

export class TagIndex {
  private index = new Map<string, number[]>();

  constructor(
    private tags: string[],
    categories: string[],
    private thresholds: Record<string, number>,
  ) {
    if (tags.length !== categories.length) throw new Error("tag categories do not cover the tag list");
    categories.forEach((c, i) => {
      const list = this.index.get(c);
      if (list) list.push(i);
      else this.index.set(c, [i]);
    });
  }

  private entries(probs: Float32Array, ids: number[]): Tag[] {
    return ids.map((j) => ({ tag: at(this.tags, j), score: Math.round(at(probs, j) * 1e4) / 1e4 }));
  }

  result(probs: Float32Array): Pick<TagResult, "rating" | "tags" | "candidates"> {
    const tags: Record<string, Tag[]> = {};
    const candidates: Record<string, Tag[]> = {};
    let rating: Tag | null = null;
    const p = (j: number) => at(probs, j);
    const byScore = (a: number, b: number) => p(b) - p(a) || a - b;
    for (const [c, idx] of this.index) {
      if (c === "rating") {
        const j = idx.reduce((best, i) => (p(i) > p(best) ? i : best), at(idx, 0));
        rating = at(this.entries(probs, [j]), 0);
        continue;
      }
      const cutoff = this.thresholds[c];
      if (cutoff === undefined) throw new Error(`no threshold for category ${c}`);
      const th = Math.fround(cutoff);
      const accepted = idx.filter((j) => p(j) >= th).sort(byScore);
      tags[c] = this.entries(probs, accepted.slice(0, LIMIT[c] ?? accepted.length));
      if (IDENTITY[c]) {
        const rest = idx.filter((j) => p(j) < th && p(j) >= CANDIDATE_FLOOR).sort(byScore);
        candidates[c] = this.entries(probs, rest.slice(0, CANDIDATES));
      }
    }
    return { rating, tags, candidates };
  }
}
