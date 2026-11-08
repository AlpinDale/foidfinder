import type { Tag, TagResult } from "./tagger";

/** "kafka_(honkai:_star_rail)" -> { name: "Kafka", qualifier: "Honkai: Star Rail" } */
export function splitTag(tag: string): { name: string; qualifier: string } {
  const text = tag.replace(/_/g, " ").trim();
  const match = /^(.*\S)\s*\(([^()]+)\)$/.exec(text);
  return match ? { name: titleCase(match[1] ?? text), qualifier: titleCase(match[2] ?? "") } : { name: titleCase(text), qualifier: "" };
}

/** Danbooru tags are lowercase; capitalize each word. */
export function titleCase(text: string): string {
  return text.replace(/(^|[\s\-/])(\p{Ll})/gu, (_, sep: string, ch: string) => sep + ch.toUpperCase());
}

export function displayTag(tag: string): string {
  return tag.replace(/_/g, " ");
}

export const RATINGS: Record<string, string> = {
  "rating:g": "General",
  "rating:s": "Sensitive",
  "rating:q": "Questionable",
  "rating:e": "Explicit",
  rating_g: "General",
  rating_s: "Sensitive",
  rating_q: "Questionable",
  rating_e: "Explicit",
};

export function ratingLabel(rating: Tag | null): string | null {
  return rating ? (RATINGS[rating.tag] ?? displayTag(rating.tag)) : null;
}

export function percent(score: number): string {
  if (score >= 0.995) return "99%";
  if (score < 0.01) return "<1%";
  return `${Math.round(score * 100)}%`;
}

export function danbooruUrl(tag: string): string {
  return `https://danbooru.donmai.us/posts?tags=${encodeURIComponent(tag)}`;
}

/** Accepted character, series, artist and general tags, in that order, as a comma-separated list. */
export function tagsForCopy(result: TagResult): string {
  return ["character", "copyright", "artist", "general"]
    .flatMap((c) => result.tags[c] ?? [])
    .map((t) => displayTag(t.tag))
    .join(", ");
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1 << 20) return `${(bytes / (1 << 20)).toFixed(bytes >= 10 << 20 ? 0 : 1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export function formatDuration(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}
