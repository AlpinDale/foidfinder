// What the in-browser engine (engine/) produces and the UI shows.

export interface Tag {
  tag: string;
  score: number;
}

export interface TagResult {
  rating: Tag | null;
  /** Accepted tags per category (Haruka: character, copyright, artist, general, year). */
  tags: Record<string, Tag[]>;
  /** Best below-threshold guesses for character, copyright and artist. */
  candidates: Record<string, Tag[]>;
  image: { width: number; height: number };
  timings_ms: { decode: number; inference: number; total: number };
}

/** Image types the browser decodes for the engine. */
export const FORMATS = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/avif", "image/bmp"];
export const MAX_BYTES = 32 << 20;
export const MAX_PIXELS = 64_000_000;

export class TaggerError extends Error {
  /** unreadable_image, too_many_pixels, inference_failed or unavailable (the engine stopped). */
  readonly code: string;

  constructor(message: string, code: string) {
    super(message);
    this.name = "TaggerError";
    this.code = code;
  }

  /** Whether trying the same image again can help. */
  get retryable(): boolean {
    return this.code === "inference_failed";
  }
}

export interface Tagger {
  tag(file: Blob, signal?: AbortSignal): Promise<TagResult>;
}
