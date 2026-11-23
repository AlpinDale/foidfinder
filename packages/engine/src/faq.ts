/** Public FAQ URL and error-to-anchor mapping shared by the web app and browser extension. */

export const FAQ_URL = "https://foidfinder.net/faq";

const ERROR_ANCHORS: ReadonlyArray<[pattern: RegExp, anchor: string]> = [
  [/has no WebGPU/i, "no-webgpu"],
  [/offers no GPU/i, "no-gpu"],
  [/shader-f16|16-bit floats/i, "no-f16"],
  [/GPU device lost|failed to start|Unreadable message|did not answer|model is not loaded/i, "device-lost"],
  [/model download/i, "download-failed"],
  [/could not be downloaded|server refused|sent a page|could not be read/i, "image-fetch"],
  [/Larger than|Wider or taller|Unsupported|damaged|Empty file/i, "image-limits"],
];

/** Returns the FAQ fragment id matching an error message or code, or undefined for the page top. */
export function faqAnchorForError(message: string, code?: string): string | undefined {
  if (code === "fetch_failed") return "image-fetch";
  if (code === "too_large" || code === "too_many_pixels" || code === "unreadable_image") return "image-limits";
  for (const [pattern, anchor] of ERROR_ANCHORS) {
    if (pattern.test(message)) return anchor;
  }
  return undefined;
}

/** Same-origin FAQ path (for the web app), with the matching section anchor when known. */
export function faqPathForError(message: string, code?: string): string {
  const anchor = faqAnchorForError(message, code);
  return anchor ? `/faq#${anchor}` : "/faq";
}

/** Full FAQ URL (for the browser extension), with the matching section anchor when known. */
export function faqUrlForError(message: string, code?: string): string {
  const anchor = faqAnchorForError(message, code);
  return anchor ? `${FAQ_URL}#${anchor}` : FAQ_URL;
}
