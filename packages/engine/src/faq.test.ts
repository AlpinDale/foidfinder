import { describe, expect, it } from "vitest";

import { FAQ_URL, faqPathForError, faqUrlForError } from "./faq";

describe("faqUrlForError", () => {
  it.each([
    ["This browser has no WebGPU, which foidfinder needs to run the model", undefined, "#no-webgpu"],
    ["WebGPU is turned on, but the browser offers no GPU to it", undefined, "#no-gpu"],
    ["This GPU or browser lacks 16-bit floats in WebGPU shaders (shader-f16), which the model needs", undefined, "#no-f16"],
    ["GPU device lost: reset", undefined, "#device-lost"],
    ["The model did not answer", undefined, "#device-lost"],
    ["The model download failed (HTTP 503)", undefined, "#download-failed"],
    ["The model download stalled", undefined, "#download-failed"],
    ["The image could not be downloaded", "fetch_failed", "#image-fetch"],
    ["The image server sent a page instead of the image", undefined, "#image-fetch"],
    ["Larger than 32 MB", "too_large", "#image-limits"],
    ["Unsupported or damaged image", "unreadable_image", "#image-limits"],
    ["Wider or taller than 8192 pixels", "too_many_pixels", "#image-limits"],
    ["Something unexpected happened", undefined, ""],
  ])("maps %s to %s", (message, code, suffix) => {
    expect(faqUrlForError(message, code)).toBe(`${FAQ_URL}${suffix}`);
    expect(faqPathForError(message, code)).toBe(`/faq${suffix}`);
  });
});
