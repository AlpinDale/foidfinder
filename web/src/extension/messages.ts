// Messages between the extension's service worker (background.ts), its offscreen document (offscreen.ts), which
// runs the model, and the script it injects into pages (content.ts).

import type { TagResult } from "../tagger";

/** Optional: lets the extension download images from any site, instead of only what activeTab allows. */
export const ALL_SITES = { origins: ["<all_urls>"] };

/** Where the offscreen document gets the image: a URL (http(s) or a data: URL read from the page), or a
 * screenshot of the visible tab and the image's rectangle in it, in screenshot pixels. */
export type ImageSource = { kind: "url"; url: string } | { kind: "screenshot"; dataUrl: string; rect: Rect };

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** "fetch_failed" means the source gave no image bytes, and another source may work. */
export type Failure = { code: string; message: string };

export type IdentifyReply = { ok: true; result: TagResult } | { ok: false; error: Failure };

export type ToOffscreen = { target: "offscreen"; type: "identify"; image: ImageSource };

export type ToBackground =
  | { target: "background"; type: "progress"; loaded: number; total: number }
  /** The model has been unused for a while; the offscreen document can be closed to free GPU memory. */
  | { target: "background"; type: "idle" }
  /** From a card: open the options page, where the user can allow downloads from all sites. */
  | { target: "background"; type: "open-options" };

/** What a result card shows. */
export type CardState =
  | { state: "loading" }
  | { state: "download"; loaded: number; total: number }
  // `upgrade`: the image could not be downloaded, and allowing downloads from all sites would help.
  | { state: "result"; result: TagResult; screenshot: boolean; upgrade: boolean }
  | { state: "error"; message: string; upgrade: boolean };

export type ToContent =
  | { type: "ff:card"; id: number; image: string; card: CardState }
  /** Read the image from inside the page (blob: URLs, same-origin or CORS images). */
  | { type: "ff:read"; url: string }
  /** The image's rectangle in the viewport, in CSS pixels, and the device pixel ratio. */
  | { type: "ff:locate"; url: string };

/** The image as a data: URL, or null when the page cannot read it either. */
export type ReadReply = string | null;
export type LocateReply = { rect: Rect; dpr: number } | null;
