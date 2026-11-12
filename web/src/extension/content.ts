// Injected into a tab when its image is identified (never preloaded into pages). Shows the result cards in the
// top frame, and in the image's frame reads or locates the image when the extension cannot download it.

import { MAX_BYTES } from "../tagger";
import { CardStack } from "./card";
import type { LocateReply, ReadReply, ToContent } from "./messages";

declare global {
  interface Window {
    __foidfinder?: true;
  }
}

/** The image as a data: URL, read with the page's own origin (blob: URLs, same-origin and CORS images). */
async function read(url: string): Promise<ReadReply> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const blob = await res.blob();
    if (blob.size > MAX_BYTES) return null;
    const reader = new FileReader();
    const done = Promise.withResolvers<ReadReply>();
    reader.onload = () => done.resolve(typeof reader.result === "string" ? reader.result : null);
    reader.onerror = () => done.resolve(null);
    reader.readAsDataURL(blob);
    return await done.promise;
  } catch {
    return null;
  }
}

function nextFrame(): Promise<number> {
  const { promise, resolve } = Promise.withResolvers<number>();
  requestAnimationFrame(resolve);
  return promise;
}

/** The visible part of the largest on-screen <img> showing `url`, once the cards are hidden and repainted. */
async function locate(url: string, cards: CardStack): Promise<LocateReply> {
  let best: LocateReply = null;
  let bestArea = 0;
  for (const img of document.images) {
    if (img.currentSrc !== url && img.src !== url) continue;
    const r = img.getBoundingClientRect();
    const x = Math.max(0, r.left);
    const y = Math.max(0, r.top);
    const width = Math.min(window.innerWidth, r.right) - x;
    const height = Math.min(window.innerHeight, r.bottom) - y;
    if (width > 0 && height > 0 && width * height > bestArea) {
      bestArea = width * height;
      best = { rect: { x, y, width, height }, dpr: window.devicePixelRatio };
    }
  }
  if (best) {
    cards.setHidden(true);
    for (let i = 0; i < 2; i++) await nextFrame();
  }
  return best;
}

if (!window.__foidfinder) {
  window.__foidfinder = true;
  const cards = new CardStack();
  chrome.runtime.onMessage.addListener((msg: ToContent, _sender, sendResponse) => {
    switch (msg.type) {
      case "ff:card":
        cards.show(msg.id, msg.image, msg.card);
        return false;
      case "ff:read":
        void read(msg.url).then(sendResponse);
        return true;
      case "ff:locate":
        void locate(msg.url, cards).then(sendResponse);
        return true;
    }
  });
}
