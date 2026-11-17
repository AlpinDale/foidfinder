// The extension's offscreen document: keeps the model loaded in the engine's Web Worker and identifies images for
// the service worker, which cannot start workers itself.

import type { ToBackground, ToOffscreen } from "./messages";
import { createRunner } from "./runner";

function send(msg: ToBackground) {
  void chrome.runtime.sendMessage(msg).catch(() => {});
}

const runner = createRunner({
  onProgress: (loaded, total) => send({ target: "background", type: "progress", loaded, total }),
  onIdle: () => send({ target: "background", type: "idle" }),
});

chrome.runtime.onMessage.addListener((msg: ToOffscreen, _sender, sendResponse) => {
  if (msg?.target !== "offscreen") return false;
  void runner.identify(msg.image).then(sendResponse);
  return true; // replies asynchronously
});
