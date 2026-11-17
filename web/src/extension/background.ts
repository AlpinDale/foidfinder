// Chrome extension service worker: forwards image identification to the offscreen document.

import { createBackground } from "./background-core";
import type { IdentifyReply, ImageSource, ToBackground, ToOffscreen } from "./messages";

let creating: Promise<void> | null = null;

async function ensureOffscreen() {
  const open = await chrome.runtime.getContexts({ contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT] });
  if (open.length) return;
  creating ??= chrome.offscreen
    .createDocument({
      url: "offscreen.html",
      reasons: [chrome.offscreen.Reason.WORKERS],
      justification: "Runs the image tagger in a Web Worker with WebGPU.",
    })
    .finally(() => {
      creating = null;
    });
  await creating;
}

async function offscreen(image: ImageSource): Promise<IdentifyReply> {
  await ensureOffscreen();
  const msg: ToOffscreen = { target: "offscreen", type: "identify", image };
  const reply = (await chrome.runtime.sendMessage(msg)) as IdentifyReply | undefined;
  return reply ?? { ok: false, error: { code: "inference_failed", message: "The model did not answer" } };
}

const bg = createBackground(offscreen);

chrome.runtime.onMessage.addListener((msg: ToBackground) => {
  if (msg?.target !== "background") return false;
  if (msg.type === "progress") bg.notifyProgress(msg.loaded, msg.total);
  else if (msg.type === "idle" && !bg.hasInFlight()) void chrome.offscreen.closeDocument().catch(() => {});
  return false;
});
