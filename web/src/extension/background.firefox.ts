// Firefox extension background event page: runs the WebGPU tagger worker directly (Firefox lacks offscreen documents).
// Keeps the model and compiled WebGPU pipelines warm for the browser session (Firefox has no on-disk WebGPU shader
// cache, and suspends idle MV3 event pages after ~30s), and pre-warms from the Cache API on startup/load.

import { createBackground } from "./background-core";
import { createRunner } from "./runner";

const KEEPALIVE_MS = 20_000;
let keepAlive: ReturnType<typeof setInterval> | undefined;

function startKeepAlive() {
  if (keepAlive !== undefined) return;
  // Firefox suspends MV3 event pages after ~30s of inactivity, which destroys the GPUDevice and forces shader
  // recompilation. Calling a lightweight extension API resets the idle timer so compiled shaders stay in memory.
  keepAlive = globalThis.setInterval(() => {
    void chrome.runtime.getPlatformInfo().catch(() => {});
  }, KEEPALIVE_MS);
}

function stopKeepAlive() {
  globalThis.clearInterval(keepAlive);
  keepAlive = undefined;
}

const runner = createRunner({
  onProgress: (loaded, total) => bg.notifyProgress(loaded, total),
  // Keep the compiled pipelines alive for the session instead of releasing after 5 minutes.
  onIdle: () => {},
});

const bg = createBackground((image) => {
  startKeepAlive();
  return runner.identify(image);
});

async function warm() {
  if (!(await runner.isCached())) return;
  startKeepAlive();
  const ok = await runner.warm();
  if (!ok && !bg.hasInFlight()) stopKeepAlive();
}

chrome.runtime.onStartup.addListener(() => void warm());
void warm();
