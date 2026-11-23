// Firefox extension background event page: runs the WebGPU tagger worker directly (Firefox lacks offscreen documents).
// Keeps the event page alive while an identification (including the initial model download) is in flight so Firefox's
// ~30s MV3 idle timeout does not kill the download mid-stream. Once idle, Firefox may suspend the event page normally
// unless the user opts in via the options page (`keepAlive` in chrome.storage.local).

import { createBackground } from "./background-core";
import { createRunner } from "./runner";

export const KEEP_ALIVE_KEY = "keepAlive";
const KEEPALIVE_MS = 15_000;

let keepAliveEnabled = false;
let active = 0;
let keepAliveTimer: ReturnType<typeof setInterval> | undefined;

function ping() {
  void chrome.runtime.getPlatformInfo().catch(() => {});
}

function startKeepAlive() {
  if (keepAliveTimer !== undefined) return;
  // Firefox suspends MV3 event pages after ~30s of inactivity, killing in-flight worker downloads and destroying the
  // GPUDevice. Calling a lightweight extension API resets the idle timer while active or while the user opted in.
  ping();
  keepAliveTimer = globalThis.setInterval(ping, KEEPALIVE_MS);
}

function stopKeepAlive() {
  globalThis.clearInterval(keepAliveTimer);
  keepAliveTimer = undefined;
}

function updateKeepAlive() {
  if (keepAliveEnabled || active > 0) startKeepAlive();
  else stopKeepAlive();
}

const runner = createRunner({
  onProgress: (loaded, total) => bg.notifyProgress(loaded, total),
  onIdle: () => {
    if (!keepAliveEnabled) runner.dispose();
  },
});

const bg = createBackground(async (image) => {
  active += 1;
  updateKeepAlive();
  try {
    return await runner.identify(image);
  } finally {
    active -= 1;
    updateKeepAlive();
  }
});

async function warmIfCached() {
  if (!keepAliveEnabled) return;
  if (!(await runner.isCached())) return;
  updateKeepAlive();
  const ok = await runner.warm();
  if (!ok && active === 0) stopKeepAlive();
}

async function applyKeepAlive(enabled: boolean) {
  keepAliveEnabled = enabled;
  if (enabled) {
    await warmIfCached();
  } else {
    updateKeepAlive();
    runner.dispose();
  }
}

async function syncKeepAlive() {
  const stored = await chrome.storage.local.get(KEEP_ALIVE_KEY);
  await applyKeepAlive(Boolean(stored[KEEP_ALIVE_KEY]));
}

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName !== "local" || !(KEEP_ALIVE_KEY in changes)) return;
  void applyKeepAlive(Boolean(changes[KEEP_ALIVE_KEY]?.newValue));
});

chrome.runtime.onStartup.addListener(() => void syncKeepAlive());
void syncKeepAlive();
