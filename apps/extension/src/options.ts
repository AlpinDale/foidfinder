// Settings page: grants or revokes optional access to all sites, and (on Firefox) toggles keeping the model loaded.

import { ALL_SITES } from "./messages";

const KEEP_ALIVE_KEY = "keepAlive";

const status = document.getElementById("status");
const toggle = document.getElementById("toggle");
let granted = false;

async function renderPermissions() {
  granted = await chrome.permissions.contains(ALL_SITES);
  if (status) status.textContent = granted ? "On: images are downloaded in full quality." : "Off: images from other sites may be cropped from a screenshot.";
  if (toggle) toggle.textContent = granted ? "Turn off" : "Allow access to all sites";
}

// permissions.request needs the user's click, so it is called synchronously in the handler.
toggle?.addEventListener("click", () => void (granted ? chrome.permissions.remove(ALL_SITES) : chrome.permissions.request(ALL_SITES)));
chrome.permissions.onAdded.addListener(() => void renderPermissions());
chrome.permissions.onRemoved.addListener(() => void renderPermissions());
void renderPermissions();

if (__FIREFOX__) {
  const section = document.getElementById("keepalive-section");
  const keepAliveStatus = document.getElementById("keepalive-status");
  const keepAliveToggle = document.getElementById("keepalive-toggle");
  let keepAlive = false;

  if (section) section.hidden = false;

  const renderKeepAlive = async () => {
    const stored = await chrome.storage.local.get(KEEP_ALIVE_KEY);
    keepAlive = Boolean(stored[KEEP_ALIVE_KEY]);
    if (keepAliveStatus) {
      keepAliveStatus.textContent = keepAlive
        ? "On: the model stays warm in GPU memory between identifications."
        : "Off: Firefox releases the model when idle to save memory.";
    }
    if (keepAliveToggle) keepAliveToggle.textContent = keepAlive ? "Turn off" : "Keep model loaded";
  };

  keepAliveToggle?.addEventListener("click", () => {
    void chrome.storage.local.set({ [KEEP_ALIVE_KEY]: !keepAlive });
  });
  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName === "local" && KEEP_ALIVE_KEY in changes) void renderKeepAlive();
  });
  void renderKeepAlive();
}
