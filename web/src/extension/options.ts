// Settings page: grants or revokes the optional access to all sites (full-quality image downloads).

import { ALL_SITES } from "./messages";

const status = document.getElementById("status");
const toggle = document.getElementById("toggle");
let granted = false;

async function render() {
  granted = await chrome.permissions.contains(ALL_SITES);
  if (status) status.textContent = granted ? "On: images are downloaded in full quality." : "Off: images from other sites may be cropped from a screenshot.";
  if (toggle) toggle.textContent = granted ? "Turn off" : "Allow access to all sites";
}

// permissions.request needs the user's click, so it is called synchronously in the handler.
toggle?.addEventListener("click", () => void (granted ? chrome.permissions.remove(ALL_SITES) : chrome.permissions.request(ALL_SITES)));
chrome.permissions.onAdded.addListener(() => void render());
chrome.permissions.onRemoved.addListener(() => void render());
void render();
