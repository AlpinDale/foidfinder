export const FIREFOX_ADDON_ID = "extension@foidfinder.net";
export const FIREFOX_MIN_VERSION = "142.0";

/** Transforms the Chrome MV3 manifest into the Firefox MV3 manifest (event page + gecko settings). */
export function buildFirefoxManifest(chromeManifest: Record<string, unknown>): Record<string, unknown> {
  const manifest: Record<string, unknown> = { ...chromeManifest };
  delete manifest.minimum_chrome_version;
  manifest.background = { scripts: ["background.js"], type: "module" };
  if (Array.isArray(manifest.permissions)) {
    const perms = manifest.permissions.filter((p) => p !== "offscreen");
    if (!perms.includes("storage")) perms.push("storage");
    manifest.permissions = perms;
  }
  manifest.browser_specific_settings = {
    gecko: {
      id: FIREFOX_ADDON_ID,
      strict_min_version: FIREFOX_MIN_VERSION,
      data_collection_permissions: {
        required: ["none"],
      },
    },
  };
  return manifest;
}
