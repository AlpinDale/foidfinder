export const FIREFOX_ADDON_ID = "extension@foidfinder.net";
export const FIREFOX_MIN_VERSION = "142.0";

const EXTENSION_VERSION_RE = /^(?:0|[1-9]\d*)(?:\.(?:0|[1-9]\d*)){0,3}$/;

/** Normalizes and validates an extension version (strips optional leading `v`, enforces Chrome/AMO numeric format). */
export function parseExtensionVersion(raw: string): string {
  const version = raw.trim().replace(/^v/, "");
  if (!EXTENSION_VERSION_RE.test(version)) {
    throw new Error(`Invalid extension version "${raw}": expected 1 to 4 dot-separated non-negative integers (e.g. 0.3.0)`);
  }
  const parts = version.split(".").map(Number);
  if (parts.some((n) => n > 65535) || parts.every((n) => n === 0)) {
    throw new Error(`Invalid extension version "${raw}": parts must be <= 65535 and not all zero`);
  }
  return version;
}

/** Returns the Chrome MV3 manifest, overriding `version` when supplied. */
export function buildChromeManifest(chromeManifest: Record<string, unknown>, version?: string): Record<string, unknown> {
  return version ? { ...chromeManifest, version: parseExtensionVersion(version) } : { ...chromeManifest };
}

/** Transforms the Chrome MV3 manifest into the Firefox MV3 manifest (event page + gecko settings). */
export function buildFirefoxManifest(chromeManifest: Record<string, unknown>, version?: string): Record<string, unknown> {
  const manifest = buildChromeManifest(chromeManifest, version);
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
