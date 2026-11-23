// Builds the browser extension into dist/chrome/ or dist/firefox/, in two passes per browser:
// 1. background/options/(offscreen) modules, then
// 2. content.ts (--mode content / --mode firefox-content) as a standalone classic IIFE with no shared chunks.
import { execFileSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig, type Plugin } from "vite";

import { buildChromeManifest, buildFirefoxManifest, parseExtensionVersion } from "./src/manifest.ts";

const root = fileURLToPath(new URL("./src", import.meta.url));
const publicDir = fileURLToPath(new URL("./public", import.meta.url));
const chromeOutDir = fileURLToPath(new URL("./dist/chrome", import.meta.url));
const firefoxOutDir = fileURLToPath(new URL("./dist/firefox", import.meta.url));

function resolveExtensionVersion(): string | undefined {
  const env = process.env.EXTENSION_VERSION?.trim();
  if (env) return parseExtensionVersion(env);
  try {
    const tag = execFileSync("git", ["describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
    return tag ? parseExtensionVersion(tag) : undefined;
  } catch {
    return undefined;
  }
}

function manifestPlugin(manifestSrc: string, outDir: string, firefox: boolean, version?: string): Plugin {
  return {
    name: "extension-manifest",
    async closeBundle() {
      const raw = JSON.parse(await readFile(manifestSrc, "utf8")) as Record<string, unknown>;
      const manifest = firefox ? buildFirefoxManifest(raw, version) : buildChromeManifest(raw, version);
      await writeFile(join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
    },
  };
}

export default defineConfig(({ mode }) => {
  const firefox = mode === "firefox" || mode === "firefox-content";
  const content = mode === "content" || mode === "firefox-content";
  const outDir = firefox ? firefoxOutDir : chromeOutDir;
  const define = { __FIREFOX__: JSON.stringify(firefox) };

  if (content) {
    return {
      root,
      publicDir: false,
      define,
      build: {
        outDir,
        emptyOutDir: false,
        rollupOptions: {
          input: `${root}/content.ts`,
          output: { format: "iife", entryFileNames: "content.js" },
        },
      },
    };
  }

  const input: Record<string, string> = firefox
    ? { background: `${root}/background.firefox.ts`, options: `${root}/options.html` }
    : { background: `${root}/background.ts`, offscreen: `${root}/offscreen.html`, options: `${root}/options.html` };

  return {
    root,
    publicDir,
    define,
    plugins: [manifestPlugin(join(publicDir, "manifest.json"), outDir, firefox, resolveExtensionVersion())],
    build: {
      outDir,
      emptyOutDir: true,
      modulePreload: { polyfill: false }, // no inline script (extension CSP)
      rollupOptions: {
        input,
        output: {
          entryFileNames: "[name].js",
          chunkFileNames: "chunks/[name]-[hash].js",
        },
      },
    },
  };
});
