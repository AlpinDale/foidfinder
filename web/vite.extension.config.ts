// The browser extension (src/extension) into dist-extension/ (Chrome) or dist-extension-firefox/ (Firefox), in two
// passes: the background/options/offscreen bundle, then the content script (--mode content / --mode firefox-content),
// which browsers inject as one classic script and so must not share chunks.
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig, type Plugin } from "vite";

const root = fileURLToPath(new URL("./src/extension", import.meta.url));
const chromeOutDir = fileURLToPath(new URL("./dist-extension", import.meta.url));
const firefoxOutDir = fileURLToPath(new URL("./dist-extension-firefox", import.meta.url));

function firefoxManifestPlugin(manifestSrc: string, outDir: string): Plugin {
  return {
    name: "firefox-manifest",
    async closeBundle() {
      const raw = JSON.parse(await readFile(manifestSrc, "utf8")) as Record<string, unknown>;
      delete raw.minimum_chrome_version;
      raw.background = { scripts: ["background.js"], type: "module" };
      if (Array.isArray(raw.permissions)) {
        raw.permissions = raw.permissions.filter((p) => p !== "offscreen");
      }
      raw.browser_specific_settings = {
        gecko: {
          id: "foidfinder@alpindale.dev",
          strict_min_version: "128.0",
          data_collection_permissions: {
            required: ["none"],
          },
        },
      };
      await writeFile(join(outDir, "manifest.json"), `${JSON.stringify(raw, null, 2)}\n`);
    },
  };
}

export default defineConfig(({ mode }) => {
  const firefox = mode === "firefox" || mode === "firefox-content";
  const content = mode === "content" || mode === "firefox-content";
  const outDir = firefox ? firefoxOutDir : chromeOutDir;

  if (content) {
    return {
      root,
      publicDir: false,
      build: {
        outDir,
        emptyOutDir: false,
        rollupOptions: { input: `${root}/content.ts`, output: { format: "iife", entryFileNames: "content.js" } },
      },
    };
  }

  const input: Record<string, string> = firefox
    ? { background: `${root}/background.firefox.ts`, options: `${root}/options.html` }
    : { background: `${root}/background.ts`, offscreen: `${root}/offscreen.html`, options: `${root}/options.html` };

  return {
    root,
    plugins: firefox ? [firefoxManifestPlugin(`${root}/public/manifest.json`, outDir)] : [],
    build: {
      outDir,
      emptyOutDir: true,
      modulePreload: { polyfill: false }, // no inline script (extension CSP)
      rollupOptions: {
        input,
        output: { entryFileNames: "[name].js", chunkFileNames: "chunks/[name]-[hash].js" },
      },
    },
  };
});
