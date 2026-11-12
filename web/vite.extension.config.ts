// The browser extension (src/extension) into dist-extension/, in two passes: `vite build -c vite.extension.config.ts`
// for the service worker and the offscreen document, then `--mode content` for the content script, which Chrome
// injects as one classic script and so must not share chunks.
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

const root = fileURLToPath(new URL("./src/extension", import.meta.url));
const outDir = fileURLToPath(new URL("./dist-extension", import.meta.url));

export default defineConfig(({ mode }) =>
  mode === "content"
    ? {
        root,
        publicDir: false,
        build: {
          outDir,
          emptyOutDir: false,
          rollupOptions: { input: `${root}/content.ts`, output: { format: "iife", entryFileNames: "content.js" } },
        },
      }
    : {
        root,
        build: {
          outDir,
          emptyOutDir: true,
          modulePreload: { polyfill: false }, // no inline script (extension CSP)
          rollupOptions: {
            input: { background: `${root}/background.ts`, offscreen: `${root}/offscreen.html`, options: `${root}/options.html` },
            output: { entryFileNames: "[name].js", chunkFileNames: "chunks/[name]-[hash].js" },
          },
        },
      },
);
