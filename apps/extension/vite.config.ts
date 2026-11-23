// Built in two passes: `vite build` for the service worker, offscreen document and options page, then
// `vite build --mode content` for the content script, which Chrome injects as one classic script with no shared chunks.
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

const root = fileURLToPath(new URL("./src", import.meta.url));
const publicDir = fileURLToPath(new URL("./public", import.meta.url));
const outDir = fileURLToPath(new URL("./dist", import.meta.url));

export default defineConfig(({ mode }) =>
  mode === "content"
    ? {
        root,
        publicDir: false,
        build: {
          outDir,
          emptyOutDir: false,
          rollupOptions: {
            input: `${root}/content.ts`,
            output: { format: "iife", entryFileNames: "content.js" },
          },
        },
      }
    : {
        root,
        publicDir,
        build: {
          outDir,
          emptyOutDir: true,
          modulePreload: { polyfill: false }, // no inline script (extension CSP)
          rollupOptions: {
            input: {
              background: `${root}/background.ts`,
              offscreen: `${root}/offscreen.html`,
              options: `${root}/options.html`,
            },
            output: {
              entryFileNames: "[name].js",
              chunkFileNames: "chunks/[name]-[hash].js",
            },
          },
        },
      },
);
