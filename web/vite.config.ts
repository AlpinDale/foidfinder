import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  build: {
    // Shipped in the Python package.
    outDir: "../foidfinder/webui",
    emptyOutDir: true,
    modulePreload: { polyfill: false }, // no inline script (CSP)
  },
  server: {
    proxy: { "/api": process.env.FOIDFINDER_API ?? "http://127.0.0.1:8000" },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
  },
});
