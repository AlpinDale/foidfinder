import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [
    react(),
    {
      name: "clean-static-urls",
      configureServer(server) {
        server.middlewares.use((req, _res, next) => {
          if (req.url === "/faq" || req.url === "/privacy") req.url += ".html";
          next();
        });
      },
    },
  ],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    modulePreload: { polyfill: false }, // no inline script (CSP)
  },
});
