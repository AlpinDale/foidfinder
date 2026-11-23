# Validation, builds, and releases

## Standard checks

Run commands from the repository root. Run focused test files before expanding
to the full suite:

```sh
npx vitest run packages/engine/src/models/registry.test.ts
npm run lint
npm test
npm run build
```

- `npm run lint` runs ESLint across all workspaces plus `tsc --noEmit` for
  `packages/engine`, `apps/web`, and `apps/extension`.
- `npm test` runs Vitest (`vitest.config.ts`, `jsdom` environment) across all
  workspaces.
- `npm run build` runs `build:web` (`apps/web/dist/`) and `build:extension`
  (`apps/extension/dist/chrome/` and `apps/extension/dist/firefox/`).

## Bundle and runtime verification

- After changes to `packages/engine` exports, `apps/extension`, or Vite
  configs, inspect the build output sizes and files for both browsers:
  - `apps/extension/dist/{chrome,firefox}/content.js` must remain a small
    standalone IIFE (~7 kB) with no shared chunk imports and no duplicate
    worker asset.
  - `apps/extension/dist/chrome/offscreen.js` and
    `apps/extension/dist/firefox/background.js` must stay lightweight, with
    Haruka and WGSL code isolated in `assets/worker-*.js`.
- Vitest runs in `jsdom` with stubbed workers and no GPU or Chrome runtime.
  Unit tests do not validate WGSL compilation, GPU numerical parity, Cache API
  downloads, or live MV3 extension flows (context menu, offscreen worker, DNR
  Referer rules, screenshot cropping). When those paths change, exercise them
  in a WebGPU-capable Chromium browser instead of stopping at unit tests and
  listing what was skipped; only report a gap when a suitable browser or GPU is
  genuinely unavailable.
- For bug fixes, add a focused regression test covering observable behavior or
  state transitions rather than asserting on source text or internal wiring.

## CI, releases, and repository hygiene

- `.github/workflows/ci.yml` runs `npm ci`, `npm run lint`, `npm test`,
  `npm run build`, Mozilla's `addons-linter` on `apps/extension/dist/firefox`,
  and packages Chrome, Firefox, and AMO source archives (attaching the Chrome
  and Firefox zips to GitHub Releases on `v*` tags). The web UI deploys to
  Cloudflare Workers Static Assets via its GitHub integration on pushes to
  `master` (configured by `wrangler.jsonc`).
- Extension version is derived from git tags (`v*`) at build time in
  `apps/extension/vite.config.ts` (overridable via `EXTENSION_VERSION`, falling
  back to `apps/extension/public/manifest.json` when no git tags exist). On
  `v*` tags, CI sets `EXTENSION_VERSION` from the pushed tag, injects it into
  both `dist/chrome/manifest.json` and `dist/firefox/manifest.json`, publishes
  a GitHub release, and submits the Chrome zip to the Chrome Web Store via API
  v2 using Google Cloud OIDC (`CWS_WIF_PROVIDER`, `CWS_SERVICE_ACCOUNT`,
  `CWS_PUBLISHER_ID`, `CWS_EXTENSION_ID` repo variables).
- Anchor root-only `.gitignore` rules with a leading `/` (for example
  `/models/`, `/.venv/`) so nested source directories are never silently
  excluded from Git.
