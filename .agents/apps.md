# Web UI and extension

## Shared privacy and security invariants

- Inference runs strictly on-device. Never upload user images, send tag results
  to external endpoints, or add analytics, tracking, or cookies.
- Preserve strict CSP compatibility across both `deploy/Caddyfile` and the
  extension manifest: keep `modulePreload: { polyfill: false }` in Vite
  configs, avoid inline scripts or `eval`, and keep `connect-src` restricted to
  the configured model hosts (`https://huggingface.co https://*.hf.co` unless
  explicitly overridden via `VITE_MODEL_BASE`).
- Keep `apps/web/public/privacy.html` accurate whenever network destinations,
  storage usage, or extension permissions change.

## Web app (`apps/web/`)

- `apps/web` is a purely static React app built to `apps/web/dist/`. Do not
  reintroduce server API assumptions or dev proxies.
- Manage object URLs and `AbortController` lifecycles carefully in
  `src/hooks/useTagging.ts` so removed items and unmounted components do not
  leak blobs or leave stale work running.
- Keep `src/App.tsx` focused on layout and global events (paste, drag/drop,
  keyboard navigation); keep state transitions in pure reducers (`src/queue.ts`)
  and GPU lifecycle in `src/hooks/useEngine.ts`.

## Browser extension (`apps/extension/`)

- **Permissions**: Keep `activeTab` as the default and `<all_urls>` strictly in
  `optional_host_permissions`, requested synchronously from a user click in
  `src/options.ts`. Do not add `content_scripts` to `public/manifest.json`;
  inject `content.js` on demand via `chrome.scripting.executeScript` only when
  the context menu item is clicked.
- **Shadow DOM and XSS safety**: Render result cards (`src/card.ts`) inside a
  closed Shadow DOM root (`mode: "closed"`) on a custom tag name
  (`foidfinder-cards`). Populate all tag text and error strings via
  `textContent` only—never `innerHTML`. Keep card styles in `src/card.css`
  imported via `?inline`.
- **Two-pass build**: `apps/extension/vite.config.ts` builds the service
  worker, offscreen document, and options page first, then builds `content.ts`
  in `--mode content` as a standalone classic `iife` (`content.js`) with zero
  shared chunks. Do not break that separation.
- **Networking and cleanup**: Fetch remote images in `src/offscreen.ts` with
  `credentials: "omit"` and reject HTML/JSON hotlink error pages. Scope
  `declarativeNetRequest` session Referer rules (`src/referer.ts`) to
  `initiatorDomains: [extensionId]`, remove each rule in a `finally` block, and
  clear leftover session rules on install/startup. Keep the offscreen idle
  unload timer intact so GPU memory is released when unused.
