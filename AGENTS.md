# Working on foidfinder

foidfinder identifies anime characters, series, artists, and tags locally in the
browser with WebGPU. Images stay on the user's device; there is no inference
server. The repository is an npm-workspaces TypeScript monorepo with a shared
engine (`packages/engine`), a static web app (`apps/web`), and a Chromium MV3
extension (`apps/extension`).

## Read for the current task

Read [README.md](README.md) for setup and supported browsers. Load only the
relevant guides below, plus any nested `AGENTS.md` covering files you change.

- [Engine and WebGPU](.agents/engine.md) for `packages/engine/`, WGSL shaders,
  preprocessing, weight loading, or adding model architectures.
- [Web UI and extension](.agents/apps.md) for `apps/web/`, `apps/extension/`,
  CSP/privacy constraints, or `deploy/Caddyfile`.
- [Validation and releases](.agents/validation.md) for choosing checks,
  verifying bundle outputs, or changing `.github/workflows/`.

## Working rules

- Trace the affected execution path before editing. Reuse existing utilities
  and conventions. Address demonstrated failures rather than speculative ones.
- Make the smallest complete change. Keep unrelated refactors and formatting
  out of the patch. New abstractions or dependencies need a concrete consumer.
- Keep modules small and focused, respecting workspace boundaries:
  `packages/engine` has no React or Chrome dependencies, `apps/web` has no
  Chrome dependencies, and `apps/extension` has no React dependencies.
- Preserve user changes. Keep experiments and downloaded model weights outside
  the tracked source tree. Check free storage before downloading checkpoints.
- Use `npm` at the repository root for dependencies, linting, tests, and
  builds. Reuse the existing Node toolchain without modifying global installs.
- Update affected callers when an interface changes. Preserve serialized
  checkpoint formats (`foidfinder-haruka-rowstep1`) and `config.json` contracts
  unless the task explicitly requires a coordinated change.
- Comments should explain constraints or non-obvious decisions. Keep task
  history and conversations out of code and user documentation.
- Keep documentation concise and factual. Do not use punchy slogans or
  promotional language.
- Update documentation where users need it to understand changed usage. Small
  features do not automatically need a README headline, new section, or feature
  announcement. Prefer a short update in the existing relevant section.
- Do not commit superfluous documentation or scatter unnecessary Markdown files
  through the repository. New documents need a clear purpose beyond recording
  the current task. Link to authoritative code and configuration instead of
  duplicating details.
- Do not commit build outputs (`dist/`), extension `.zip` archives, model
  weights, or profiling dumps. Intentional tracked assets are limited to
  extension icons, static web assets, and explicitly requested docs media.
- Run every test, build, or measurement needed to validate the change instead
  of skipping it and adding a caveat that it was not run. Only report a
  browser, GPU, or environment gap when the required runtime or hardware is
  genuinely unavailable. Report the checks actually run and their results, and
  do not present inferred correctness or speedups as measured.
- Do not write the PR description yourself. Ask the user to write it out after
  giving them sufficient information about the changes. If the user still asks
  you to write the description yourself, give them a warning but proceed with
  noting this override in the PR description.
- Do not reply to the reviewer's comment yourself. Same rule as above.

## Review and publication

Do not commit, push, tag, publish packages, or post GitHub comments unless
asked. When asked to commit, use a conventional commit subject without an
extended body. Preserve licensing and author attribution. Do not invent
co-author credits.

Keep this file short. Put task-specific detail in the linked guides and update
it when the implementation changes.
