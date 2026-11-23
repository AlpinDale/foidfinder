# Engine and WebGPU

## Module boundaries

All paths below are relative to `packages/engine/src/`:

- `types.ts`, `tags.ts`, `tagging.ts` own the shared tagging contract, tag
  formatting, and float32 probability thresholding (`TagIndex`).
- `client.ts` (`LocalTagger`, `webgpuAvailable`) and `worker.ts` manage the
  Web Worker lifecycle and image decoding. Both must remain model-agnostic.
- `webgpu/` holds reusable GPU primitives: device negotiation (`device.ts`),
  cached weight downloads and safetensors/gzip helpers (`weights.ts`),
  Pillow-exact bicubic planning (`bicubic.ts`), and the GPU resampler
  (`preprocess.ts`).
- `models/types.ts` and `models/registry.ts` define `ModelDefinition`,
  `LoadedModel`, `ModelSource`, and `ModelRegistry`.
- `models/haruka/` implements the built-in Haruka Tagger (`config.ts`,
  `loader.ts`, `model.ts`, `index.ts`) and its modular WGSL generators under
  `models/haruka/wgsl/`.

## Adding models

New model architectures implement `ModelDefinition` and `LoadedModel`, and
register with `defaultRegistry` in `models/registry.ts`. Architecture selection
happens from `config.json` (`matchesConfig`) or an explicit `source.modelId` so
`worker.ts`, `client.ts`, `apps/web`, and `apps/extension` need no
model-specific logic.

## Tree-shaking and non-WebGPU safety

- Keep `"sideEffects": false` in `packages/engine/package.json` and wrap
  module-level factory initializers such as `defaultRegistry` in `/*#__PURE__*/`
  helpers. `client.ts` must not import concrete models or `defaultRegistry`;
  model code and WGSL shaders belong strictly inside the Web Worker chunk.
- Never access WebGPU globals (`GPUBufferUsage`, `GPUTextureUsage`,
  `GPUMapMode`, `GPUShaderStage`) at module top level. Importing
  `@foidfinder/engine` must work safely in jsdom and browsers without WebGPU so
  UI code can render feature-error states cleanly.

## Numerics, shaders, and checkpoints

- Keep `webgpu/bicubic.ts` and `webgpu/preprocess.ts` bit-exact with Pillow's
  `BICUBIC` resize and alpha-over-white compositing. Preserve the fixed-point
  checksum tests in `webgpu/bicubic.test.ts`.
- Keep `TagIndex` threshold comparisons in float32 (`Math.fround`) with stable
  index-order tie-breaking.
- In `models/haruka/wgsl/matmul.ts`, preserve f16 tile accumulation with
  periodic flushes to f32 (`flushK = 8`). In `attention.ts`, keep both the
  16-lane subgroup path (`subgroups16`) and the workgroup-memory fallback path
  working.
- Model download URLs cached via the Cache API (`webgpu/weights.ts`) must point
  to immutable pinned revisions (such as commit hashes) so cached files always
  match their URLs.
- Always release temporary GPU textures/buffers (`destroy()`) and unmap
  readback buffers on every inference pass, including error paths.
