# foidfinder

foidfinder names the characters, series and artists in anime images. It runs one of two taggers, chosen by the
model directory:

- [PixAI Tagger v1.0](https://huggingface.co/pixai-labs/pixai-tagger-v1.0): 30,877 tags, Danbooru up to May 2026.
- [Haruka Tagger](https://huggingface.co/KaraKaraWitch/haruka-tagger) (EVA02-B/14 at 448 px, 18-query tree head):
  98,782 tags, including 38,447 artists. Runs on the MLX backend, and in the browser with WebGPU.

The web UI runs Haruka in the browser with WebGPU and needs no server. The CLI and the HTTP API run on these compute
backends:

| Backend | Hardware | Precision | Time per image |
|---|---|---|---|
| `cuda` | NVIDIA, compute capability 8.0+ | int8 weights and activations, custom kernels | 20 ms on an RTX 4090 |
| `cuda` on Volta | NVIDIA V100 (compute capability 7.0) | fp16, CUDA 12.6 | 123 ms on a V100-SXM2 |
| `mlx` | Apple Silicon | fp16 | PixAI 0.8-1.4 s, Haruka 37 ms on an M4 Pro |
| WebGPU, in the web UI | browsers with WebGPU and `shader-f16` | fp16, f32 accumulation | Haruka 52 ms on an M4 Pro in Chrome |

The backend is chosen at runtime. The default, `auto`, uses MLX on Apple Silicon and CUDA when an NVIDIA GPU is
visible.

## Docker (NVIDIA)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) and a driver
for CUDA 13.0 (580 or newer).

```sh
docker compose up -d --build   # http://127.0.0.1:8000
```

On first start the container downloads the int8 checkpoint
[alpindale/PixAI-v1.0-INT8](https://huggingface.co/alpindale/PixAI-v1.0-INT8) (0.6 GB) into the `models` volume.
The kernels are compiled at build time for the `CUDA_ARCHS` build argument (default `8.9`, add `8.6` for RTX 30-series
cards). Compose publishes its port on loopback only, so put a reverse proxy
with TLS and rate limiting in front of it.

For a V100, use `Dockerfile.volta` and the official checkpoint:

```sh
FOIDFINDER_DOCKERFILE=Dockerfile.volta FOIDFINDER_IMAGE=foidfinder:volta \
FOIDFINDER_MODEL_REPO=pixai-labs/pixai-tagger-v1.0 docker compose up -d --build
```

Volta uses PyTorch 2.8 with CUDA 12.6, without custom kernels. Persist these settings in `.env` for later restarts.

## Hosting the web UI

The web UI is static files. Build it with `(cd web && npm ci && npm run build)`, copy `foidfinder/webui/` to the web
root and serve it over HTTPS (WebGPU and the Cache API need a secure context). `deploy/Caddyfile` serves it from
`/srv/foidfinder/webui` with a CSP whose `connect-src` allows the model download from Hugging Face.

## From source

```sh
uv sync --extra mlx                  # Apple Silicon
uv sync --extra cuda                 # NVIDIA, needs nvcc (CUDA 12.8+) to build the kernels on first use
uv sync --extra volta                # V100, pinned PyTorch 2.8/CUDA 12.6; no compiler needed
(cd web && npm ci && npm run build)  # builds the UI into foidfinder/webui/

uv run foidfinder download           # official checkpoint into models/pixai-tagger-v1.0
uv run foidfinder serve              # http://127.0.0.1:8000
uv run foidfinder tag a.jpg b.png    # --json prints one result per line
uv run foidfinder bench a.jpg
```

Convert Haruka once. The converter downloads it from Hugging Face, or reads a local copy, and writes a 386 MB fp16
model.

```sh
uv run foidfinder convert-haruka models/haruka
uv run foidfinder --model-dir models/haruka serve
```

`foidfinder quantize-haruka <source> <dir> --calib <images>` writes a 187 MB checkpoint for downloading into a
browser (see [Haruka for the browser](#haruka-for-the-browser)). It calibrates on the given images, about 1,500 like
the expected uploads, in under 3 minutes on an M4 Pro. The MLX backend loads it too.

`foidfinder convert <dir>` writes the int8 checkpoint for the CUDA backend, with `config.json` and a model card, ready
for `hf upload`. On compute capability 8.0+, the CUDA backend can quantize the official checkpoint at startup,
which needs 3.3 GB of VRAM during loading instead of 0.9 GB. The MLX and Volta paths need the official checkpoint.

## Configuration

Each option is set by an environment variable or the matching `foidfinder serve` flag.

| Variable | Default | Docker default | Meaning |
|---|---|---|---|
| `FOIDFINDER_BACKEND` | `auto` | `cuda` | `auto`, `mlx` or `cuda` |
| `FOIDFINDER_MODEL_DIR` | `models/pixai-tagger-v1.0` | `/models/pixai-tagger-v1.0` | model directory (PixAI, or Haruka from `convert-haruka`) |
| `FOIDFINDER_MODEL_REPO` | `pixai-labs/pixai-tagger-v1.0` | `alpindale/PixAI-v1.0-INT8` (Volta: official) | Hugging Face repo for `download` |
| `FOIDFINDER_DOWNLOAD_MISSING` | off | on | `serve` downloads the model when the directory is empty |
| `FOIDFINDER_HOST` | `127.0.0.1` | `0.0.0.0` | listen address |
| `FOIDFINDER_PORT` | `8000` | `8000` | listen port |
| `FOIDFINDER_MAX_QUEUE` | `16` | `16` | requests waiting for the model before the API answers 503 |
| `FOIDFINDER_FORWARDED_ALLOW_IPS` | `127.0.0.1` | `127.0.0.1` | proxies trusted for `X-Forwarded-*` |
| `FOIDFINDER_KERNELS` | unset | `/opt/kernels/pixai_kernels.so` (Volta: unset) | prebuilt INT8 kernels from `foidfinder build-kernels` |

## API

| Endpoint | Response |
|---|---|
| `POST /api/tag` (image file as the body) | `rating`, `tags` by category (at the model's per-category thresholds, sorted by score), `candidates` (up to 3 below-threshold characters, series and, for Haruka, artists), `image` size, `timings_ms` |
| `GET /api/info` | model, backend, device, upload limits |
| `GET /api/health` | `{"status": "ok"}` once the model is loaded |

Errors have the form `{"error": {"code": ..., "message": ...}}`.

| Status | Cause |
|---|---|
| 400 | empty body |
| 413 | file over 32 MB or image over 64 megapixels |
| 415 | not an image |
| 422 | damaged image |
| 503 | queue full; retry after `Retry-After` seconds |

Uploads are decoded in memory and never written to disk.

## Accuracy and speed

Both backends keep the 72×72 patch tokens in window order (3×3 windows of 24×24), so windowed attention needs no
partition or merge transposes. The MLX backend matches the official implementation within 1e-5 in fp32 and 0.003 in
fp16.

The INT8 CUDA path quantizes every transformer GEMM to int8. Activations are rotated by a Hadamard matrix and scaled per
token, and weights carry the same rotation and are scaled per output channel. Attention runs on fp16 tensor cores.
Bias, RoPE, GELU and residual adds are fused into the GEMM epilogues, and each batch size runs as one CUDA graph. The GEMMs are
limited by L2 bandwidth, so batching does not lower the time per image.

| RTX 4090 | ms per image |
|---|---|
| Official code, fp32 | 168 |
| Official code, bf16, `torch.compile(mode="max-autotune")` | 43.7 |
| foidfinder `cuda` backend | 19.8 |

Agreement with the official fp32 model on 588 Danbooru posts from July to October 2026, at PixAI's thresholds
(character 0.27, copyright 0.24). The top-1 column counts the 345 posts tagged with a character.

| | max \|Δp\| | mean \|Δp\| | character decisions changed | copyright decisions changed | top-1 character |
|---|---|---|---|---|---|
| Official code, fp16 | 0.020 | 1.1e-6 | 1 | 1 | 100% |
| Official code, bf16 | 0.079 | 7.4e-6 | 1 | 4 | 99.7% |
| foidfinder `cuda` backend | 0.077 | 1.1e-5 | 2 | 9 | 100% |

### Haruka on MLX

The converter folds the input normalization into the patch embedding and the attention scales into the query and key
weights, and merges the QKV and SwiGLU projections into single matrices. Custom Metal kernels apply RoPE and the SwiGLU
gate with its LayerNorm in single passes. The bicubic resize and white padding run on the GPU, bit-exact with the
model card's Pillow preprocessing.

| M4 Pro, one image | ms |
|---|---|
| Model card code, PyTorch MPS fp32 | 55.4 |
| Model card code, PyTorch MPS fp16 | 45.1 |
| foidfinder `mlx` (preprocessing + model) | 37.2 |

The backbone's matrix multiplies run at 6.2-7.0 TFLOPS, close to MLX's 7.3 TFLOPS best case on this GPU. The tree
pool and tag head take 1.9 ms, most of it reading the 228 MB classifier. Apple GPUs have no int8 matrix units, so
INT8 weights do not help. Quantizing every linear layer made the model slower (38.4 ms) and changed 4% of artist
decisions. Quantizing only the classifier saved 0.4 ms and doubled the changed decisions.

Agreement with the fp32 model card code on 400 recent safebooru.org posts, at the card's thresholds (character,
copyright and general 0.35, artist 0.25):

| | max \|Δp\| | character | copyright | artist | general | top-1 character |
|---|---|---|---|---|---|---|
| foidfinder `mlx` fp16 | 0.008 | 2 of 577 | 1 of 717 | 0 of 304 | 38 of 20,632 | 100% of 309 |

### Haruka for the browser

A browser downloads the model before its first run, so this checkpoint is compressed for transfer. GPTQ rounds each
large matrix (the backbone linears, the pool's key/value projection and the tag classifier) to integer multiples of
one step per output row. The step is the row's RMS divided by 40 in the backbone and by 32 in the classifier, which
is about 4 times less sensitive per weight. The codes are split into byte planes and gzipped, so the browser's
built-in `DecompressionStream` unpacks them. Each weight is rebuilt once as code × step, and inference uses ordinary
fp16 matrices.

Agreement with the fp32 model on the same 400 posts:

| | size | KL per image | max \|Δp\| | character | copyright | artist | general | top-1 character |
|---|---|---|---|---|---|---|---|---|
| fp16 | 386 MB | 0.0007 | 0.008 | 2 | 1 | 0 | 33 | 100% |
| bf16, the training precision | 386 MB | 0.046 | 0.079 | 9 | 5 | 10 | 242 | 99.7% |
| `quantize-haruka` | 187 MB | 0.023 | 0.031 | 5 | 6 | 2 | 175 | 100% |

KL is summed over all tags. Integers with a scale per group of 128 weights, the usual format, need 195 MB for
the same KL (8 bits in the backbone, 7 in the classifier). Rounding to a per-row step and gzipping the codes saves
0.46 bits per weight. At 4 bits per weight the best method tried, GPTQ with Hadamard rotation, changed 2% of top-1
characters, so this model needs about 7 bits per weight to stay within bf16's deviation.

### Haruka on WebGPU

The web UI tags images in the browser and sends nothing to a server, so any static host can serve it. It downloads
[alpindale/haruka-compressed](https://huggingface.co/alpindale/haruka-compressed), pinned to one commit, keeps it in
the Cache API and runs the model in a Web Worker. Browsers without WebGPU, or without 16-bit floats in WebGPU shaders,
show an error that names the missing feature. To load the model from elsewhere, set `VITE_MODEL_BASE` when building
and allow that origin in the page's `connect-src`.

The engine (`web/src/engine/`) is the MLX forward in WGSL. Preprocessing runs on the GPU and matches foidfinder's
Pillow path bit for bit on all 64 test images, decoded by the browser. The matrix multiplications add products in
f16 and move the sums into f32 accumulators every 8 products. On Apple GPUs that reaches 4.5 TFLOPS against 2.9 for
f32 arithmetic, because f16 halves the workgroup-memory traffic that limits these kernels.

| M4 Pro, Chrome 150 | |
|---|---|
| model forward | 52 ms |
| per image in a batch, with JPEG decoding | 61 ms |
| first visit, model download | 15-19 s |
| later visits, load from the cache | 0.8 s |

Against the fp32 MLX forward of the same weights on 64 posts, the largest probability difference is 0.006 and the KL
per image is 0.001, where MLX in fp16 measures 0.0007. 3 general-tag decisions change, and no character, copyright or
artist ones.

`web/bench.html` checks and times the engine. Write its reference data with
`uv run python web/reference.py models/haruka-web <image dir>`, start `npm run dev` in `web/`, open `/bench.html` and
call `bench.load()`, then `bench.validate()` or `bench.profile()`, from the console.

## License

Apache-2.0. PixAI Tagger v1.0 is also Apache-2.0.
