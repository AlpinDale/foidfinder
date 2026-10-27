# foidfinder

foidfinder names the characters and series in anime images. It runs
[PixAI Tagger v1.0](https://huggingface.co/pixai-labs/pixai-tagger-v1.0) (Danbooru tags up to May 2026).

We have two compute backends at the moment:

| Backend | Hardware | Precision | Time per image |
|---|---|---|---|
| `cuda` | NVIDIA, compute capability 8.0+ | int8 weights and activations, custom kernels | 20 ms on an RTX 4090 |
| `mlx` | Apple Silicon | fp16 | 0.8-1.4 s on an M4 Pro |

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

### Behind a proxy

If the GPU host should not hold TLS or Cloudflare credentials, it can dial out to a proxy host over SSH instead of
accepting connections. Its key can open one loopback port on the proxy and nothing else.

- On the proxy, create a `foidfinder-tunnel` user with shell `/usr/sbin/nologin`, add the GPU host's public key to its
  `authorized_keys` with the prefix `restrict,port-forwarding,permitlisten="127.0.0.1:18000",command="/usr/sbin/nologin"`,
  and install `deploy/sshd-tunnel.conf` into `/etc/ssh/sshd_config.d/`. Add the site from `deploy/Caddyfile` and copy
  the built UI to `/srv/foidfinder/webui`.
- On the GPU host, put the private key `id_ed25519` and a `known_hosts` entry for the proxy in a directory. Set
  `COMPOSE_FILE=compose.yaml:deploy/tunnel.compose.yaml`, `FOIDFINDER_TUNNEL_DESTINATION=foidfinder-tunnel@<proxy>` and
  `FOIDFINDER_TUNNEL_SECRETS=<directory>` in `.env`, then run `docker compose up -d`.

The proxy sends the GPU host no client IPs, Cloudflare headers or cookies, and serves its responses as JSON under a
sandboxing CSP. A compromised GPU host can return wrong tags but cannot serve pages on your domain. It does see the
uploaded images.

## From source

```sh
uv sync --extra mlx                  # Apple Silicon
uv sync --extra cuda                 # NVIDIA, needs nvcc (CUDA 12.8+) to build the kernels on first use
(cd web && npm ci && npm run build)  # builds the UI into foidfinder/webui/

uv run foidfinder download           # official checkpoint into models/pixai-tagger-v1.0
uv run foidfinder serve              # http://127.0.0.1:8000
uv run foidfinder tag a.jpg b.png    # --json prints one result per line
uv run foidfinder bench a.jpg
```

`foidfinder convert <dir>` writes the int8 checkpoint for the CUDA backend, with `config.json` and a model card, ready
for `hf upload`. The CUDA backend also loads the official checkpoint and quantizes it at startup, which takes longer
and needs 3.3 GB of VRAM during loading instead of 0.9 GB. The MLX backend needs the official checkpoint.

## Configuration

Each option is set by an environment variable or the matching `foidfinder serve` flag.

| Variable | Default | Docker default | Meaning |
|---|---|---|---|
| `FOIDFINDER_BACKEND` | `auto` | `cuda` | `auto`, `mlx` or `cuda` |
| `FOIDFINDER_MODEL_DIR` | `models/pixai-tagger-v1.0` | `/models/pixai-tagger-v1.0` | checkpoint directory |
| `FOIDFINDER_MODEL_REPO` | `pixai-labs/pixai-tagger-v1.0` | `alpindale/PixAI-v1.0-INT8` | Hugging Face repo for `download` |
| `FOIDFINDER_DOWNLOAD_MISSING` | off | on | `serve` downloads the model when the directory is empty |
| `FOIDFINDER_HOST` | `127.0.0.1` | `0.0.0.0` | listen address |
| `FOIDFINDER_PORT` | `8000` | `8000` | listen port |
| `FOIDFINDER_MAX_QUEUE` | `16` | `16` | requests waiting for the model before the API answers 503 |
| `FOIDFINDER_FORWARDED_ALLOW_IPS` | `127.0.0.1` | `127.0.0.1` | proxies trusted for `X-Forwarded-*` |
| `FOIDFINDER_KERNELS` | unset | `/opt/kernels/pixai_kernels.so` | prebuilt kernels from `foidfinder build-kernels`; unset builds them with nvcc |

## API

| Endpoint | Response |
|---|---|
| `POST /api/tag` (image file as the body) | `rating`, `tags` by category (at PixAI's per-category thresholds, sorted by score), `candidates` (up to 3 below-threshold characters and series), `image` size, `timings_ms` |
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

The CUDA backend quantizes every transformer GEMM to int8. Activations are rotated by a Hadamard matrix and scaled per
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


## License

Apache-2.0. PixAI Tagger v1.0 is also Apache-2.0.
