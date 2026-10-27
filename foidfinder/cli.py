"""foidfinder command line. Every option defaults to a FOIDFINDER_* environment variable (shown in --help)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

from .tagging import TagIndex

OFFICIAL_REPO = "pixai-labs/pixai-tagger-v1.0"
MODEL_FILES = ["config.json", "model.safetensors"]
REFERENCE_FILES = ["preprocessor_config.json", "tagger_pipeline.py"]  # for the parity tests

log = logging.getLogger("foidfinder")


def env(name: str, default: str) -> str:
    return os.environ.get(f"FOIDFINDER_{name}", default)


def env_flag(name: str) -> bool:
    return env(name, "").lower() in ("1", "true", "yes")


def download(repo: str, model_dir: Path, *, reference: bool = False) -> None:
    from huggingface_hub import snapshot_download

    files = MODEL_FILES + (REFERENCE_FILES if reference else [])
    snapshot_download(repo, allow_patterns=files, local_dir=model_dir)
    log.info("downloaded %s to %s", repo, model_dir)


def load_backend(name: str, model_dir: Path):
    from .backends import load

    start = time.perf_counter()
    backend = load(name, model_dir)
    log.info("%s backend on %s loaded in %.1fs", backend.name, backend.device, time.perf_counter() - start)
    return backend


def serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .web import create_app

    if not (args.model_dir / "model.safetensors").exists():
        if not args.download_missing:
            sys.exit(f"no model in {args.model_dir}: run `foidfinder download` or set FOIDFINDER_DOWNLOAD_MISSING=1")
        download(args.repo, args.model_dir)
    app = create_app(
        lambda: load_backend(args.backend, args.model_dir),
        TagIndex.from_model_dir(args.model_dir),
        max_queue=args.max_queue,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        proxy_headers=True,
        forwarded_allow_ips=args.forwarded_allow_ips,
        server_header=False,
        timeout_graceful_shutdown=30,
    )


def tag(args: argparse.Namespace) -> None:
    from .image import ImageError, decode

    tags = TagIndex.from_model_dir(args.model_dir)
    backend = load_backend(args.backend, args.model_dir)
    for path in args.images:
        try:
            image = decode(Path(path).read_bytes())
        except (OSError, ImageError) as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            continue
        result = tags.result(backend.predict(image))
        if args.json:
            print(json.dumps({"path": path, **result}))
            continue
        print(path)
        for category in ("character", "copyright"):
            for entry in result["tags"][category]:
                print(f"  {category:10s} {entry['score']:.2f}  {entry['tag']}")
            for entry in result["candidates"][category]:
                print(f"  {category:10s} {entry['score']:.2f}  {entry['tag']}  (below threshold)")
        print(f"  general    {', '.join(e['tag'] for e in result['tags']['general'])}")


def bench(args: argparse.Namespace) -> None:
    from .image import decode

    images = [decode(Path(p).read_bytes()) for p in args.images]
    backend = load_backend(args.backend, args.model_dir)
    for image in images[:2]:
        backend.predict(image)
    times = []
    for _ in range(args.runs):
        for image in images:
            start = time.perf_counter()
            backend.predict(image)
            times.append((time.perf_counter() - start) * 1000)
    print(
        f"{backend.name} on {backend.device}: {len(times)} images, median {statistics.median(times):.1f} ms, "
        f"min {min(times):.1f} ms (preprocessing + model, decode excluded)"
    )


MODEL_CARD = """---
license: apache-2.0
base_model: {repo}
tags: [image-classification, anime, danbooru, int8]
---
# PixAI Tagger v1.0, int8 for foidfinder

[PixAI Tagger v1.0]({url}) converted for the CUDA backend of foidfinder.

Changes from the original: the four linear layers of each of the 32 transformer blocks (`attn.qkv`, `attn.proj`,
`mlp.fc1`, `mlp.fc2`) are multiplied by a Hadamard matrix (1024-point, block-128 for `fc2`) and quantized to int8
per output channel, stored as `<layer>.qweight` (int8) and `<layer>.wscale` (fp32). All other tensors and
`config.json` are unchanged. Format: `{format}`.

With foidfinder's int8 activations, compared with the fp32 original on 588 Danbooru posts from July to October 2026,
the largest probability difference is 0.077, 2 character and 9 copyright decisions change, and the top-1 character
matches on all 345 posts that name one.

```sh
FOIDFINDER_MODEL_REPO=<this repo> foidfinder download
foidfinder serve --backend cuda
```

Licensed under Apache-2.0, like the original.
"""


def convert(args: argparse.Namespace) -> None:
    import shutil

    import torch
    from safetensors.torch import load_file, save_file

    from .backends.cuda.engine import FORMAT, convert_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = convert_checkpoint(load_file(str(args.model_dir / "model.safetensors"), device=device))
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous().cpu() for k, v in out.items()}, str(args.output / "model.safetensors"), {"format": FORMAT}
    )
    shutil.copy(args.model_dir / "config.json", args.output / "config.json")
    card = MODEL_CARD.format(repo=OFFICIAL_REPO, url=f"https://huggingface.co/{OFFICIAL_REPO}", format=FORMAT)
    (args.output / "README.md").write_text(card)
    size = (args.output / "model.safetensors").stat().st_size / 2**30
    log.info("wrote %s (%.2f GiB of weights); upload with `hf upload <repo> %s`", args.output, size, args.output)


def build_kernels(args: argparse.Namespace) -> None:
    from .backends.cuda.kernels import NAME, build

    build(args.build_dir, args.archs)
    log.info("built %s", args.build_dir / f"{NAME}.so")


def main() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))  # aligned with uvicorn's lines
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(prog="foidfinder", description="Identify anime characters and series in images.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(env("MODEL_DIR", "models/pixai-tagger-v1.0")),
        help="PixAI checkpoint directory ($FOIDFINDER_MODEL_DIR, default models/pixai-tagger-v1.0)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def with_backend(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument(
            "--backend",
            choices=["auto", "mlx", "cuda"],
            default=env("BACKEND", "auto"),
            help="compute backend ($FOIDFINDER_BACKEND, default auto: mlx on Apple Silicon, else cuda)",
        )
        return p

    repo_help = f"Hugging Face repo ($FOIDFINDER_MODEL_REPO, default {OFFICIAL_REPO})"
    p = sub.add_parser("download", help="fetch the model from Hugging Face")
    p.add_argument("--repo", default=env("MODEL_REPO", OFFICIAL_REPO), help=repo_help)
    p.add_argument("--reference", action="store_true", help="also fetch the official code (for the parity tests)")
    p.set_defaults(run=lambda a: download(a.repo, a.model_dir, reference=a.reference))

    p = with_backend(sub.add_parser("serve", help="run the web UI and HTTP API"))
    p.add_argument("--host", default=env("HOST", "127.0.0.1"), help="$FOIDFINDER_HOST, default 127.0.0.1")
    p.add_argument("--port", type=int, default=int(env("PORT", "8000")), help="$FOIDFINDER_PORT, default 8000")
    p.add_argument(
        "--max-queue",
        type=int,
        default=int(env("MAX_QUEUE", "16")),
        help="requests waiting for the model before answering 503 ($FOIDFINDER_MAX_QUEUE, default 16)",
    )
    p.add_argument(
        "--forwarded-allow-ips",
        default=env("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        help="proxies trusted for X-Forwarded-* ($FOIDFINDER_FORWARDED_ALLOW_IPS, default 127.0.0.1)",
    )
    p.add_argument(
        "--download-missing",
        action="store_true",
        default=env_flag("DOWNLOAD_MISSING"),
        help="download the model first if --model-dir is empty ($FOIDFINDER_DOWNLOAD_MISSING)",
    )
    p.add_argument("--repo", default=env("MODEL_REPO", OFFICIAL_REPO), help=repo_help)
    p.set_defaults(run=serve)

    p = with_backend(sub.add_parser("tag", help="tag image files"))
    p.add_argument("images", nargs="+")
    p.add_argument("--json", action="store_true", help="one JSON result per line")
    p.set_defaults(run=tag)

    p = with_backend(sub.add_parser("bench", help="time inference on image files"))
    p.add_argument("images", nargs="+")
    p.add_argument("--runs", type=int, default=5)
    p.set_defaults(run=bench)

    p = sub.add_parser("convert", help="write the int8 checkpoint for the CUDA backend (needs the cuda extra)")
    p.add_argument("output", type=Path)
    p.set_defaults(run=convert)

    p = sub.add_parser("build-kernels", help="compile the CUDA kernels ahead of time (for FOIDFINDER_KERNELS)")
    p.add_argument("build_dir", type=Path)
    p.add_argument("--archs", nargs="+", default=["8.9"], help="compute capabilities, e.g. 8.6 8.9 9.0")
    p.set_defaults(run=build_kernels)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
