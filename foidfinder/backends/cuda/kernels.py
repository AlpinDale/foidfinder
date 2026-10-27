"""Registers the csrc/ ops as torch.ops.foidfinder: prebuilt from $FOIDFINDER_KERNELS (`foidfinder build-kernels`),
else JIT-built with nvcc for the visible GPU."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

NAME = "pixai_kernels"
CSRC = Path(__file__).parent / "csrc"
FLAGS = ["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-lineinfo"]


def _gencode(archs: list[str]) -> list[str]:
    """["8.9", "9.0"] -> SASS for each, plus PTX of the newest so later GPUs can JIT it."""
    nums = [a.replace(".", "") for a in archs]
    flags = [f"-gencode=arch=compute_{n},code=sm_{n}" for n in nums]
    return [*flags, f"-gencode=arch=compute_{nums[-1]},code=compute_{nums[-1]}"]


def build(build_dir: Path | None, archs: list[str]) -> None:
    """Compiles csrc/ for compute capabilities such as "8.9" into build_dir and loads it."""
    from torch.utils.cpp_extension import load

    if build_dir is not None:
        build_dir.mkdir(parents=True, exist_ok=True)
    load(
        name=NAME,
        sources=[str(CSRC / "kernels.cu")],
        extra_include_paths=[str(CSRC)],
        extra_cuda_cflags=FLAGS + _gencode(sorted(archs, key=float)),
        build_directory=str(build_dir) if build_dir is not None else None,
        is_python_module=False,
    )


@cache
def kernels():
    import torch

    prebuilt = os.environ.get("FOIDFINDER_KERNELS")
    if prebuilt:
        torch.ops.load_library(prebuilt)
    else:
        major, minor = torch.cuda.get_device_capability()
        build(None, [f"{major}.{minor}"])
    return torch.ops.foidfinder
