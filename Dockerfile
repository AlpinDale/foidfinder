# syntax=docker/dockerfile:1.7
# docker build -t foidfinder --build-arg CUDA_ARCHS="8.6 8.9" .
# docker run --gpus all --init -p 8000:8000 -v foidfinder-models:/models foidfinder

ARG CUDA_VERSION=13.0.1
ARG UBUNTU=ubuntu24.04

# Web UI
FROM node:22-bookworm-slim AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build  # -> /src/foidfinder/webui

# Python environment and prebuilt kernels
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.12 /uv /usr/local/bin/uv
ENV PATH=/opt/venv/bin:$PATH \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=3.12 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=automatic
WORKDIR /src
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra cuda --no-install-project
COPY foidfinder ./foidfinder
COPY --from=web /src/foidfinder/webui ./foidfinder/webui
# uv's cache would reuse a stale wheel of this project (it keys on pyproject.toml only).
RUN uv sync --frozen --no-dev --extra cuda --no-editable --no-cache
ARG CUDA_ARCHS="8.9"
RUN foidfinder build-kernels /opt/kernels --archs ${CUDA_ARCHS} \
    && find /opt/kernels -mindepth 1 ! -name pixai_kernels.so -delete

# Runtime: no compiler, no sources, non-root
FROM nvidia/cuda:${CUDA_VERSION}-base-${UBUNTU}
RUN useradd --system --uid 10001 --home-dir /models --no-create-home foidfinder \
    && mkdir /models && chown foidfinder /models
COPY --from=build /opt/python /opt/python
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/kernels/pixai_kernels.so /opt/kernels/pixai_kernels.so
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/.huggingface \
    FOIDFINDER_KERNELS=/opt/kernels/pixai_kernels.so \
    FOIDFINDER_BACKEND=cuda \
    FOIDFINDER_HOST=0.0.0.0 \
    FOIDFINDER_PORT=8000 \
    FOIDFINDER_MODEL_DIR=/models/pixai-tagger-v1.0 \
    FOIDFINDER_MODEL_REPO=alpindale/PixAI-v1.0-INT8 \
    FOIDFINDER_DOWNLOAD_MISSING=1
USER foidfinder
VOLUME /models
EXPOSE 8000
# First start downloads the model.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 CMD ["python", "-c", \
    "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/health' % os.environ['FOIDFINDER_PORT'], timeout=4)"]
ENTRYPOINT ["foidfinder"]
CMD ["serve"]
