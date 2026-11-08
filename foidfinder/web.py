"""HTTP API and web UI over a compute backend."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from PIL import Image
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .backends import Backend
from .image import MAX_PIXELS, ImageError, decode
from .tagging import TagIndex

MAX_UPLOAD = 32 << 20
MAX_QUEUE = 16  # requests waiting for the model before new ones get 503
FORMATS = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/avif", "image/bmp", "image/tiff"]
WEBUI = Path(__file__).parent / "webui"
IMAGE_ERROR_STATUS = {"too_many_pixels": 413, "unsupported_format": 415, "unreadable_image": 422}
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; "
        # The in-browser engine downloads the model from Hugging Face (redirects to its CDN on *.hf.co).
        "connect-src 'self' https://huggingface.co https://*.hf.co; "
        "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers


class SecurityHeaders:
    """ASGI middleware: SECURITY_HEADERS on every response, long-lived caching for content-hashed /assets/."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        immutable = scope["path"].startswith("/assets/")

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.extend((k.lower().encode(), v.encode()) for k, v in SECURITY_HEADERS.items())
                if immutable:
                    headers.append((b"cache-control", b"public, max-age=31536000, immutable"))
            await send(message)

        await self.app(scope, receive, send_with_headers)


def create_app(
    load_backend: Callable[[], Backend],
    tags: TagIndex,
    *,
    max_upload: int = MAX_UPLOAD,
    max_pixels: int = MAX_PIXELS,
    max_queue: int = MAX_QUEUE,
) -> Starlette:
    """load_backend runs on the inference thread, which then owns the model (MLX requires this)."""
    executor = ThreadPoolExecutor(1, thread_name_prefix="foidfinder-model")
    backend: Backend | None = None
    pending = 0  # admitted requests not yet answered; event-loop only

    def load() -> Backend:
        loaded = load_backend()
        loaded.predict(Image.new("RGB", (64, 64)))  # compile / capture before the first request
        return loaded

    @asynccontextmanager
    async def lifespan(app: Starlette):
        nonlocal backend
        backend = await asyncio.get_running_loop().run_in_executor(executor, load)
        yield
        executor.shutdown(wait=True)

    async def read_body(request: Request) -> bytes:
        too_large = ApiError(413, "too_large", f"Larger than {max_upload >> 20} MB")
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > max_upload:
            raise too_large
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > max_upload:
                raise too_large
            chunks.append(chunk)
        if not size:
            raise ApiError(400, "empty_body", "Send the image as the request body")
        return b"".join(chunks)

    async def info(request: Request) -> Response:
        return JSONResponse(
            {
                "model": tags.name,
                "backend": backend.name,
                "device": backend.device,
                "max_upload_bytes": max_upload,
                "max_pixels": max_pixels,
                "formats": FORMATS,
            },
            headers={"Cache-Control": "no-cache"},
        )

    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok"}, headers={"Cache-Control": "no-store"})

    async def tag(request: Request) -> Response:
        nonlocal pending
        start = time.perf_counter()
        if pending >= max_queue:
            raise ApiError(503, "busy", "Server busy", {"Retry-After": "2"})
        pending += 1
        try:
            data = await read_body(request)
            try:
                image = await run_in_threadpool(decode, data, max_pixels)
            except ImageError as exc:
                raise ApiError(IMAGE_ERROR_STATUS[exc.code], exc.code, str(exc)) from exc
            decoded = time.perf_counter()

            def predict():  # excludes time spent waiting for the inference thread
                t = time.perf_counter()
                return backend.predict(image), time.perf_counter() - t

            probs, model_s = await asyncio.get_running_loop().run_in_executor(executor, predict)
        finally:
            pending -= 1
        return JSONResponse(
            {
                **tags.result(probs),
                "image": {"width": image.width, "height": image.height},
                "timings_ms": {
                    "decode": round((decoded - start) * 1000, 1),
                    "inference": round(model_s * 1000, 1),
                    "total": round((time.perf_counter() - start) * 1000, 1),
                },
            },
            headers={"Cache-Control": "no-store"},
        )

    async def index_page(request: Request) -> Response:
        page = WEBUI / "index.html"
        if not page.exists():
            return Response("web UI not built: run `npm run build` in web/", 503, media_type="text/plain")
        return FileResponse(page, headers={"Cache-Control": "no-cache"})

    async def api_error(request: Request, exc: ApiError) -> Response:
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, exc.status, headers=exc.headers)

    routes = [
        Route("/api/info", info),
        Route("/api/health", health),
        Route("/api/tag", tag, methods=["POST"]),
        Route("/", index_page),
    ]
    if WEBUI.is_dir():
        routes.append(Mount("/", StaticFiles(directory=WEBUI)))
    app = Starlette(routes=routes, lifespan=lifespan, exception_handlers={ApiError: api_error})
    app.add_middleware(SecurityHeaders)
    return app
