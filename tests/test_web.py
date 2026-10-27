"""HTTP contract of foidfinder.web with a stub backend (no model needed)."""

import io
import threading

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from foidfinder import web
from foidfinder.tagging import TagIndex

TAGS = [
    "1girl",
    "solo",
    "kafka_(honkai:_star_rail)",
    "himeko_(honkai:_star_rail)",
    "honkai:_star_rail",
    "rating:g",
    "rating:e",
]
SPLIT = [["general", 2], ["character", 2], ["copyright", 1], ["rating", 2]]
PROBS = np.array([0.99, 0.10, 0.95, 0.12, 0.90, 0.70, 0.01], dtype=np.float32)


class StubBackend:
    name = "stub"
    device = "none"

    def __init__(self, gate: threading.Event | None = None):
        self.gate = gate
        self.seen: list[tuple[str, tuple[int, int]]] = []

    def predict(self, image: Image.Image) -> np.ndarray:
        self.seen.append((image.mode, image.size))
        if self.gate is not None and image.size != (64, 64):  # let the warmup through
            self.gate.wait(5)
        return PROBS


@pytest.fixture
def tags() -> TagIndex:
    config = {
        "tags": TAGS,
        "tags_split": SPLIT,
        "num_classes": len(TAGS),
        "category_best_threshold": {"general": 0.17, "character": 0.27, "copyright": 0.24, "rating": 0.41},
    }
    return TagIndex(config)


def png(size=(40, 30), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, "PNG")
    return buf.getvalue()


def test_tag_result_shape(tags):
    backend = StubBackend()
    with TestClient(web.create_app(lambda: backend, tags)) as client:
        r = client.post("/api/tag", content=png())
    assert r.status_code == 200
    body = r.json()
    assert body["tags"]["character"] == [{"tag": "kafka_(honkai:_star_rail)", "score": 0.95}]
    assert body["candidates"]["character"] == [{"tag": "himeko_(honkai:_star_rail)", "score": 0.12}]
    assert body["tags"]["general"] == [{"tag": "1girl", "score": 0.99}]
    assert body["rating"] == {"tag": "rating:g", "score": 0.7}
    assert body["image"] == {"width": 40, "height": 30}
    assert set(body["timings_ms"]) == {"decode", "inference", "total"}


def test_exif_orientation_is_applied_before_inference(tags):
    backend = StubBackend()
    image = Image.new("RGB", (40, 30))
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90 degrees on display
    buf = io.BytesIO()
    image.save(buf, "JPEG", exif=exif)
    with TestClient(web.create_app(lambda: backend, tags)) as client:
        assert client.post("/api/tag", content=buf.getvalue()).json()["image"] == {"width": 30, "height": 40}
    assert backend.seen[-1][1] == (30, 40)


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        (b"", 400, "empty_body"),
        (b"not an image", 415, "unsupported_format"),
        (png()[:60], 422, "unreadable_image"),
    ],
)
def test_bad_uploads(tags, body, status, code):
    with TestClient(web.create_app(lambda: StubBackend(), tags)) as client:
        r = client.post("/api/tag", content=body)
    assert (r.status_code, r.json()["error"]["code"]) == (status, code)


def test_upload_and_pixel_limits(tags):
    with TestClient(web.create_app(lambda: StubBackend(), tags, max_upload=1000, max_pixels=1000)) as client:
        too_large = client.post("/api/tag", content=b"x" * 1001)
        too_many_pixels = client.post("/api/tag", content=png((50, 50)))  # 2500 px, ~100 bytes
    assert (too_large.status_code, too_large.json()["error"]["code"]) == (413, "too_large")
    assert (too_many_pixels.status_code, too_many_pixels.json()["error"]["code"]) == (413, "too_many_pixels")


def test_busy_server_answers_503_with_retry_after(tags):
    gate = threading.Event()
    backend = StubBackend(gate)
    with TestClient(web.create_app(lambda: backend, tags, max_queue=1)) as client:
        first = threading.Thread(target=client.post, args=("/api/tag",), kwargs={"content": png()})
        first.start()
        for _ in range(500):  # until the first request holds the only queue slot
            if len(backend.seen) > 1:
                break
            threading.Event().wait(0.01)
        r = client.post("/api/tag", content=png())
        gate.set()
        first.join()
    assert r.status_code == 503 and r.headers["retry-after"] == "2"
    assert r.json()["error"]["code"] == "busy"


def test_security_headers_and_info(tags):
    with TestClient(web.create_app(lambda: StubBackend(), tags)) as client:
        r = client.get("/api/info")
    assert r.json()["max_upload_bytes"] == web.MAX_UPLOAD
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
