"""Image decoding with size limits, and the color handling every backend shares."""

from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 64_000_000

# decode() enforces the limit itself; this stops Pillow from warning below 2x it.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageError(ValueError):
    """An upload that is not a usable image. `code` is stable for API clients."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def decode(data: bytes, max_pixels: int = MAX_PIXELS) -> Image.Image:
    """Bytes -> fully loaded image, EXIF orientation applied. Raises ImageError."""
    too_large = ImageError("too_many_pixels", f"Larger than {max_pixels // 1_000_000} megapixels")
    try:
        image = Image.open(io.BytesIO(data))  # reads the header only
    except UnidentifiedImageError as exc:
        raise ImageError("unsupported_format", "Unsupported image format") from exc
    except Image.DecompressionBombError as exc:
        raise too_large from exc
    if image.width * image.height > max_pixels:
        raise too_large
    try:
        image.load()
        return ImageOps.exif_transpose(image)
    except (OSError, ValueError, SyntaxError) as exc:  # truncated or corrupt data
        raise ImageError("unreadable_image", "Damaged or incomplete image") from exc


def rgb_on_white(image: Image.Image) -> Image.Image:
    """RGB with any transparency composited onto white, as PixAI's reference preprocessing does."""
    if image.mode == "RGB":
        return image
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")
