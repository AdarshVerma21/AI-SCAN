from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}


def load_image(path: str | Path, min_size: int = 32) -> Image.Image:
    image_path = Path(path)
    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")
    try:
        with Image.open(image_path) as image:
            image.load()
            if min(image.size) < min_size:
                raise ValueError(f"Image dimensions must be at least {min_size}px.")
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unable to decode image: {image_path}") from exc

