from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageFilter

from ..inference.predictor import Predictor


def evaluate_transforms(predictor: Predictor, image_path: str | Path) -> dict[str, dict]:
    """Run explicitly named perturbations; metrics still require labeled data."""
    source = Image.open(image_path).convert("RGB")
    transforms = {
        "original": source,
        "resize": source.resize((max(32, source.width // 2), max(32, source.height // 2))),
        "crop": source.crop((0, 0, max(32, source.width // 2), max(32, source.height // 2))),
        "blur": source.filter(ImageFilter.GaussianBlur(radius=1.0)),
    }
    results = {}
    with TemporaryDirectory() as directory:
        for name, image in transforms.items():
            path = Path(directory) / f"{name}.png"
            image.save(path)
            results[name] = predictor.predict_image(path).to_dict()
    return results

