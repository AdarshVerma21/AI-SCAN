import numpy as np
import pytest
from PIL import Image

from aiscan.fusion.logistic_fusion import LogisticFusion
from aiscan.models.base import select_device
from aiscan.preprocessing.image_loader import load_image


def test_fusion_round_trip(tmp_path):
    path = tmp_path / "fusion.json"
    original = LogisticFusion(np.array([1.0, -0.5]), 0.2, 0.6)
    original.save(path)
    loaded = LogisticFusion.load(path)
    assert np.allclose(original.predict_proba([[1, 2]]), loaded.predict_proba([[1, 2]]))
    assert loaded.threshold == 0.6


def test_invalid_extension_is_rejected(tmp_path):
    path = tmp_path / "image.gif"
    Image.new("RGB", (50, 50)).save(path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_image(path)


def test_corrupt_image_is_rejected(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="decode"):
        load_image(path)


def test_fusion_shape_is_validated():
    fusion = LogisticFusion(np.array([1.0, 1.0]), 0.0)
    with pytest.raises(ValueError):
        fusion.predict_proba([[1.0]])


def test_cpu_device_selection():
    assert select_device("cpu").type == "cpu"
