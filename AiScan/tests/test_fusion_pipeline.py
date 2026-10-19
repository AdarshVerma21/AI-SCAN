import csv

import numpy as np
from PIL import Image

from aiscan.fusion import train as train_module
from aiscan.fusion.dataset import ManifestValidationError, load_manifest
from aiscan.fusion.logistic_fusion import LogisticFusion
from aiscan.fusion.scores import ScoreRecord


def _manifest(tmp_path):
    rows = []
    for index, (label, split, generator, score) in enumerate([
        (0, "train", "camera", -2.0),
        (1, "train", "sd", 2.0),
        (0, "validation", "camera", -1.5),
        (1, "validation", "sd", 1.5),
        (0, "test", "camera", -1.0),
        (1, "test", "sd", 1.0),
    ]):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (8, 8), (index, index, index)).save(path)
        rows.append((path.name, label, split, generator, score))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "split", "generator"])
        writer.writerows(row[:4] for row in rows)
    return manifest, rows


def test_manifest_validates_required_splits_and_paths(tmp_path):
    manifest, _ = _manifest(tmp_path)
    assert len(load_manifest(manifest)) == 6
    manifest.write_text("path,label,split,generator\nmissing.png,0,train,x\n", encoding="utf-8")
    try:
        load_manifest(manifest)
    except ManifestValidationError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("missing image was accepted")


def test_training_fits_scaler_only_on_training_scores(tmp_path, monkeypatch):
    manifest, rows = _manifest(tmp_path)
    records = [
        ScoreRecord(str(tmp_path / row[0]), row[1], row[2], row[3], "success", row[4], row[4])
        for row in rows
    ]
    monkeypatch.setattr(train_module, "extract_scores", lambda *args, **kwargs: records)
    output = tmp_path / "fusion.json"
    fusion = train_module.train(manifest, output, tmp_path / "scores.csv", seed=7)
    loaded = LogisticFusion.load(output)
    assert fusion.metadata["seed"] == 7
    assert loaded.feature_means is not None
    assert np.allclose(loaded.feature_means, [0.0, 0.0])
    assert loaded.artifact_version == "fusion-v1"
