from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from ..inference.predictor import Predictor
from .logistic_fusion import LogisticFusion


def _rows(manifest: Path, split: str) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise ValueError(f"Manifest contains no rows for split={split!r}.")
    return rows


def _cache_key(rows: list[dict[str, str]]) -> str:
    return hashlib.sha256("".join(f"{r['path']}|{r['label']}" for r in rows).encode()).hexdigest()[:16]


def train(manifest: Path, output: Path, cache_dir: Path, seed: int = 42) -> None:
    rows = _rows(manifest, "fusion")
    validation = _rows(manifest, "validation")
    predictor = Predictor.from_environment()
    cache_dir.mkdir(parents=True, exist_ok=True)

    def features(items: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        cache = cache_dir / f"{_cache_key(items)}.npz"
        if cache.exists():
            values = np.load(cache)
            return values["x"], values["y"]
        predictions = predictor.predict_batch([row["path"] for row in items])
        failures = [p.error for p in predictions if p.status != "success"]
        if failures:
            raise RuntimeError(f"Detector inference failed while building cache: {failures[0]}")
        x = np.asarray([[p.model_scores["SAFE"], p.model_scores["PROBE-DINOv2"]] for p in predictions])
        y = np.asarray([int(row["label"]) for row in items])
        np.savez(cache, x=x, y=y)
        return x, y

    x_train, y_train = features(rows)
    x_val, y_val = features(validation)
    model = LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", random_state=seed)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_val)[:, 1]
    thresholds = np.unique(np.r_[0.0, probabilities, 1.0])
    threshold = max(thresholds, key=lambda value: balanced_accuracy_score(y_val, probabilities >= value))
    fusion = LogisticFusion(model.coef_[0], float(model.intercept_[0]), float(threshold))
    output.parent.mkdir(parents=True, exist_ok=True)
    fusion.save(output)
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "seed": seed, "manifest": str(manifest), "fusion_rows": len(rows),
        "validation_rows": len(validation), "threshold_metric": "balanced_accuracy",
        "cache_dir": str(cache_dir),
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    args = parser.parse_args()
    train(args.manifest, args.output, args.cache_dir)


if __name__ == "__main__":
    main()

