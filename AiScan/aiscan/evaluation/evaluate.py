from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .metrics import binary_metrics
from ..fusion.logistic_fusion import LogisticFusion
from ..inference.predictor import Predictor


def evaluate(manifest: Path, fusion_path: Path, output: Path) -> None:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    if not rows:
        raise ValueError("Manifest contains no split=test rows.")
    fusion = LogisticFusion.load(fusion_path)
    predictor = Predictor.from_environment()
    predictions = predictor.predict_batch([row["path"] for row in rows])
    failures = [p.error for p in predictions if p.status != "success"]
    if failures:
        raise RuntimeError(f"Evaluation inference failed: {failures[0]}")
    labels = [int(row["label"]) for row in rows]
    features = [[p.model_scores["SAFE"], p.model_scores["PROBE-DINOv2"]] for p in predictions]
    fused = fusion.predict_proba(features)
    report = {
        "split": "test", "sample_count": len(rows), "threshold": fusion.threshold,
        "fused": binary_metrics(labels, fused, fusion.threshold),
        "individual": {
            "SAFE": binary_metrics(labels, [x[0] for x in features], 0.0),
            "PROBE-DINOv2": binary_metrics(labels, [x[1] for x in features], 0.0),
        },
        "note": "Scores are logit margins; individual 0.0 thresholds are not calibrated probabilities.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.manifest, args.fusion, args.output)


if __name__ == "__main__":
    main()

