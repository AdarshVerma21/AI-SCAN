from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..fusion.dataset import leakage_report, load_manifest
from ..fusion.logistic_fusion import LogisticFusion
from ..fusion.scores import ScoreRecord, extract_scores, read_scores
from .metrics import binary_metrics


def _records_for_split(records: list[ScoreRecord], split: str) -> list[ScoreRecord]:
    selected = [record for record in records if record.split == split]
    failures = [record for record in selected if record.status != "success"]
    if failures:
        raise RuntimeError(
            f"{len(failures)} {split} score rows failed; first error: {failures[0].error}"
        )
    if not selected:
        raise ValueError(f"Score cache contains no split={split!r} rows.")
    return selected


def _metrics(records: list[ScoreRecord], fusion: LogisticFusion) -> dict[str, object]:
    labels = [record.label for record in records]
    features = np.asarray(
        [[record.safe_logit, record.probe_logit] for record in records],
        dtype=np.float64,
    )
    fused = fusion.predict_proba(features)
    return {
        "fused": binary_metrics(labels, fused, fusion.threshold),
        "individual": {
            "SAFE": binary_metrics(labels, features[:, 0], 0.0),
            "PROBE-DINOv2": binary_metrics(labels, features[:, 1], 0.0),
        },
    }


def evaluate(manifest: Path, fusion_path: Path, output: Path, scores_path: Path) -> dict[str, object]:
    rows = load_manifest(manifest)
    fusion = LogisticFusion.load(fusion_path)
    if len(fusion.weights) != 2:
        raise ValueError("Fusion artifact must contain exactly two detector weights.")
    if scores_path.is_file() and scores_path.with_suffix(".metadata.json").is_file():
        records = read_scores(scores_path)
    else:
        records = extract_scores(manifest, scores_path)
    test_records = _records_for_split(records, "test")
    by_generator: dict[str, list[ScoreRecord]] = defaultdict(list)
    for record in test_records:
        by_generator[record.generator].append(record)
    report: dict[str, object] = {
        "schema_version": 1,
        "split": "test",
        "sample_count": len(test_records),
        "threshold": fusion.threshold,
        **_metrics(test_records, fusion),
        "by_generator": {
            generator: {"sample_count": len(items), **_metrics(items, fusion)}
            for generator, items in sorted(by_generator.items())
        },
        "by_extension": {
            extension: {"sample_count": len(items), **_metrics(items, fusion)}
            for extension, items in sorted(_group_by_extension(test_records).items())
        },
        "leakage_checks": leakage_report(rows),
        "degradation": {
            "status": "not_available",
            "reason": "Manifest has no degradation/compression columns.",
        },
        "artifact_version": fusion.artifact_version,
        "note": "Detector features are logits; individual thresholds use logit 0 and are not calibrated probabilities.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _group_by_extension(records: list[ScoreRecord]) -> dict[str, list[ScoreRecord]]:
    grouped: dict[str, list[ScoreRecord]] = defaultdict(list)
    for record in records:
        grouped[Path(record.path).suffix.lower() or "<none>"].append(record)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained fusion on held-out test data.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--scores", type=Path, default=Path("artifacts/scores.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/report.json"))
    args = parser.parse_args()
    evaluate(args.manifest, args.fusion, args.output, args.scores)


if __name__ == "__main__":
    main()
