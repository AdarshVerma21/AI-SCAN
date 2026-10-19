from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from .dataset import load_manifest, leakage_report, split_rows
from .logistic_fusion import LogisticFusion
from .scores import ScoreRecord, extract_scores, read_scores


FEATURES = ("safe_logit", "probe_logit")


def _successful(records: list[ScoreRecord], split: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [record for record in records if record.split == split]
    failures = [record for record in selected if record.status != "success"]
    if failures:
        raise RuntimeError(
            f"{len(failures)} {split} score rows failed; first error: {failures[0].error}"
        )
    if not selected:
        raise ValueError(f"Score cache contains no split={split!r} rows.")
    values = np.asarray([[record.safe_logit, record.probe_logit] for record in selected], dtype=np.float64)
    labels = np.asarray([record.label for record in selected], dtype=np.int64)
    if not np.isfinite(values).all():
        raise ValueError(f"Score cache contains non-finite values in split={split!r}.")
    return values, labels


def _select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    thresholds = np.unique(np.r_[0.0, probabilities, 1.0])
    return float(max(
        thresholds,
        key=lambda value: balanced_accuracy_score(labels, probabilities >= value),
    ))


def train(
    manifest: Path,
    output: Path,
    scores_path: Path,
    seed: int = 42,
    force_scores: bool = False,
) -> LogisticFusion:
    rows = load_manifest(manifest)
    split_rows(rows, "train")
    split_rows(rows, "validation")
    records = extract_scores(manifest, scores_path, force=force_scores)
    x_train, y_train = _successful(records, "train")
    x_validation, y_validation = _successful(records, "validation")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training split must contain both labels 0 and 1.")
    if len(np.unique(y_validation)) < 2:
        raise ValueError("Validation split must contain both labels 0 and 1.")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = LogisticRegression(
        C=1.0, penalty="l2", class_weight="balanced", random_state=seed,
    )
    model.fit(x_train_scaled, y_train)
    validation_probability = model.predict_proba(scaler.transform(x_validation))[:, 1]
    threshold = _select_threshold(y_validation, validation_probability)
    metadata = {
        "seed": seed,
        "features": list(FEATURES),
        "train_rows": len(y_train),
        "validation_rows": len(y_validation),
        "threshold_metric": "balanced_accuracy",
        "manifest": str(manifest),
        "scores": str(scores_path),
        "leakage": leakage_report(rows),
    }
    fusion = LogisticFusion(
        model.coef_[0],
        float(model.intercept_[0]),
        threshold,
        scaler.mean_,
        scaler.scale_,
        artifact_version="fusion-v1",
        metadata=metadata,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fusion.save(output)
    return fusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SAFE + PROBE logistic fusion classifier.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fusion.json"))
    parser.add_argument("--scores", type=Path, default=Path("artifacts/scores.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-scores", action="store_true")
    args = parser.parse_args()
    train(args.manifest, args.output, args.scores, args.seed, args.force_scores)


if __name__ == "__main__":
    main()
