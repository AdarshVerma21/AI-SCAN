from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LogisticFusion:
    weights: np.ndarray
    bias: float
    threshold: float = 0.5
    feature_means: np.ndarray | None = None
    feature_scales: np.ndarray | None = None

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] != len(self.weights):
            raise ValueError("Fusion feature count does not match learned weights.")
        if self.feature_means is not None:
            values = (values - self.feature_means) / self.feature_scales
        return values

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = self.transform(features) @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features) >= self.threshold).astype(np.int64)

    def save(self, path: str | Path) -> None:
        payload = {
            "weights": self.weights.tolist(), "bias": self.bias, "threshold": self.threshold,
            "feature_means": None if self.feature_means is None else self.feature_means.tolist(),
            "feature_scales": None if self.feature_scales is None else self.feature_scales.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LogisticFusion":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        means = payload.get("feature_means")
        scales = payload.get("feature_scales")
        return cls(np.asarray(payload["weights"], dtype=np.float64), float(payload["bias"]),
                   float(payload["threshold"]),
                   None if means is None else np.asarray(means, dtype=np.float64),
                   None if scales is None else np.asarray(scales, dtype=np.float64))

