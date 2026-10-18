from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..models.base import Detector
from ..models.probe_dinov2_adapter import PROBEDINOv2Adapter
from ..models.safe_adapter import SAFEAdapter
from ..preprocessing.image_loader import load_image
from ..fusion.logistic_fusion import LogisticFusion


@dataclass(frozen=True)
class Prediction:
    prediction: str | None
    ai_score: float | None
    threshold: float | None
    model_scores: dict[str, float | None]
    inference_time_ms: float
    status: str
    error: str | None = None
    uncertainty: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class Predictor:
    def __init__(self, detectors: list[Detector], fusion: LogisticFusion):
        if not detectors:
            raise ValueError("At least one detector is required.")
        self.detectors = detectors
        self.fusion = fusion

    @classmethod
    def from_environment(cls) -> "Predictor":
        safe_repo = os.environ.get("SAFE_REPO")
        safe_checkpoint = os.environ.get("SAFE_CHECKPOINT")
        probe_repo = os.environ.get("PROBE_REPO")
        probe_checkpoint = os.environ.get("PROBE_CHECKPOINT")
        probe_backbone = os.environ.get("PROBE_DINOV2_BACKBONE")
        missing = [name for name, value in {
            "SAFE_REPO": safe_repo, "SAFE_CHECKPOINT": safe_checkpoint,
            "PROBE_REPO": probe_repo, "PROBE_CHECKPOINT": probe_checkpoint,
            "PROBE_DINOV2_BACKBONE": probe_backbone,
        }.items() if not value]
        if missing:
            raise ValueError("Missing required checkpoint configuration: " + ", ".join(missing))
        fusion_path = os.environ.get("FUSION_ARTIFACT", "artifacts/fusion.json")
        return cls([
            SAFEAdapter(safe_repo, safe_checkpoint),
            PROBEDINOv2Adapter(probe_repo, probe_checkpoint, probe_backbone),
        ],
                   LogisticFusion.load(fusion_path))

    def predict_image(self, image_path: str | Path) -> Prediction:
        start = time.perf_counter()
        try:
            image = load_image(image_path)
            results = [detector.predict(image) for detector in self.detectors]
            scores = {result.model_name: result.ai_logit for result in results}
            final = float(self.fusion.predict_proba(np.array([[result.ai_logit for result in results]]))[0])
            margin = abs(final - self.fusion.threshold)
            return Prediction(
                "AI-GENERATED" if final >= self.fusion.threshold else "REAL",
                final, self.fusion.threshold, scores,
                (time.perf_counter() - start) * 1000, "success",
                uncertainty=margin < 0.1,
            )
        except (ValueError, FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
            return Prediction(None, None, self.fusion.threshold, {}, (time.perf_counter() - start) * 1000,
                              "error", str(exc))

    def predict_batch(self, image_paths: list[str | Path]) -> list[Prediction]:
        return [self.predict_image(path) for path in image_paths]
