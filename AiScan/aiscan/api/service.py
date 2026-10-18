from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Callable

import torch

from ..inference.predictor import Predictor, Prediction


PredictorFactory = Callable[[], Predictor]


class DetectorService:
    def __init__(
        self,
        predictor_factory: PredictorFactory = Predictor.from_environment,
        readiness_override: bool = False,
    ):
        self._predictor_factory = predictor_factory
        self._readiness_override = readiness_override
        self._predictor: Predictor | None = None
        self._load_error: str | None = None

    def readiness(self) -> dict[str, object]:
        values = {
            "SAFE_REPO": os.environ.get("SAFE_REPO"),
            "SAFE_CHECKPOINT": os.environ.get("SAFE_CHECKPOINT"),
            "PROBE_REPO": os.environ.get("PROBE_REPO"),
            "PROBE_CHECKPOINT": os.environ.get("PROBE_CHECKPOINT"),
            "PROBE_DINOV2_BACKBONE": os.environ.get("PROBE_DINOV2_BACKBONE"),
        }
        checks = {
            "safe_repository": bool(values["SAFE_REPO"]) and Path(values["SAFE_REPO"]).is_dir(),
            "safe_checkpoint": bool(values["SAFE_CHECKPOINT"]) and Path(values["SAFE_CHECKPOINT"]).is_file(),
            "probe_repository": bool(values["PROBE_REPO"]) and Path(values["PROBE_REPO"]).is_dir(),
            "probe_checkpoint": bool(values["PROBE_CHECKPOINT"]) and Path(values["PROBE_CHECKPOINT"]).is_file(),
            "probe_backbone_configured": bool(values["PROBE_DINOV2_BACKBONE"]),
            "fusion_artifact": Path(os.environ.get("FUSION_ARTIFACT", "artifacts/fusion.json")).is_file(),
            "transformers_installed": importlib.util.find_spec("transformers") is not None,
        }
        warnings = [name.replace("_", " ") + " is unavailable" for name, passed in checks.items() if not passed]
        if self._load_error:
            warnings.append("model initialization failed: " + self._load_error)
        ready = (self._readiness_override or all(checks.values())) and self._load_error is None
        if self._readiness_override:
            warnings.append("readiness is overridden by an injected predictor.")
        return {
            "status": "ok" if ready else "degraded",
            "model_ready": ready,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "checks": checks,
            "warnings": warnings,
        }

    def predictor(self) -> Predictor:
        if self._predictor is None:
            try:
                self._predictor = self._predictor_factory()
            except (ValueError, FileNotFoundError, OSError, ImportError, RuntimeError) as exc:
                self._load_error = str(exc)
                raise
        return self._predictor

    def predict(self, image_path: Path) -> Prediction:
        return self.predictor().predict_image(image_path)
