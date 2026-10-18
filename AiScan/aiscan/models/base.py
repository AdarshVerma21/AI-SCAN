from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from PIL import Image


@dataclass(frozen=True)
class DetectorResult:
    model_name: str
    raw_output: Any
    ai_logit: float
    inference_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Detector(Protocol):
    model_name: str

    def predict(self, image: Image.Image) -> DetectorResult:
        ...


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device

