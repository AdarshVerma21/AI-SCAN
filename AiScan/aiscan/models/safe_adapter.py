from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from .base import DetectorResult, select_device


class SAFEAdapter:
    """Adapter for the official Ouxiang-Li/SAFE ResNet implementation."""

    model_name = "SAFE"

    def __init__(self, repo: str | Path, checkpoint: str | Path, device: str = "auto"):
        self.repo = Path(repo)
        self.checkpoint = Path(checkpoint)
        self.device = select_device(device)
        self.model: torch.nn.Module | None = None
        self._transform = transforms.Compose(
            [
                transforms.CenterCrop((256, 256)),
                transforms.ToTensor(),
            ]
        )

    def _load(self) -> None:
        if self.model is not None:
            return
        if not self.repo.is_dir():
            raise FileNotFoundError(f"SAFE repository not found: {self.repo}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SAFE checkpoint not found: {self.checkpoint}")
        model_file = self.repo / "models" / "resnet.py"
        if not model_file.is_file():
            raise FileNotFoundError(f"SAFE model implementation not found: {model_file}")
        spec = importlib.util.spec_from_file_location("aiscan_safe_resnet", model_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import SAFE model from {model_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, "resnet50"):
            raise ImportError("SAFE resnet.py does not expose the official resnet50 factory.")
        model = module.resnet50(num_classes=2)
        state = torch.load(self.checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if not isinstance(state, dict):
            raise ValueError("SAFE checkpoint is not a state dictionary.")
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()

    def predict(self, image: Image.Image) -> DetectorResult:
        self._load()
        assert self.model is not None
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model(tensor)
        if output.ndim != 2 or output.shape[1] != 2:
            raise ValueError(f"SAFE returned unexpected shape: {tuple(output.shape)}")
        # SAFE follows the official real=0, fake=1 dataset convention.
        logit = float((output[0, 1] - output[0, 0]).detach().cpu())
        return DetectorResult(
            self.model_name, output.detach().cpu(), logit,
            (time.perf_counter() - start) * 1000,
            {"device": str(self.device), "label_mapping": {"0": "REAL", "1": "AI-GENERATED"}},
        )
