from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from .base import DetectorResult, select_device


class PROBEDINOv2Adapter:
    """Adapter for the official PROBE Detector/model/dino_classifier.py."""

    model_name = "PROBE-DINOv2"

    def __init__(
        self,
        repo: str | Path,
        checkpoint: str | Path,
        backbone: str,
        device: str = "auto",
    ):
        self.repo = Path(repo)
        self.checkpoint = Path(checkpoint)
        self.backbone = backbone
        self.device = select_device(device)
        self.model: torch.nn.Module | None = None
        self._transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _load(self) -> None:
        if self.model is not None:
            return
        if not self.repo.is_dir():
            raise FileNotFoundError(f"PROBE repository not found: {self.repo}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"PROBE-DINOv2 checkpoint not found: {self.checkpoint}")
        model_dir = self.repo / "Detector"
        classifier_file = model_dir / "model" / "dino_classifier.py"
        if not classifier_file.is_file():
            raise FileNotFoundError(f"PROBE classifier implementation not found: {classifier_file}")
        sys.path.insert(0, str(model_dir))
        spec = importlib.util.spec_from_file_location("aiscan_probe_dino_classifier", classifier_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import PROBE model from {classifier_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        model = module.dino_classifier(model_name=self.backbone)
        state = torch.load(self.checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        if not isinstance(state, dict):
            raise ValueError("PROBE checkpoint is not a state dictionary.")
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()

    def predict(self, image: Image.Image) -> DetectorResult:
        self._load()
        assert self.model is not None
        image_tensor = transforms.ToTensor()(image.convert("RGB"))
        height, width = image_tensor.shape[-2:]
        pad_height = max(0, 336 - height)
        pad_width = max(0, 336 - width)
        if pad_height or pad_width:
            image_tensor = TF.pad(image_tensor, [0, 0, pad_width, pad_height])
        patches = image_tensor.unfold(1, 336, 336).unfold(2, 336, 336)
        patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, 336, 336)
        patches = torch.stack([self._transform(transforms.ToPILImage()(patch)) for patch in patches])
        patches = patches.to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            patch_outputs = self.model(patches)
        if patch_outputs.ndim != 2 or patch_outputs.shape[1] != 1:
            raise ValueError(
                f"PROBE-DINOv2 returned unexpected shape: {tuple(patch_outputs.shape)}"
            )
        output = patch_outputs.mean(dim=0, keepdim=True)
        logit = float(output[0, 0].detach().cpu())
        return DetectorResult(
            self.model_name, output.detach().cpu(), logit,
            (time.perf_counter() - start) * 1000,
            {
                "device": str(self.device),
                "input_size": 336,
                "patch_count": int(patches.shape[0]),
                "output_semantics": "AI-vs-real logit",
            },
        )
