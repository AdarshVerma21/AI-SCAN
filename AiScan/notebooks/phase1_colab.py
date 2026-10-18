"""Ordered Colab recipe; copy these cells into a notebook.

This executable script intentionally does not download weights or fabricate
metrics. Set checkpoint paths after accepting upstream licenses and terms.
"""

# 1. GPU
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# 2. Install (run in a Colab cell):
# !pip install -r requirements.txt

# 3. Configure official repositories and checkpoints.
# import os
# os.environ["SAFE_REPO"] = "/content/SAFE"
# os.environ["SAFE_CHECKPOINT"] = "/content/checkpoints/checkpoint-best.pth"
# os.environ["PROBE_REPO"] = "/content/PROBE-AIGI-Detection"
# os.environ["PROBE_CHECKPOINT"] = "/content/checkpoints/PROBE-DINOv2.pth"

# 4. Initialize only after all four paths exist.
# from aiscan.inference.predictor import Predictor
# predictor = Predictor.from_environment()

# 5-10. Build a manifest, train fusion, evaluate, and save artifacts:
# !python -m aiscan.fusion.train --manifest data/manifest.csv --output artifacts/fusion.json
# !python -m aiscan.evaluation.evaluate --manifest data/manifest.csv --fusion artifacts/fusion.json --output artifacts/report.json
