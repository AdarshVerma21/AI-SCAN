from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from aiscan.api.main import create_app
from aiscan.inference.predictor import Prediction


def _image_bytes(format_name: str = "PNG") -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 80, 40)).save(stream, format=format_name)
    return stream.getvalue()


class FakePredictor:
    def predict_image(self, path):
        return Prediction(
            prediction="AI-GENERATED",
            ai_score=0.91,
            threshold=0.63,
            model_scores={"SAFE": 1.2, "PROBE-DINOv2": 2.1},
            inference_time_ms=2.0,
            status="success",
        )


def test_health_reports_not_ready_without_assets(monkeypatch):
    for name in (
        "SAFE_REPO", "SAFE_CHECKPOINT", "PROBE_REPO",
        "PROBE_CHECKPOINT", "PROBE_DINOV2_BACKBONE", "FUSION_ARTIFACT",
    ):
        monkeypatch.delenv(name, raising=False)
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_ready"] is False


def test_root_contains_api_info():
    response = TestClient(create_app()).get("/")
    assert response.status_code == 200
    assert "POST /predict" in response.json()["endpoints"]


def test_cors_allows_localhost_and_loopback_vite_origins():
    client = TestClient(create_app())
    for origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
        response = client.get("/health", headers={"Origin": origin})
        assert response.headers["access-control-allow-origin"] == origin


def test_valid_upload_uses_injected_predictor():
    client = TestClient(create_app(lambda: FakePredictor()))
    response = client.post("/predict", files={"file": ("image.bin", _image_bytes(), "application/octet-stream")})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "AI-generated"
    assert body["model_scores"]["SAFE"] == 1.2
    assert body["image"]["format"] == "PNG"


def test_corrupt_upload_is_rejected():
    client = TestClient(create_app(lambda: FakePredictor()))
    response = client.post("/predict", files={"file": ("image.png", b"not image", "image/png")})
    assert response.status_code == 400


def test_unsupported_image_is_rejected():
    client = TestClient(create_app(lambda: FakePredictor()))
    response = client.post("/predict", files={"file": ("image.gif", _image_bytes("GIF"), "image/gif")})
    assert response.status_code == 415


def test_not_ready_prediction_does_not_call_model(monkeypatch):
    for name in (
        "SAFE_REPO", "SAFE_CHECKPOINT", "PROBE_REPO",
        "PROBE_CHECKPOINT", "PROBE_DINOV2_BACKBONE", "FUSION_ARTIFACT",
    ):
        monkeypatch.delenv(name, raising=False)
    response = TestClient(create_app()).post(
        "/predict", files={"file": ("image.png", _image_bytes(), "image/png")}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "model_not_ready"
