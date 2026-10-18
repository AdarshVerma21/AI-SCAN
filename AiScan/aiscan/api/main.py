from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .schemas import HealthResponse, InfoResponse, PredictionResponse
from .service import DetectorService, PredictorFactory

MAX_UPLOAD_BYTES = int(os.environ.get("AIScan_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.environ.get("AIScan_MAX_IMAGE_PIXELS", "25000000"))
API_VERSION = "2.0.0"


def _cors_origins() -> list[str]:
    configured = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,"
        "http://127.0.0.1:3000,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def _validated_image(data: bytes) -> tuple[Image.Image, str]:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_BYTES} bytes.")
    try:
        with Image.open(__import__("io").BytesIO(data)) as image:
            image.verify()
        with Image.open(__import__("io").BytesIO(data)) as image:
            width, height = image.size
            if width < 32 or height < 32:
                raise HTTPException(status_code=400, detail="Image dimensions must be at least 32x32 pixels.")
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image has too many pixels.")
            image.load()
            image_format = (image.format or "").upper()
            if image_format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP images are supported.")
            return image.convert("RGB"), image_format
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc


def create_app(predictor_factory: PredictorFactory | None = None) -> FastAPI:
    service = DetectorService(predictor_factory or __import__(
        "aiscan.inference.predictor", fromlist=["Predictor"]
    ).Predictor.from_environment, readiness_override=predictor_factory is not None)
    app = FastAPI(title="AiScan API", version=API_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/", response_model=InfoResponse)
    async def info() -> InfoResponse:
        readiness = service.readiness()
        return InfoResponse(
            name="AiScan AI-generated image detector",
            version=API_VERSION,
            endpoints=["GET /health", "GET /", "POST /predict"],
            model_status="ready" if readiness["model_ready"] else "not_ready",
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(**service.readiness())

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)) -> PredictionResponse:
        started = time.perf_counter()
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        image, image_format = _validated_image(data)
        readiness = service.readiness()
        if not readiness["model_ready"]:
            return PredictionResponse(
                prediction=None,
                ai_score=None,
                threshold=None,
                model_scores={},
                processing_time_ms=(time.perf_counter() - started) * 1000,
                status="model_not_ready",
                warnings=list(readiness["warnings"]),
            )
        suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[image_format]
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                image.save(temporary_path, format=image_format)
                result = service.predict(temporary_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        except (ValueError, FileNotFoundError, OSError, ImportError, RuntimeError) as exc:
            return PredictionResponse(
                prediction=None,
                ai_score=None,
                threshold=None,
                model_scores={},
                processing_time_ms=(time.perf_counter() - started) * 1000,
                status="inference_error",
                warnings=["Model inference failed."],
                error=str(exc),
            )
        warnings = ["AI score is a fused detector score, not a guaranteed calibrated probability."]
        if result.uncertainty:
            warnings.append("Result is near the configured decision threshold.")
        prediction = {
            "AI-GENERATED": "AI-generated",
            "REAL": "Real",
        }.get(result.prediction, result.prediction)
        return PredictionResponse(
            prediction=prediction,
            ai_score=result.ai_score,
            threshold=result.threshold,
            model_scores=result.model_scores,
            processing_time_ms=(time.perf_counter() - started) * 1000,
            status=result.status,
            warnings=warnings,
            error=result.error,
            uncertainty=result.uncertainty,
            image={"format": image_format, "width": image.width, "height": image.height},
        )

    return app


app = create_app()
