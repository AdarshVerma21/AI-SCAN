from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    device: str
    checks: dict[str, bool]
    warnings: list[str] = Field(default_factory=list)


class InfoResponse(BaseModel):
    name: str
    version: str
    endpoints: list[str]
    model_status: str


class PredictionResponse(BaseModel):
    prediction: str | None
    ai_score: float | None
    threshold: float | None
    model_scores: dict[str, float | None]
    processing_time_ms: float
    status: str
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    uncertainty: bool = False
    image: dict[str, Any] | None = None

