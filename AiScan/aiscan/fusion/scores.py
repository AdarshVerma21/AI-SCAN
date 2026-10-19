from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..models.probe_dinov2_adapter import PROBEDINOv2Adapter
from ..models.safe_adapter import SAFEAdapter
from ..preprocessing.image_loader import load_image
from .dataset import ManifestRow, manifest_digest


@dataclass(frozen=True)
class ScoreRecord:
    path: str
    label: int
    split: str
    generator: str
    status: str
    safe_logit: float | None = None
    probe_logit: float | None = None
    error: str | None = None


def _file_identifier(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size": path.stat().st_size}


def _checkpoint_identifier(path: Path) -> dict[str, object]:
    return _file_identifier(path)


def _metadata(manifest: Path, safe: SAFEAdapter, probe: PROBEDINOv2Adapter) -> dict[str, object]:
    return {
        "schema_version": 1,
        "manifest_sha256": manifest_digest(manifest),
        "features": ["SAFE.ai_logit", "PROBE-DINOv2.ai_logit"],
        "score_semantics": {
            "SAFE.ai_logit": "SAFE class-1 fake logit minus class-0 real logit",
            "PROBE-DINOv2.ai_logit": "official scalar AI-vs-real logit, averaged over 336px patches",
        },
        "models": {
            "SAFE": _checkpoint_identifier(safe.checkpoint),
            "PROBE-DINOv2": _checkpoint_identifier(probe.checkpoint),
            "PROBE-backbone": probe.backbone,
        },
    }


def _write_csv(path: Path, records: Iterable[ScoreRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ScoreRecord.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def read_scores(path: str | Path) -> list[ScoreRecord]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [ScoreRecord(
            row["path"], int(row["label"]), row["split"], row["generator"],
            row["status"],
            None if row["safe_logit"] == "" else float(row["safe_logit"]),
            None if row["probe_logit"] == "" else float(row["probe_logit"]),
            row["error"] or None,
        ) for row in csv.DictReader(handle)]


def extract_scores(manifest: Path, output: Path, force: bool = False) -> list[ScoreRecord]:
    from .dataset import load_manifest

    rows = load_manifest(manifest)
    safe_repo = Path(os.environ["SAFE_REPO"])
    safe_checkpoint = Path(os.environ["SAFE_CHECKPOINT"])
    probe_repo = Path(os.environ["PROBE_REPO"])
    probe_checkpoint = Path(os.environ["PROBE_CHECKPOINT"])
    backbone = os.environ["PROBE_DINOV2_BACKBONE"]
    safe = SAFEAdapter(safe_repo, safe_checkpoint)
    probe = PROBEDINOv2Adapter(probe_repo, probe_checkpoint, backbone)
    metadata_path = output.with_suffix(".metadata.json")
    metadata = _metadata(manifest, safe, probe)
    if not force and output.is_file() and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in metadata.items()):
            cached = read_scores(output)
            if len(cached) == len(rows):
                return cached

    records: list[ScoreRecord] = []
    for row in rows:
        try:
            image = load_image(row.path)
            safe_result = safe.predict(image)
            probe_result = probe.predict(image)
            records.append(ScoreRecord(
                str(row.path), row.label, row.split, row.generator, "success",
                safe_result.ai_logit, probe_result.ai_logit,
            ))
        except (ValueError, FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
            records.append(ScoreRecord(
                str(row.path), row.label, row.split, row.generator, "error",
                error=str(exc),
            ))
    _write_csv(output, records)
    metadata["record_count"] = len(records)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return records
