from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REQUIRED_COLUMNS = ("path", "label", "split", "generator")
VALID_SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class ManifestRow:
    path: Path
    label: int
    split: str
    generator: str


class ManifestValidationError(ValueError):
    pass


def manifest_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: str | Path) -> list[ManifestRow]:
    manifest = Path(path)
    if not manifest.is_file():
        raise ManifestValidationError(f"Manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManifestValidationError("Manifest is missing a header row.")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ManifestValidationError("Manifest is missing columns: " + ", ".join(missing))
        raw_rows = list(reader)

    errors: list[str] = []
    rows: list[ManifestRow] = []
    seen_paths: set[Path] = set()
    for number, raw in enumerate(raw_rows, start=2):
        raw_path = (raw.get("path") or "").strip()
        raw_label = (raw.get("label") or "").strip()
        split = (raw.get("split") or "").strip().lower()
        generator = (raw.get("generator") or "").strip()
        try:
            label = int(raw_label)
        except ValueError:
            errors.append(f"line {number}: label must be 0 or 1")
            continue
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = (manifest.parent / image_path).resolve()
        if label not in (0, 1):
            errors.append(f"line {number}: label must be 0 or 1")
        if split not in VALID_SPLITS:
            errors.append(f"line {number}: split must be train, validation, or test")
        if not generator:
            errors.append(f"line {number}: generator is required")
        if not raw_path:
            errors.append(f"line {number}: path is required")
        elif not image_path.is_file():
            errors.append(f"line {number}: image file not found: {image_path}")
        if image_path in seen_paths:
            errors.append(f"line {number}: duplicate image path: {image_path}")
        seen_paths.add(image_path)
        rows.append(ManifestRow(image_path, label, split, generator))

    if not rows:
        errors.append("manifest contains no rows")
    for split in VALID_SPLITS:
        if not any(row.split == split for row in rows):
            errors.append(f"manifest contains no split={split!r} rows")
    if errors:
        raise ManifestValidationError("\n".join(errors))
    return rows


def split_rows(rows: Iterable[ManifestRow], split: str) -> list[ManifestRow]:
    selected = [row for row in rows if row.split == split]
    if not selected:
        raise ValueError(f"Manifest contains no split={split!r} rows.")
    return selected


def leakage_report(rows: Iterable[ManifestRow]) -> dict[str, object]:
    materialized = list(rows)
    by_generator: dict[str, set[str]] = {}
    hashes: dict[str, list[str]] = {}
    for row in materialized:
        by_generator.setdefault(row.generator, set()).add(row.split)
        digest = hashlib.sha256(row.path.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(str(row.path))
    generator_overlap = {
        generator: sorted(splits)
        for generator, splits in by_generator.items()
        if len(splits) > 1
    }
    return {
        "duplicate_paths": [],
        "duplicate_content": [paths for paths in hashes.values() if len(paths) > 1],
        "generator_split_overlap": generator_overlap,
        "generator_source_leakage": bool(generator_overlap),
        "class_counts": {
            split: {
                str(label): sum(row.split == split and row.label == label for row in materialized)
                for label in (0, 1)
            }
            for split in VALID_SPLITS
        },
    }
