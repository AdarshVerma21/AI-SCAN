from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from timm import create_model
from timm.data import create_transform, resolve_data_config


REQUIRED_COLUMNS = {"filepath", "label_id", "source_video_ids"}
MODEL_NAME = "xception41"


@dataclass(frozen=True)
class Row:
    filepath: Path
    label: int
    source_video_ids: tuple[str, ...]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_validate_manifest(manifest: Path) -> list[Row]:
    if not manifest.is_file():
        raise ValueError(f"Manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        rows: list[Row] = []
        errors: list[str] = []
        seen: set[Path] = set()
        for line, raw in enumerate(reader, start=2):
            raw_path = (raw.get("filepath") or "").strip()
            path = Path(raw_path)
            if not path.is_absolute():
                path = (manifest.parent / path).resolve()
            raw_label = (raw.get("label_id") or "").strip()
            raw_groups = (raw.get("source_video_ids") or "").strip()
            try:
                label = int(raw_label)
            except ValueError:
                errors.append(f"line {line}: label_id must be 0 or 1")
                continue
            groups = tuple(part.strip() for part in raw_groups.split("|") if part.strip())
            if label not in (0, 1):
                errors.append(f"line {line}: label_id must be 0 or 1")
            if not raw_path or not path.is_file():
                errors.append(f"line {line}: image does not exist: {path}")
            if not groups:
                errors.append(f"line {line}: source_video_ids is required")
            if path in seen:
                errors.append(f"line {line}: duplicate filepath: {path}")
            seen.add(path)
            rows.append(Row(path, label, groups))
    if errors:
        raise ValueError("Manifest validation failed:\n" + "\n".join(errors[:20]))
    if not rows:
        raise ValueError("Manifest contains no rows")
    label_counts = {label: sum(row.label == label for row in rows) for label in (0, 1)}
    if label_counts != {0: 200, 1: 200}:
        raise ValueError(
            "Manifest must contain exactly 200 examples of each class; "
            f"observed {label_counts}"
        )
    return rows


def _connected_group_ids(rows: list[Row]) -> list[str]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in rows:
        for group in row.source_video_ids:
            find(group)
        for group in row.source_video_ids[1:]:
            union(row.source_video_ids[0], group)
    return [find(row.source_video_ids[0]) for row in rows]


def _split_score(indices: np.ndarray, rows: list[Row], target_size: float) -> float:
    labels = np.asarray([rows[index].label for index in indices])
    size_penalty = abs(len(indices) / len(rows) - target_size)
    class_penalty = abs(float(labels.mean()) - 0.5) if len(labels) else 1.0
    return size_penalty + class_penalty


def make_group_splits(rows: list[Row], seed: int) -> dict[str, list[int]]:
    groups = np.asarray(_connected_group_ids(rows))
    all_indices = np.arange(len(rows))
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for attempt in range(64):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=seed + attempt
        )
        train_val, test = next(splitter.split(all_indices, groups=groups))
        score = _split_score(test, rows, 0.20)
        if best is None or score < best[0]:
            best = (score, train_val, test)
    assert best is not None
    train_val, test = best[1], best[2]
    best_val: tuple[float, np.ndarray, np.ndarray] | None = None
    train_val_groups = groups[train_val]
    for attempt in range(64):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=seed + 100 + attempt
        )
        train_relative, validation_relative = next(
            splitter.split(train_val, groups=train_val_groups)
        )
        validation = train_val[validation_relative]
        score = _split_score(validation, rows, 0.16)
        if best_val is None or score < best_val[0]:
            best_val = (score, train_val[train_relative], validation)
    assert best_val is not None
    splits = {
        "train": sorted(best_val[1].tolist()),
        "validation": sorted(best_val[2].tolist()),
        "test": sorted(test.tolist()),
    }
    memberships = {index: split for split, values in splits.items() for index in values}
    if len(memberships) != len(rows):
        raise ValueError("Group split did not assign every manifest row exactly once")
    seen_groups: dict[str, set[str]] = {}
    seen_source_ids: dict[str, set[str]] = {}
    for index, split in memberships.items():
        seen_groups.setdefault(groups[index], set()).add(split)
        for source_id in rows[index].source_video_ids:
            seen_source_ids.setdefault(source_id, set()).add(split)
    leaked = [group for group, assigned in seen_groups.items() if len(assigned) > 1]
    source_leaks = [group for group, assigned in seen_source_ids.items() if len(assigned) > 1]
    if leaked or source_leaks:
        raise ValueError(
            "Source-video overlap detected across splits: "
            f"{(source_leaks or leaked)[:5]}"
        )
    return splits


class ImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, rows: list[Row], indices: Iterable[int], transform: transforms.Compose):
        self.rows = rows
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        row = self.rows[self.indices[item]]
        with Image.open(row.filepath) as image:
            image = image.convert("RGB")
            return self.transform(image), row.label


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )[0]),
        "recall": float(precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )[1]),
        "f1": float(precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )[2]),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "per_class": {
            "real_0": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0])},
            "fake_1": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1])},
        },
        "class_counts": {"real_0": int((labels == 0).sum()), "fake_1": int((labels == 1).sum())},
    }
    result["false_positive_rate"] = float(
        result["confusion_matrix"][0][1] / max(1, result["class_counts"]["real_0"])
    )
    return result


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    probabilities: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            losses.append(float(criterion(logits, targets.to(device)).item()))
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
            labels.extend(targets.numpy().tolist())
    return float(np.mean(losses)), np.asarray(labels), np.asarray(probabilities)


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    rows = load_and_validate_manifest(manifest)
    splits = make_group_splits(rows, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = create_model(MODEL_NAME, pretrained=True, num_classes=2)
    model.to(device)
    data_config = resolve_data_config({}, model=model)
    train_transform = create_transform(**data_config, is_training=True)
    eval_transform = create_transform(**data_config, is_training=False)
    loaders = {
        "train": DataLoader(ImageDataset(rows, splits["train"], train_transform), args.batch_size, shuffle=True, num_workers=0),
        "validation": DataLoader(ImageDataset(rows, splits["validation"], eval_transform), args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(ImageDataset(rows, splits["test"], eval_transform), args.batch_size, shuffle=False, num_workers=0),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, float | int]] = []
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for images, targets in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        validation_loss, _, _ = _evaluate(model, loaders["validation"], device)
        entry = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": validation_loss}
        history.append(entry)
        if validation_loss < best_loss:
            best_loss, best_epoch, wait = validation_loss, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": MODEL_NAME,
                "num_classes": 2,
                "seed": args.seed,
                "best_validation_loss": best_loss,
                "epoch": epoch,
                "preprocessing": data_config,
            }, checkpoint)
        else:
            wait += 1
            if wait >= args.patience:
                break
    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    test_loss, test_labels, test_probabilities = _evaluate(model, loaders["test"], device)
    validation_loss, validation_labels, validation_probabilities = _evaluate(model, loaders["validation"], device)
    metrics = {
        "test_loss": test_loss,
        "test": _metrics(test_labels, test_probabilities),
        "validation": _metrics(validation_labels, validation_probabilities),
    }
    (output / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    split_payload = {
        split: [
            {"filepath": str(rows[index].filepath), "label_id": rows[index].label,
             "source_video_ids": list(rows[index].source_video_ids)}
            for index in indices
        ] for split, indices in splits.items()
    }
    (output / "splits.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    config = {
        "manifest": str(manifest),
        "seed": args.seed,
        "device": str(device),
        "architecture": MODEL_NAME,
        "pretrained_weights": "timm ImageNet pretrained weights",
        "input_size": data_config["input_size"],
        "mean": data_config["mean"],
        "std": data_config["std"],
        "interpolation": str(data_config["interpolation"]),
        "best_epoch": best_epoch,
        "early_stopping_patience": args.patience,
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "class_counts": {
            key: {str(label): sum(rows[index].label == label for index in value) for label in (0, 1)}
            for key, value in splits.items()
        },
        "original_path_used": False,
    }
    (output / "preprocessing.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a group-aware binary deepfake image classifier.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/deepfake_classifier"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
