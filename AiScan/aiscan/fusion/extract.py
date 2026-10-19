from __future__ import annotations

import argparse
from pathlib import Path

from .scores import extract_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract real SAFE and PROBE logits into a CSV cache.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/scores.csv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    extract_scores(args.manifest, args.output, force=args.force)


if __name__ == "__main__":
    main()
