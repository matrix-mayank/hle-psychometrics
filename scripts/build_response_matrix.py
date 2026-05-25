#!/usr/bin/env python3
"""
CLI for building the HLE text-only multiple-choice response matrix.

Examples
--------
# 1. Export and cache the analytic item subset
python scripts/build_response_matrix.py export-items

# 2. Score one model's predictions (HLE eval JSON format)
python scripts/build_response_matrix.py score \\
    --predictions data/responses/raw/gpt-4o-2024-11-20.json \\
    --output data/responses/scored/gpt-4o-2024-11-20.csv

# 3. Build the full matrix once response files exist
python scripts/build_response_matrix.py build
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hle_matrix.items import item_summary, load_analytic_subset, save_items
from hle_matrix.matrix import (
    build_response_matrix,
    load_model_registry,
    save_matrix_artifacts,
)
from hle_matrix.scoring import score_model_responses


def cmd_export_items(args: argparse.Namespace) -> None:
    items = load_analytic_subset(use_cache=not args.refresh, cache_path=args.items_path)
    save_items(items, args.items_path)
    summary = item_summary(items)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {summary['n_items']} items to {args.items_path}")


def cmd_score(args: argparse.Namespace) -> None:
    items = load_analytic_subset(cache_path=args.items_path)
    scored = score_model_responses(items, args.predictions, output_path=args.output)
    n_correct = scored["correct"].sum()
    n_scored = scored["correct"].notna().sum()
    acc = n_correct / n_scored if n_scored else 0.0
    print(f"Scored {n_scored}/{len(items)} items | accuracy = {acc:.1%}")
    if args.output:
        print(f"Wrote {args.output}")


def cmd_build(args: argparse.Namespace) -> None:
    items = load_analytic_subset(cache_path=args.items_path)
    models = load_model_registry(args.models_config)
    matrix, report = build_response_matrix(
        items,
        models,
        min_coverage=args.min_coverage,
        drop_zero_variance=not args.keep_zero_variance,
        project_root=PROJECT_ROOT,
    )
    save_matrix_artifacts(matrix, items, models, report, args.output_dir)
    print(json.dumps(report, indent=2))
    print(f"\nMatrix shape: {matrix.shape[0]} models x {matrix.shape[1]} items")
    print(f"Saved artifacts to {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items-path",
        type=Path,
        default=PROJECT_ROOT / "data/items/analytic_subset.parquet",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export-items", help="Download/cache analytic subset")
    export.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download from HuggingFace even if cache exists",
    )
    export.set_defaults(func=cmd_export_items)

    score = sub.add_parser("score", help="Score one model's prediction file")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, default=None)
    score.set_defaults(func=cmd_score)

    build = sub.add_parser("build", help="Assemble matrix from all model files")
    build.add_argument(
        "--models-config",
        type=Path,
        default=PROJECT_ROOT / "config/models.yaml",
    )
    build.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/matrix",
    )
    build.add_argument(
        "--min-coverage",
        type=float,
        default=1.0,
        help="Min fraction of items with responses to include a model (default 1.0)",
    )
    build.add_argument(
        "--keep-zero-variance",
        action="store_true",
        help="Keep items with no variance across models (default: drop them)",
    )
    build.set_defaults(func=cmd_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
