"""Assemble the N x J binary response matrix and apply variance filters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .scoring import load_predictions, score_responses


def load_model_registry(config_path: Path) -> list[dict[str, Any]]:
    with config_path.open() as f:
        config = yaml.safe_load(f)
    return config["models"]


def build_response_matrix(
    items: pd.DataFrame,
    model_entries: list[dict[str, Any]],
    *,
    min_coverage: float = 1.0,
    drop_zero_variance: bool = True,
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build binary response matrix X with shape (N_models, J_items).

    Parameters
    ----------
    items : DataFrame with columns item_id, answer, ...
    model_entries : list from config/models.yaml
    min_coverage : minimum fraction of items with non-NaN scores to include a model
    drop_zero_variance : remove items where all included models are 0 or all 1

    Returns
    -------
    matrix : DataFrame, index=model_id, columns=item_id, values in {0.0, 1.0}
    report : dict with metadata and dropped item/model info
    """
    if project_root is None:
        project_root = Path.cwd()

    item_ids = items["item_id"].tolist()
    matrix_rows: dict[str, pd.Series] = {}
    skipped_models: list[dict[str, str]] = []

    for entry in model_entries:
        model_id = entry["id"]
        response_path = project_root / entry["response_file"]
        if not response_path.exists():
            skipped_models.append(
                {"model_id": model_id, "reason": f"missing file: {response_path}"}
            )
            continue

        raw = load_predictions(response_path)
        scores = score_responses(items, raw)
        coverage = scores.notna().mean()
        if coverage < min_coverage:
            skipped_models.append(
                {
                    "model_id": model_id,
                    "reason": f"coverage {coverage:.1%} < {min_coverage:.0%}",
                }
            )
            continue

        matrix_rows[model_id] = scores

    if not matrix_rows:
        raise RuntimeError(
            "No models with response files found. "
            "Add JSON files under data/responses/raw/ and update config/models.yaml."
        )

    matrix = pd.DataFrame(matrix_rows).T
    matrix = matrix.reindex(columns=item_ids)

    dropped_items: list[str] = []
    if drop_zero_variance:
        # Only use items where every included model has a score
        complete = matrix.dropna(axis=1, how="any")
        variances = complete.var(axis=0, ddof=0)
        keep_cols = variances[variances > 0].index.tolist()
        dropped_items = [c for c in matrix.columns if c not in keep_cols]
        matrix = matrix[keep_cols]

    report = {
        "n_models": matrix.shape[0],
        "n_items": matrix.shape[1],
        "n_items_before_filter": len(item_ids),
        "n_items_dropped_zero_variance": len(dropped_items),
        "dropped_item_ids": dropped_items,
        "skipped_models": skipped_models,
        "model_ids": matrix.index.tolist(),
        "mean_accuracy": matrix.mean(axis=1).to_dict(),
    }
    return matrix, report


def save_matrix_artifacts(
    matrix: pd.DataFrame,
    items: pd.DataFrame,
    model_entries: list[dict[str, Any]],
    report: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write matrix CSV/npy, model metadata, item metadata, and report JSON."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    matrix.to_csv(output_dir / "response_matrix.csv")
    np.save(output_dir / "response_matrix.npy", matrix.values)

    model_meta = pd.DataFrame(model_entries)
    model_meta = model_meta[model_meta["id"].isin(matrix.index)]
    model_meta.to_csv(output_dir / "models.csv", index=False)

    item_meta = items[items["item_id"].isin(matrix.columns)].copy()
    item_meta.to_csv(output_dir / "items.csv", index=False)

    with (output_dir / "build_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    # Long format for inspection / merging with other tools
    long = matrix.reset_index().melt(
        id_vars="index",
        var_name="item_id",
        value_name="correct",
    )
    long = long.rename(columns={"index": "model_id"})
    long.to_csv(output_dir / "responses_long.csv", index=False)
