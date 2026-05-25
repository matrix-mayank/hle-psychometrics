"""Load and cache the text-only multiple-choice HLE analytic subset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

DATASET_ID = "cais/hle"
DATASET_SPLIT = "test"


def is_text_only(row: dict[str, Any]) -> bool:
    """Return True when the item has no image attachment."""
    image = row.get("image")
    if image is None:
        return True
    text = str(image).strip()
    return text in ("", "None", "null")


def is_analytic_item(row: dict[str, Any]) -> bool:
    """Text-only multiple-choice items used in the pre-analysis plan."""
    return row.get("answer_type") == "multipleChoice" and is_text_only(row)


def load_analytic_subset(*, use_cache: bool = True, cache_path: Path | None = None) -> pd.DataFrame:
    """
    Load the text-only multiple-choice subset (~513 items).

    If cache_path exists and use_cache is True, read from parquet instead of
    re-downloading from HuggingFace.
    """
    if cache_path is None:
        cache_path = Path("data/items/analytic_subset.parquet")

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    rows: list[dict[str, Any]] = []
    try:
        dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load cais/hle from HuggingFace. The dataset is gated: "
            "accept the license at https://huggingface.co/datasets/cais/hle, "
            "then run `huggingface-cli login` or set HF_TOKEN before export-items."
        ) from exc
    for row in dataset:
        if not is_analytic_item(row):
            continue
        rows.append(
            {
                "item_id": row["id"],
                "question": row["question"],
                "answer": str(row["answer"]).strip().upper(),
                "answer_type": row["answer_type"],
                "category": row["category"],
                "raw_subject": row["raw_subject"],
            }
        )

    items = pd.DataFrame(rows)
    if items.empty:
        raise RuntimeError("No analytic items found; check dataset filters.")

    items = items.sort_values("item_id").reset_index(drop=True)
    return items


def save_items(items: pd.DataFrame, path: Path) -> None:
    """Persist item metadata for reproducible matrix construction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(path, index=False)
    items.to_csv(path.with_suffix(".csv"), index=False)


def item_summary(items: pd.DataFrame) -> dict[str, Any]:
    """Quick counts for logging and sanity checks."""
    return {
        "n_items": len(items),
        "n_categories": items["category"].nunique(),
        "category_counts": items["category"].value_counts().to_dict(),
    }
