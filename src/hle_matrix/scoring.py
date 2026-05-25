"""Score multiple-choice model responses against ground-truth letters."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

# Official HLE prompt asks models to respond with:
#   Answer: {chosen answer}
ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*answer\s*:\s*(.+?)\s*$",
)
# Fallback: isolated letter A–Z (HLE has answers up to V)
LETTER_RE = re.compile(r"\b([A-Z])\b", re.IGNORECASE)


def extract_answer_letter(response: str) -> str | None:
    """
    Extract the model's chosen answer letter from a raw response string.

    Priority:
    1. Explicit "Answer: ..." line (HLE eval format)
    2. Last standalone letter A–E in the text
    """
    if not response or not str(response).strip():
        return None

    text = str(response).strip()
    matches = ANSWER_LINE_RE.findall(text)
    if matches:
        candidate = matches[-1].strip()
        letter = LETTER_RE.search(candidate)
        if letter:
            return letter.group(1).upper()
        # Some models return the full choice text; take first letter if valid
        if len(candidate) == 1 and candidate.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return candidate.upper()

    letters = LETTER_RE.findall(text)
    if letters:
        return letters[-1].upper()

    return None


def load_predictions(path: Path) -> dict[str, str]:
    """
    Load per-item raw responses from an HLE-style predictions JSON file.

    Expected format (from hle_eval/run_model_predictions.py):
        {item_id: {"model": "...", "response": "...", "usage": {...}}}
    """
    with path.open() as f:
        data = json.load(f)

    responses: dict[str, str] = {}
    for item_id, payload in data.items():
        if isinstance(payload, str):
            responses[item_id] = payload
        elif isinstance(payload, dict) and "response" in payload:
            responses[item_id] = payload["response"]
        else:
            raise ValueError(
                f"Unexpected payload for item {item_id} in {path}: {type(payload)}"
            )
    return responses


def score_responses(
    items: pd.DataFrame,
    raw_responses: dict[str, str],
) -> pd.Series:
    """
    Return a Series indexed by item_id with values in {0, 1, NaN}.

    NaN means the model did not provide a scorable response for that item.
    """
    correct_answers = items.set_index("item_id")["answer"]
    scores: dict[str, float] = {}

    for item_id, truth in correct_answers.items():
        response = raw_responses.get(item_id)
        if response is None:
            scores[item_id] = float("nan")
            continue

        predicted = extract_answer_letter(response)
        if predicted is None:
            scores[item_id] = float("nan")
        else:
            scores[item_id] = float(predicted == truth)

    return pd.Series(scores, name="correct").sort_index()


def score_model_responses(
    items: pd.DataFrame,
    predictions_path: Path,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Score one model's predictions and optionally write a scored CSV.

    Columns: item_id, predicted_letter, correct (0/1), missing_response
    """
    raw = load_predictions(predictions_path)
    rows = []
    answer_lookup = items.set_index("item_id")["answer"]

    for item_id in items["item_id"]:
        truth = answer_lookup[item_id]
        response = raw.get(item_id)
        predicted = extract_answer_letter(response) if response is not None else None
        rows.append(
            {
                "item_id": item_id,
                "ground_truth": truth,
                "predicted_letter": predicted,
                "correct": float("nan")
                if predicted is None
                else float(predicted == truth),
                "missing_response": response is None,
            }
        )

    scored = pd.DataFrame(rows)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)

    return scored
