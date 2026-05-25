"""Tests for response scoring helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hle_matrix.matrix import build_response_matrix
from hle_matrix.scoring import extract_answer_letter, score_responses


def test_extract_answer_letter_from_hle_format() -> None:
    response = (
        "Explanation: Critical-level views violate this condition.\n"
        "Answer: D\n"
        "Confidence: 85%"
    )
    assert extract_answer_letter(response) == "D"


def test_build_matrix_drops_zero_variance_items() -> None:
    items = pd.DataFrame(
        {
            "item_id": ["q1", "q2", "q3"],
            "answer": ["A", "B", "C"],
            "category": ["Math"] * 3,
            "raw_subject": ["Algebra"] * 3,
            "answer_type": ["multipleChoice"] * 3,
            "question": ["?"] * 3,
        }
    )
    model_entries = [
        {
            "id": "model-a",
            "response_file": "tests/fixtures/model_a.json",
            "group": "standard",
        },
        {
            "id": "model-b",
            "response_file": "tests/fixtures/model_b.json",
            "group": "reasoning",
        },
    ]
    root = Path(__file__).resolve().parents[1]
    matrix, report = build_response_matrix(
        items,
        model_entries,
        drop_zero_variance=True,
        project_root=root,
    )
    assert matrix.shape == (2, 2)
    assert "q1" in report["dropped_item_ids"]  # both models correct on q1


def test_score_responses() -> None:
    items = pd.DataFrame({"item_id": ["q1"], "answer": ["A"]})
    scores = score_responses(items, {"q1": "Answer: A"})
    assert scores["q1"] == 1.0
