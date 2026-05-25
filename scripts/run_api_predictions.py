#!/usr/bin/env python3
"""
Run HLE predictions for the analytic subset via an OpenAI-compatible API.

Requires OPENAI_API_KEY (or provider-specific base URL env vars).

This mirrors the official HLE eval prompt format so responses can be scored
with direct letter matching for multiple-choice items.

Example:
    python scripts/run_api_predictions.py \\
        --model gpt-4o-2024-11-20 \\
        --output data/responses/raw/gpt-4o-2024-11-20.json \\
        --max-samples 5   # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hle_matrix.items import load_analytic_subset

SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)


def format_messages(question: str, *, use_system_role: bool) -> list[dict]:
    role = "system" if use_system_role else "user"
    return [
        {"role": role, "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


async def run_predictions(
    items,
    *,
    model: str,
    output_path: Path,
    max_samples: int | None,
    num_workers: int,
    max_completion_tokens: int,
) -> None:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("Install openai: pip install openai") from exc

    client = AsyncOpenAI(timeout=600.0, max_retries=2)
    use_system = "o1" not in model and "o3" not in model

    questions = items.to_dict("records")
    if max_samples:
        questions = questions[:max_samples]

    if output_path.exists():
        with output_path.open() as f:
            predictions = json.load(f)
    else:
        predictions = {}

    pending = [q for q in questions if q["item_id"] not in predictions]
    semaphore = asyncio.Semaphore(num_workers)

    async def attempt(question: dict) -> tuple[str, dict] | None:
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    max_completion_tokens=max_completion_tokens,
                    messages=format_messages(
                        question["question"], use_system_role=use_system
                    ),
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                usage = json.loads(response.usage.model_dump_json())
            except Exception as exc:
                print(f"Error on {question['item_id']}: {exc}")
                return None
            if content is None:
                return None
            return question["item_id"], {
                "model": model,
                "response": content,
                "usage": usage,
            }

    from tqdm.asyncio import tqdm_asyncio

    results = await tqdm_asyncio.gather(*[attempt(q) for q in pending])
    for result in results:
        if result is None:
            continue
        item_id, payload = result
        predictions[item_id] = payload

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved {len(predictions)} predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items-path", type=Path, default=PROJECT_ROOT / "data/items/analytic_subset.parquet")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set.")

    items = load_analytic_subset(cache_path=args.items_path)
    asyncio.run(
        run_predictions(
            items,
            model=args.model,
            output_path=args.output,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            max_completion_tokens=args.max_completion_tokens,
        )
    )


if __name__ == "__main__":
    main()
