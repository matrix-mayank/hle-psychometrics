#!/usr/bin/env python3
"""
Collect HLE predictions for all models in config/models.yaml.

Supports multiple providers: OpenAI, Anthropic, Google, DeepSeek, OpenRouter, Together.

Usage:
    # Dry run to check configuration
    python scripts/collect_all_models.py --dry-run

    # Run 5 items per model (smoke test)
    python scripts/collect_all_models.py --max-samples 5

    # Full collection (513 items × all models)
    python scripts/collect_all_models.py

    # Run specific models only
    python scripts/collect_all_models.py --models gpt-4o-2024-11-20 claude-3-5-sonnet-20241022

    # Skip models that already have responses
    python scripts/collect_all_models.py --skip-existing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hle_matrix.items import load_analytic_subset
from hle_matrix.matrix import load_model_registry

SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)


class ProviderClient:
    """Unified interface for different model providers."""

    def __init__(self, provider: str, model_entry: dict[str, Any]):
        self.provider = provider
        self.model_entry = model_entry
        self.api_model_id = model_entry["api_model_id"]
        self.client = None

    async def initialize(self):
        """Initialize provider-specific client."""
        if self.provider == "openai":
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"), timeout=600.0, max_retries=2
            )
        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=600.0, max_retries=2
            )
        elif self.provider == "google":
            import google.generativeai as genai

            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self.client = genai.GenerativeModel(self.api_model_id)
        elif self.provider == "deepseek":
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                timeout=600.0,
                max_retries=2,
            )
        elif self.provider in ("openrouter", "together"):
            from openai import AsyncOpenAI

            key_var = f"{self.provider.upper()}_API_KEY"
            base_urls = {
                "openrouter": "https://openrouter.ai/api/v1",
                "together": "https://api.together.xyz/v1",
            }
            self.client = AsyncOpenAI(
                api_key=os.environ.get(key_var),
                base_url=base_urls[self.provider],
                timeout=600.0,
                max_retries=2,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def complete(self, question: str, max_tokens: int = 8192) -> str | None:
        """Get completion from the model."""
        try:
            if self.provider == "anthropic":
                # Anthropic uses different message format
                kwargs = {
                    "model": self.api_model_id,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{question}"}
                    ],
                }
                # Only add temperature for models that support it
                # Claude Opus 4.7 and other reasoning models don't support temperature
                if "opus-4-7" not in self.api_model_id:
                    kwargs["temperature"] = 0.0
                
                message = await self.client.messages.create(**kwargs)
                return message.content[0].text

            elif self.provider == "google":
                # Google Generative AI
                response = await asyncio.to_thread(
                    self.client.generate_content,
                    f"{SYSTEM_PROMPT}\n\n{question}",
                    generation_config={"temperature": 0.0, "max_output_tokens": max_tokens},
                )
                return response.text

            else:
                # OpenAI-compatible (OpenAI, DeepSeek, OpenRouter, Together)
                # Reasoning models: o1, o3, o4, gpt-5 series
                is_reasoning = any(x in self.api_model_id for x in ["o1", "o3", "o4", "gpt-5"])
                
                messages = []
                if not is_reasoning:
                    messages.append({"role": "system", "content": SYSTEM_PROMPT})
                    messages.append({"role": "user", "content": question})
                else:
                    # Reasoning models don't support system role
                    messages.append({"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{question}"})

                kwargs = {
                    "model": self.api_model_id,
                    "messages": messages,
                }
                
                # Only add temperature for non-reasoning models
                if not is_reasoning:
                    kwargs["temperature"] = 0.0
                
                # Add max_completion_tokens for reasoning models, max_tokens for others
                if is_reasoning:
                    kwargs["max_completion_tokens"] = max_tokens
                    # Add reasoning_effort if specified
                    if "reasoning_effort" in self.model_entry:
                        kwargs["reasoning_effort"] = self.model_entry["reasoning_effort"]
                else:
                    # Use model-specific max_tokens if specified, otherwise use default
                    model_max_tokens = self.model_entry.get("max_tokens", max_tokens)
                    kwargs["max_tokens"] = model_max_tokens

                response = await self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

        except Exception as e:
            print(f"Error calling {self.provider}/{self.api_model_id}: {e}")
            return None


async def collect_model_responses(
    model_entry: dict[str, Any],
    items: list[dict],
    output_path: Path,
    *,
    max_samples: int | None = None,
    num_workers: int = 10,
    max_tokens: int = 8192,
) -> dict[str, int]:
    """Collect responses for one model."""
    model_id = model_entry["id"]
    provider = model_entry["provider"]

    print(f"\n{'='*60}")
    print(f"Model: {model_entry['display_name']} ({model_id})")
    print(f"Provider: {provider} | API model: {model_entry['api_model_id']}")
    print(f"{'='*60}")

    # Load existing predictions
    if output_path.exists():
        with output_path.open() as f:
            predictions = json.load(f)
        print(f"Found {len(predictions)} existing responses")
    else:
        predictions = {}

    # Filter to pending items
    if max_samples:
        items = items[:max_samples]
    pending = [item for item in items if item["item_id"] not in predictions]

    if not pending:
        print(f"✓ All {len(items)} items already complete")
        return {"total": len(items), "collected": 0, "existing": len(predictions)}

    print(f"Collecting {len(pending)} items (skipping {len(predictions)} existing)...")

    # Initialize provider client
    client = ProviderClient(provider, model_entry)
    await client.initialize()

    # Semaphore for rate limiting
    semaphore = asyncio.Semaphore(num_workers)

    async def attempt_item(item: dict) -> tuple[str, dict] | None:
        async with semaphore:
            response = await client.complete(item["question"], max_tokens=max_tokens)
            if response is None:
                return None
            return item["item_id"], {
                "model": model_id,
                "response": response,
            }

    # Collect with progress bar
    from tqdm.asyncio import tqdm_asyncio

    results = await tqdm_asyncio.gather(*[attempt_item(item) for item in pending])

    # Update predictions
    collected = 0
    for result in results:
        if result is not None:
            item_id, payload = result
            predictions[item_id] = payload
            collected += 1

    # Save incrementally
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(predictions, f, indent=2)

    print(f"✓ Saved {len(predictions)} total responses ({collected} new)")
    return {"total": len(items), "collected": collected, "existing": len(predictions) - collected}


async def collect_all(
    models: list[dict[str, Any]],
    items: list[dict],
    *,
    max_samples: int | None = None,
    skip_existing: bool = False,
    num_workers: int = 10,
) -> None:
    """Collect responses for all models sequentially."""
    results = {}

    for model_entry in models:
        model_id = model_entry["id"]
        output_path = PROJECT_ROOT / model_entry["response_file"]

        # Skip if already complete and skip_existing is True
        if skip_existing and output_path.exists():
            with output_path.open() as f:
                existing = json.load(f)
            if len(existing) >= len(items):
                print(f"\n⊘ Skipping {model_id} (already complete)")
                continue

        try:
            stats = await collect_model_responses(
                model_entry,
                items,
                output_path,
                max_samples=max_samples,
                num_workers=num_workers,
            )
            results[model_id] = stats
        except Exception as e:
            print(f"\n✗ Failed {model_id}: {e}")
            results[model_id] = {"error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    for model_id, stats in results.items():
        if "error" in stats:
            print(f"✗ {model_id}: {stats['error']}")
        else:
            print(
                f"✓ {model_id}: {stats['collected']} new, {stats['existing']} existing, {stats['total']} total"
            )


def check_api_keys(models: list[dict[str, Any]]) -> list[str]:
    """Check which API keys are missing."""
    required_keys = set()
    for model in models:
        provider = model["provider"]
        if provider == "openai":
            required_keys.add("OPENAI_API_KEY")
        elif provider == "anthropic":
            required_keys.add("ANTHROPIC_API_KEY")
        elif provider == "google":
            required_keys.add("GOOGLE_API_KEY")
        elif provider == "deepseek":
            required_keys.add("DEEPSEEK_API_KEY")
        elif provider == "openrouter":
            required_keys.add("OPENROUTER_API_KEY")
        elif provider == "together":
            required_keys.add("TOGETHER_API_KEY")

    missing = [key for key in required_keys if not os.environ.get(key)]
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model IDs to collect (default: all from config)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit to first N items per model (for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have complete responses",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Concurrent requests per model (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit",
    )
    parser.add_argument(
        "--items-path",
        type=Path,
        default=PROJECT_ROOT / "data/items/analytic_subset.parquet",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=PROJECT_ROOT / "config/models.yaml",
    )
    args = parser.parse_args()

    # Load items and models
    print("Loading analytic subset...")
    items_df = load_analytic_subset(cache_path=args.items_path)
    items = items_df.to_dict("records")
    print(f"Loaded {len(items)} items")

    print(f"\nLoading model registry from {args.models_config}...")
    all_models = load_model_registry(args.models_config)

    # Filter to requested models
    if args.models:
        all_models = [m for m in all_models if m["id"] in args.models]
        if not all_models:
            print(f"Error: No models found matching {args.models}")
            sys.exit(1)

    print(f"Selected {len(all_models)} models")

    # Check API keys
    missing_keys = check_api_keys(all_models)
    if missing_keys:
        print(f"\n⚠ Warning: Missing API keys: {', '.join(missing_keys)}")
        print("Some models may fail. Set environment variables before running.")

    # Dry run: print config and exit
    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN - Configuration check")
        print(f"{'='*60}")
        for model in all_models:
            status = "✓" if model["provider"] not in [k.split("_")[0].lower() for k in missing_keys] else "✗"
            print(f"{status} {model['display_name']:30s} | {model['provider']:12s} | {model['api_model_id']}")
        print(f"\nTotal: {len(items)} items × {len(all_models)} models = {len(items) * len(all_models)} API calls")
        sys.exit(0)

    # Run collection
    asyncio.run(
        collect_all(
            all_models,
            items,
            max_samples=args.max_samples,
            skip_existing=args.skip_existing,
            num_workers=args.num_workers,
        )
    )


if __name__ == "__main__":
    main()
