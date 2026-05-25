# HLE Response Matrix Builder

Build the binary response matrix for psychometric analysis of Humanity's Last Exam (HLE).

Based on the pre-analysis plan: **"Is HLE Measurement-Valid?"** (Sharma, Nadela, Matteson, Stanford 2026)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Authenticate with HuggingFace (HLE is gated)
huggingface-cli login

# 3. Export the 513 text-only multiple-choice items
PYTHONPATH=src python scripts/build_response_matrix.py export-items

# 4. Set API keys for model providers
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export DEEPSEEK_API_KEY=...
export OPENROUTER_API_KEY=...  # For open models
export TOGETHER_API_KEY=...     # For Llama, Qwen, etc.

# 5. Dry run to check configuration
python scripts/collect_all_models.py --dry-run

# 6. Smoke test (5 items per model)
python scripts/collect_all_models.py --max-samples 5

# 7. Full collection (513 items × 39 models = ~20k API calls)
python scripts/collect_all_models.py

# 8. Build the response matrix
PYTHONPATH=src python scripts/build_response_matrix.py build
```

## Output

After step 8, you'll have:

```
data/matrix/
├── response_matrix.csv      # 39 models × 513 items (0/1 values)
├── response_matrix.npy      # NumPy format for analysis
├── items.csv                # Item metadata (after zero-variance filter)
├── models.csv               # Model metadata
├── responses_long.csv       # Long format (model_id, item_id, correct)
└── build_report.json        # Dropped items, accuracies, etc.
```

## Models Configured (N=39)

### Reasoning-specialized (N=15)
- o3, o3-pro, o3-mini (high), o4-mini (high), o1 (high)
- DeepSeek-R1, DeepSeek-R1-Zero
- QwQ-32B, Qwen3-235B (thinking), Sky-T1
- Gemini 2.5 Pro/Flash (thinking)
- Claude 3.7 Sonnet (extended thinking)
- Grok 3 Thinking, Kimi K2 Thinking

### Standard instruction-tuned (N=24)
- GPT-4o, GPT-4.5, GPT-4.1, GPT-4-turbo
- Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus
- Gemini 2.0 Flash, Gemini 1.5 Pro
- Llama 3.1 405B, Llama 3.3 70B, Llama 4 Maverick
- DeepSeek-V3, Qwen2.5-72B, Qwen3-72B
- Mistral Large, Command R+, Phi-4, Gemma 3 27B
- Nemotron-4-340B, Yi-Large, Falcon-180B

## Cost & Time Estimates

| Scope | API calls | Estimated cost | Time (10 workers) |
|-------|-----------|----------------|-------------------|
| Smoke test (5 items) | ~195 | $5–20 | ~5 min |
| Pilot (10 models, full) | ~5,130 | $50–500 | ~2 hours |
| Full (39 models) | ~20,007 | $200–2000+ | ~8 hours |

**Note:** Reasoning models (o3, R1, etc.) with long chain-of-thought can be 10–100× more expensive than standard models.

## Resumability

All scripts resume from where they left off:
- `collect_all_models.py` skips items already in the JSON
- Use `--skip-existing` to skip models with complete responses
- Responses are saved incrementally (safe to Ctrl+C)

## Advanced Usage

### Collect specific models only
```bash
python scripts/collect_all_models.py --models gpt-4o-2024-11-20 claude-3-5-sonnet-20241022
```

### Run one model manually (OpenAI-compatible)
```bash
PYTHONPATH=src python scripts/run_api_predictions.py \
  --model gpt-4o-2024-11-20 \
  --output data/responses/raw/gpt-4o-2024-11-20.json
```

### Score a single model's predictions
```bash
PYTHONPATH=src python scripts/build_response_matrix.py score \
  --predictions data/responses/raw/gpt-4o-2024-11-20.json \
  --output data/responses/scored/gpt-4o-2024-11-20.csv
```

## Next Steps (Psychometric Analysis)

After building the matrix, use it for:
1. **2PL IRT** — item difficulty & discrimination, ability estimates
2. **Confirmatory Factor Analysis** — test if 8 HLE domains are distinct
3. **Measurement Invariance** — compare reasoning vs. standard model groups
4. **GPQA Diamond replication** — robustness check

See `pre-analysis-plan.pdf` for full methodology.

## Troubleshooting

### HuggingFace authentication
```bash
# Accept license at https://huggingface.co/datasets/cais/hle
huggingface-cli login
# Or set token
export HF_TOKEN=hf_...
```

### Missing API keys
```bash
python scripts/collect_all_models.py --dry-run  # Check which keys are needed
```

### Rate limits
```bash
# Reduce concurrent workers
python scripts/collect_all_models.py --num-workers 3
```

### Model unavailable
Edit `config/models.yaml` to comment out or update the `api_model_id`.

## File Structure

```
cs321m-project/
├── config/models.yaml           # Model registry (edit to add/remove models)
├── scripts/
│   ├── collect_all_models.py   # Multi-provider collection (main script)
│   ├── run_api_predictions.py  # Single-model OpenAI-compatible runner
│   └── build_response_matrix.py # Matrix assembly
├── src/hle_matrix/
│   ├── items.py      # Item filtering & caching
│   ├── scoring.py    # Answer extraction & binary scoring
│   └── matrix.py     # Matrix construction & variance filtering
├── data/
│   ├── items/analytic_subset.parquet  # Cached 513 items
│   ├── responses/raw/                 # Per-model prediction JSONs
│   └── matrix/                        # Final matrix outputs
└── tests/           # Unit tests
```

## Citation

```
@unpublished{sharma2026hle,
  title={Is HLE Measurement-Valid?},
  author={Sharma, Mayank and Nadela, Savira and Matteson, Tyler},
  year={2026},
  institution={Stanford University}
}
```
