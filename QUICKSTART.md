# Quick Start Guide

Generate the 513×39 HLE response matrix in 4 steps.

## Step 1: Install & Authenticate (5 min)

```bash
cd /Users/mayanksharma/Desktop/cs321m-project

# Install Python packages
pip install -r requirements.txt

# HLE is gated - accept license at https://huggingface.co/datasets/cais/hle
huggingface-cli login
```

## Step 2: Export Items (2 min)

```bash
PYTHONPATH=src python scripts/build_response_matrix.py export-items
```

Expected output: `Saved 513 items to data/items/analytic_subset.parquet`

## Step 3: Set API Keys

```bash
# Required for the models in config/models.yaml
export OPENAI_API_KEY=sk-...        # o3, GPT-4o, etc.
export ANTHROPIC_API_KEY=sk-ant-... # Claude 3.5/3.7
export GOOGLE_API_KEY=...           # Gemini 2.5/2.0
export DEEPSEEK_API_KEY=...         # DeepSeek-R1/V3
export OPENROUTER_API_KEY=...       # Grok, Mistral, Phi-4, etc.
export TOGETHER_API_KEY=...         # Llama, Qwen, etc.
```

**Don't have all keys?** Edit `config/models.yaml` to comment out models you can't access.

## Step 4A: Smoke Test (10 min, ~$10)

Test with 5 items per model:

```bash
python scripts/collect_all_models.py --max-samples 5
```

## Step 4B: Full Collection (hours, $200–2000)

Once the smoke test works:

```bash
# Check which keys are missing
python scripts/collect_all_models.py --dry-run

# Full run (513 items × 39 models = 20,007 API calls)
python scripts/collect_all_models.py
```

**Tip:** This is resumable. Ctrl+C and restart anytime—it skips completed items.

## Step 5: Build Matrix (1 min)

```bash
PYTHONPATH=src python scripts/build_response_matrix.py build
```

Output: `data/matrix/response_matrix.csv` (39 rows × 513 columns, values 0/1)

---

## What if I only want 10 models?

```bash
# Edit config/models.yaml - comment out 29 models, keep 10
# Then run:
python scripts/collect_all_models.py --max-samples 5  # test first
python scripts/collect_all_models.py                  # full run
PYTHONPATH=src python scripts/build_response_matrix.py build
```

---

## Troubleshooting

### "Dataset 'cais/hle' is gated"
→ Accept the license at https://huggingface.co/datasets/cais/hle and run `huggingface-cli login`

### "Missing API keys: OPENAI_API_KEY"
→ Set the environment variable or remove models from `config/models.yaml`

### "Rate limit exceeded"
→ Lower `--num-workers`: `python scripts/collect_all_models.py --num-workers 3`

### Script hangs or times out
→ Ctrl+C and restart. Already-completed items are saved and will be skipped.

---

## Cost Control

| Strategy | How |
|----------|-----|
| **Test first** | Use `--max-samples 5` to verify everything works |
| **Start small** | Edit `config/models.yaml` to only 5–10 models initially |
| **Skip expensive models** | Comment out o3-pro, o3, GPT-4.5 (most expensive reasoning models) |
| **Use checkpoints** | Scripts auto-save; run in batches and check costs between runs |

---

## Next: Analysis

Once you have `data/matrix/response_matrix.csv`, use it for:
- **2PL IRT** (item difficulty, discrimination, ability estimates)
- **CFA** (test if 8 HLE domains are empirically distinct)
- **Measurement invariance** (reasoning vs. standard model groups)

See `pre-analysis-plan.pdf` for full methodology.
