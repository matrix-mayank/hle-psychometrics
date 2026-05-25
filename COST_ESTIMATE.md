# Cost Estimate for 34-Model Configuration

**Current config:** 34 active models (36 total, 3 expensive ones commented out)

## Budget-Friendly Configuration ✓

| Category | Models | Est. Cost |
|----------|--------|-----------|
| **Reasoning (11 models)** | o3-mini, o1, DeepSeek-R1/R1-Zero, QwQ, Qwen3-235B, Gemini 2.5 thinking, Claude 3.7 thinking, Grok 3, Kimi K2, Sky-T1 | $75-250 |
| **Standard (23 models)** | GPT-4o/4.5/4.1/turbo, Claude 3.5/3 series, Gemini 2.0/1.5, Llama 3.1/3.3/4, DeepSeek-V3, Qwen 2.5/3, Mistral, Command R+, Phi-4, Gemma, Nemotron, Yi, Falcon | $30-80 |
| **TOTAL** | 34 models × 513 questions = 17,442 calls | **$105-330** |

## What Was Removed (to save ~$1000-7000)

| Model | Why removed | Est. saved |
|-------|-------------|------------|
| o3 | Extremely expensive reasoning | $500-2000 |
| o3-pro | Most expensive model available | $500-5000 |
| o4-mini (high) | High reasoning effort = high cost | $200-500 |

## How to Lower Cost Further

### Option 1: Pilot Run (10 models, ~$30)
Edit `config/models.yaml` and keep only:
```yaml
# Reasoning (5):
- o3-mini (high), o1 (high), DeepSeek-R1, QwQ-32B, Gemini 2.5 Pro thinking

# Standard (5):
- GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash, Llama 3.1 405B, DeepSeek-V3
```

### Option 2: Lower Reasoning Effort
Change `reasoning_effort: high` → `reasoning_effort: medium` in config.
- Saves ~50% on reasoning models
- New total: **$70-220**

### Option 3: Use Free Tiers First
Some providers offer free credits:
- Together AI: $25 free
- OpenRouter: Some models have free tier
- Google: Gemini has generous free tier

## Smoke Test First (Recommended)

```bash
# Test with 5 questions per model (~$3-10 total)
python scripts/collect_all_models.py --max-samples 5

# Check actual costs in your provider dashboards
# Then decide: continue with full run or adjust config
```

## Cost Tracking

After smoke test, check:
- OpenAI: https://platform.openai.com/usage
- Anthropic: https://console.anthropic.com/settings/cost
- Google: https://console.cloud.google.com/billing
- Others: check respective dashboards

Multiply by ~100 to estimate full run cost.

## Re-enable Expensive Models Later

If you secure more budget or want to add o3/o3-pro later:
1. Uncomment them in `config/models.yaml`
2. Run: `python scripts/collect_all_models.py --models o3-2025-04-16 o3-pro-2025-04-16`
3. Rebuild matrix: `PYTHONPATH=src python scripts/build_response_matrix.py build`

The matrix builder will automatically include the new models.
