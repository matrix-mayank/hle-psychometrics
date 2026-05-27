# HLE Psychometrics

Running psychometric analysis on Humanity's Last Exam. We collected responses from 37 models (GPT, Claude, Gemini, DeepSeek, and open models) on 513 text-only multiple-choice questions, then built a binary response matrix for IRT and factor analysis.

Based on: **"Dimensionality and Measurement Precision in HLE's Multiple-Choice Subset"** (Sharma, Nadela, Matteson, Stanford 2026)

## What's in here

- Response collection scripts for multiple LLM APIs
- Binary response matrix (29 models × 428 items after filtering)
- IRT and CFA analysis notebooks
- Raw responses and scored data

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login  # HLE dataset is gated

# Add your API keys
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export DEEPSEEK_API_KEY=...
```

## Running it

```bash
# Export the items
PYTHONPATH=src python scripts/build_response_matrix.py export-items

# Test with a few items first
python scripts/collect_all_models.py --max-samples 5

# Full run (takes a few hours, costs vary by model)
python scripts/collect_all_models.py

# Build the matrix
PYTHONPATH=src python scripts/build_response_matrix.py build --min-coverage 0.95
```

The script saves responses as it goes, so you can stop and restart without losing progress.

## Models tested

**Reasoning models (3):** o3-mini, o4-mini, DeepSeek-R1  
**Standard models (26):** GPT-4.1, GPT-5 series, Claude Opus/Sonnet/Haiku 4.x, Gemini 2.5-3.5, DeepSeek V3/V4, open models (Qwen, Llama, Gemma, Phi, Mistral, etc.)

Final analysis used 29 models with 95%+ coverage. Accuracy ranged from 4.9% (GPT-4o) to 43.6% (Claude Opus 4.7).

## Output

```
data/matrix/
├── response_matrix.csv      # 29×428 binary matrix
├── items.csv                # Item metadata
├── models.csv               # Model metadata
└── responses_long.csv       # Long format for stats

analysis/
├── hle_psychometric_analysis.ipynb
├── hle_cfa_analysis.ipynb
└── hle_cfa_dimensionality.ipynb

figures/
└── *.png
```

## Notes

- We used `temperature=0` for all models that support it
- Responses were parsed with regex (not LLM-judged), hit 91.8% success rate
- Missing values (<1% after filtering) were filled with 0 for analysis
- Open models ran on vLLM with Modal + A100 GPUs
- 85 items had zero variance (all models got them wrong) and were dropped

See `pre-analysis-plan.pdf` for the full methodology.

## Citation

```
@unpublished{sharma2026hle,
  title={Dimensionality and Measurement Precision in HLE's Multiple-Choice Subset},
  author={Sharma, Mayank and Nadela, Savira and Matteson, Tyler},
  year={2026},
  institution={Stanford University}
}
```
