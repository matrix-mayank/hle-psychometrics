# HLE Psychometrics

Running psychometric analysis on Humanity's Last Exam. We collected responses from 37 models (GPT, Claude, Gemini, DeepSeek, and open models) on 513 text-only multiple-choice questions, then built a binary response matrix for IRT and factor analysis.

Based on: **"Dimensionality and Measurement Precision in HLE's Multiple-Choice Subset"** (Sharma, Nadela, Matteson, Stanford 2026)

## What's in here

- Response collection scripts for multiple LLM APIs
- Binary response matrix (29 models × 428 items after filtering)
- IRT and CFA analysis notebooks
- Raw responses and scored data

## Repository Structure

```
cs321m-project/
├── config/
│   └── models.yaml              # Model configurations (37 models)
├── scripts/
│   ├── collect_all_models.py    # Main data collection script
│   ├── build_response_matrix.py # Matrix construction
│   └── run_vllm.py              # Open-weight model inference
├── src/hle_matrix/
│   ├── items.py                 # HLE dataset loading & filtering
│   ├── scoring.py               # Response parsing & scoring
│   └── matrix.py                # Matrix building logic
├── data/
│   ├── items/                   # Cached HLE subset
│   ├── responses/raw/           # Per-model API responses
│   └── matrix/                  # Final matrices & metadata
├── analysis/
│   ├── hle_2pl_item_analysis.ipynb       # 2PL IRT model & item parameters
│   └── hle_cfa_dimensionality.ipynb      # Dimensionality analysis & figures
├── tests/                       # Unit tests for scoring logic
└── requirements.txt             # Python dependencies (pinned versions)
```

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

## Computational Requirements

**Data Collection**
- Time: ~20 hours total (limited by API rate limits)
- Cost: $105-330 for full collection (varies by provider)
- Hardware: No GPU needed (uses API endpoints)

**Open-Weight Models** (if applicable)
- GPU: Nvidia A100-40GB or H100 recommended
- Platform: vLLM on Modal/cloud infrastructure
- Time: ~2-3 hours for 10 models

**Matrix Building & Analysis**
- Time: <5 minutes (CPU sufficient)
- Memory: <8GB RAM

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
├── hle_2pl_item_analysis.ipynb
└── hle_cfa_dimensionality.ipynb
```

## Reproducing Paper Results

To regenerate all results from the paper:

1. **Data Collection & Matrix**
   ```bash
   python scripts/collect_all_models.py          # Collect responses (or use existing data/)
   PYTHONPATH=src python scripts/build_response_matrix.py build --min-coverage 0.95
   ```
   - Produces: `data/matrix/response_matrix.csv`, `build_report.json`
   - Used for: Model accuracies, coverage statistics

2. **Analysis**
   ```bash
   # IRT model estimation and item parameters
   jupyter notebook analysis/hle_2pl_item_analysis.ipynb
   
   # Dimensionality analysis (omega_h, PCA, domain correlations)
   jupyter notebook analysis/hle_cfa_dimensionality.ipynb
   
   # Random seeds are set in notebooks: np.random.seed(42), torch.manual_seed(42)
   ```
   - Produces all tables and statistics reported in the paper

   ```bash
   jupyter notebook analysis/hle_2pl_item_analysis.ipynb
   # Run all cells
   ```


## Testing

Run unit tests to verify scoring logic:
```bash
pytest tests/
```

Tests cover:
- Answer extraction from model responses
- Response scoring accuracy
- Matrix construction with zero-variance filtering

## Notes

- We used `temperature=0` for all models that support it
- Responses were parsed with regex (not LLM-judged), hit 91.8% success rate
- Missing values (<1% after filtering) were filled with 0 for analysis
- Open models ran on vLLM with Modal + A100 GPUs
- 85 items had zero variance (all models got them wrong) and were dropped

## Code Attribution

- **HLE Dataset**: [Center for AI Safety](https://huggingface.co/datasets/cais/hle) (gated dataset, license agreement required)
- **IRT Implementation**: Uses `torch_measure` package (MIT License) for 2PL model estimation
- **Response Parsing**: Custom implementation with regex patterns adapted from HLE evaluation guidelines
- **API Integrations**: Original implementation for OpenAI, Anthropic, Google, and DeepSeek providers
- **vLLM Inference**: Uses vLLM framework for open-weight model inference
