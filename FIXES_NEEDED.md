# Fixes Needed for Course Requirements

## Critical (Must Fix)

### 1. Pin Exact Package Versions
**File**: `requirements.txt`
**Change**: Replace `>=` with `==` for all packages
```txt
datasets==2.19.0
numpy==1.26.0
pandas==2.2.0
pyarrow==15.0.0
PyYAML==6.0.0
openai==1.30.0
anthropic==0.25.0
google-generativeai==0.8.0
tqdm==4.66.0
```

### 2. Add Repository Structure to README
**File**: `README.md`
**Add after "What's in here"**:
```markdown
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
│   ├── responses/raw/           # Per-model API responses (37 files)
│   └── matrix/                  # Final matrices & metadata
├── analysis/
│   └── hle_cfa_dimensionality.ipynb  # IRT & CFA analysis
├── figures/                     # Generated plots
├── requirements.txt             # Python dependencies
└── README.md
```
```

### 3. Add Computational Requirements Section
**File**: `README.md`
**Add after "Running it"**:
```markdown
## Computational Requirements

**Data Collection**
- Time: ~20 hours total (limited by API rate limits)
- Cost: $105-330 for 29 models (varies by provider)
- No GPU needed (uses API endpoints)

**Open-Weight Models** (if running locally)
- GPU: Nvidia A100-40GB or H100 recommended
- Framework: vLLM on Modal/cloud infrastructure
- Time: ~2-3 hours for 10 models

**Matrix Building & Analysis**
- Time: <5 minutes (CPU sufficient)
- Memory: <8GB RAM
```

### 4. Add Script-to-Output Mapping
**File**: `README.md`
**Add new section before "Notes"**:
```markdown
## Reproducing Paper Results

| Paper Section | Script/Notebook | Output Files |
|---------------|-----------------|--------------|
| Table 1: Model accuracies | `scripts/collect_all_models.py` + `build_response_matrix.py build` | `data/matrix/build_report.json` |
| Figure 1: Item difficulty & discrimination | `analysis/hle_cfa_dimensionality.ipynb` (Cells 10-15) | `figures/item_parameters.png` |
| Figure 2: Domain-level comparison | `analysis/hle_cfa_dimensionality.ipynb` (Cells 20-25) | `figures/domain_comparison.png` |
| Figure 3: Model ability estimates | `analysis/hle_cfa_dimensionality.ipynb` (Cells 30-35) | `figures/model_abilities.png` |
| Figure 4: Domain-specific abilities | `analysis/hle_cfa_dimensionality.ipynb` (Cells 40-45) | `figures/domain_abilities.png` |
| Figure 5: Test Information Function | `analysis/hle_cfa_dimensionality.ipynb` (Cells 50-55) | `figures/tif_decomposed.png` |

**To regenerate all results:**
```bash
# 1. Collect responses (or use existing data/)
python scripts/collect_all_models.py

# 2. Build matrix
PYTHONPATH=src python scripts/build_response_matrix.py build --min-coverage 0.95

# 3. Run analysis notebook
jupyter notebook analysis/hle_cfa_dimensionality.ipynb
# Execute all cells to regenerate figures/
```
```

### 5. Verify Random Seeds in Analysis
**Action**: Check `analysis/hle_cfa_dimensionality.ipynb` for:
```python
import numpy as np
import random
np.random.seed(42)
random.seed(42)
# If using torch:
import torch
torch.manual_seed(42)
```
If missing, add to first cell of notebook.

## Recommended (Nice to Have)

### 6. Add .env.example
**Create**: `.env.example`
```bash
# Copy this to .env and fill in your API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
```

### 7. Add Testing Instructions
**File**: `README.md`
**Add section**:
```markdown
## Testing

Run unit tests:
```bash
pytest tests/
```

Expected: All tests pass (scoring logic verification)
```

### 8. Code Attribution
**File**: `README.md` or new `ATTRIBUTION.md`
```markdown
## Code Attribution

- HLE dataset: [Center for AI Safety](https://huggingface.co/datasets/cais/hle)
- IRT implementation: torch_measure package (MIT License)
- Response parsing: Custom implementation with regex patterns adapted from HLE evaluation scripts
- All model API integrations: Original implementation
```

## Summary

**Must fix immediately**: Items 1-5 (exact versions, structure, requirements, mapping, random seeds)
**Should add**: Items 6-8 (examples, tests, attribution)

**Estimated time**: 1-2 hours to implement all critical fixes
