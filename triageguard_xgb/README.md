# TriageGuard — XGBoost Risk Engine

This project contains the XGBoost training branch of TriageGuard. It provides a clean, simple, and reproducible training pipeline to construct interpretable risk predictions from structured patient information and triage symptoms available at the time of ED triage.

## Architecture

```text
Patient at decision time
          │
          ├── Structured hospital data
          │        │
          │        └── compact features
          │
          └── Triage symptoms
                   │
                   └── pretrained embedding
                            │
                            └── PCA
                                  │
                                  ▼
                         Fixed feature vector
                                  │
                                  ▼
                             XGBoost
                                  │
             ┌────────────────────┼────────────────────┐
             ▼                    ▼                    ▼
        ICU 2h risk          ICU 6h risk          ICU 12h risk
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                                  ▼
                           Admission risk
                                  │
                                  ▼
                         Confidence scores
                                  │
                                  ▼
                         Future routing model
```

**Note:** This XGBoost branch is not the final routing system. It provides interpretable quantitative risk signals that a downstream routing model will consume.

## Setup

1. **Virtual Environment**:
   Create a Python virtual environment:
   ```bash
   python -m venv .venv
   ```
   Activate it:
   - Windows: `.venv\Scripts\activate`
   - Linux/macOS: `source .venv/bin/activate`

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

> The demonstration dataset is for pipeline validation and proof of concept. The full MIMIC-IV + MIMIC-IV-ED dataset is required for credible model evaluation.

## Pipeline Usage

The pipeline is entirely reproducible and can be executed with the following steps:

1. **Prepare Data**:
   Processes the raw data, applies missing-information masking, and builds feature and target vectors.
   ```bash
   python scripts/prepare_data.py
   ```

2. **Train Models**:
   Trains the XGBoost models and fits probability calibrators.
   ```bash
   python scripts/train_models.py
   ```

3. **Run Inference**:
   Demonstrates loading saved artifacts and making predictions on dummy data.
   ```bash
   python scripts/run_inference.py
   ```
