# Source Code

Contains the core logic for the TriageGuard XGBoost risk engine.

- `data/`: Data loading, state generation, target building, and missingness augmentation.
- `features/`: Extraction of structured features and text features (via pretrained embeddings).
- `training/`: Training, probability calibration, and evaluation logic.
- `inference/`: End-to-end inference pipeline loading saved artifacts.
