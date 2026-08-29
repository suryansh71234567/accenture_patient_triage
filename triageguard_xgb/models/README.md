# Models Directory

Saved model artifacts for inference.

- `xgb_icu_2h.json`, `xgb_icu_6h.json`, `xgb_icu_12h.json`: XGBoost models for ICU escalation risk targets.
- `xgb_admission.json`: XGBoost model for hospital admission risk target.

- `calibration/`: Saved joblib objects for Platt scaling probability calibration.
- `preprocessing/`: Saved PCA object, feature names list, and config metadata.
- `feature_importance/`: CSVs and plots showing feature gain scores. Note: Feature importance shows model reliance, not clinical causation.
