# Training and Evaluation

- `train.py`: Trains individual XGBoost models for each target output.
- `calibration.py`: Fits Platt scaling calibrators on the validation dataset to output certainty scores along with predictions.
- `evaluation.py`: Computes ROC-AUC, PR-AUC, Brier score, log loss, and evaluates across different missingness patterns.
