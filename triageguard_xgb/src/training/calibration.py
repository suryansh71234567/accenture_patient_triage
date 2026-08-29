import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os
import xgboost as xgb

class ProbabilityCalibrator:
    """
    Independent Platt scaling (sigmoid) calibrator.
    Fits a Logistic Regression on the XGBoost output probabilities using the validation set.
    """
    def __init__(self):
        self.calibrator = LogisticRegression(solver='lbfgs')
        
    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray):
        """
        Fits the calibrator.
        raw_probs: 1D array of predicted probabilities from XGBoost for the positive class.
        y_true: 1D array of true binary labels.
        """
        # Convert probs to 2D array for sklearn
        X_cal = raw_probs.reshape(-1, 1)
        self.calibrator.fit(X_cal, y_true)
        
    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Calibrates raw probabilities.
        """
        X_cal = raw_probs.reshape(-1, 1)
        return self.calibrator.predict_proba(X_cal)[:, 1]

def calibrate_model(model: xgb.XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> ProbabilityCalibrator:
    """
    Computes probabilities on validation set and fits the calibrator.
    """
    print(f"Fitting probability calibrator on {len(X_val)} validation samples...")
    raw_probs = model.predict_proba(X_val)[:, 1]
    calibrator = ProbabilityCalibrator()
    calibrator.fit(raw_probs, y_val.values)
    return calibrator

def calculate_confidence(calibrated_probs: np.ndarray) -> np.ndarray:
    """
    Calculates confidence as 2 * abs(p - 0.5).
    """
    return 2 * np.abs(calibrated_probs - 0.5)

def save_calibrator(calibrator: ProbabilityCalibrator, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(calibrator, filepath)
    print(f"Calibrator saved to {filepath}")
