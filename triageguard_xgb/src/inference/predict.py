import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
from src.features.text_features import TextFeatureExtractor

class TriageGuardPredictor:
    def __init__(self, models_dir: str):
        """
        Loads all required artifacts for inference.
        """
        print(f"Loading inference artifacts from {models_dir}...")
        
        # Load metadata
        meta_path = os.path.join(models_dir, 'preprocessing', 'model_metadata.json')
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Load feature names
        fn_path = os.path.join(models_dir, 'preprocessing', 'feature_names.json')
        with open(fn_path, 'r') as f:
            self.feature_names = json.load(f)
            
        # Load PCA
        pca_path = os.path.join(models_dir, 'preprocessing', 'text_pca.joblib')
        self.text_extractor = TextFeatureExtractor(
            model_name=self.metadata.get('text_embedding_model', 'all-MiniLM-L6-v2'),
            n_components=self.metadata.get('pca_dimensions', 8),
            pca_path=pca_path
        )
        
        # Load XGBoost models
        self.targets = ['icu_risk_2h', 'icu_risk_6h', 'icu_risk_12h', 'admission_risk']
        self.models = {}
        self.calibrators = {}
        
        for t in self.targets:
            # Model
            if t == 'admission_risk':
                m_path = os.path.join(models_dir, f'xgb_admission.json')
                c_path = os.path.join(models_dir, 'calibration', f'admission_calibrator.joblib')
            else:
                m_path = os.path.join(models_dir, f'xgb_{t.replace("icu_risk_", "icu_")}.json')
                c_path = os.path.join(models_dir, 'calibration', f'{t.replace("icu_risk_", "icu_")}_calibrator.joblib')
                
            model = xgb.XGBClassifier()
            model.load_model(m_path)
            self.models[t] = model
            
            # Calibrator
            calibrator = joblib.load(c_path)
            self.calibrators[t] = calibrator
            
    def _construct_features(self, patient_data: dict) -> pd.DataFrame:
        """
        Takes raw dictionary input and builds the exact feature vector.
        """
        df = pd.DataFrame([patient_data])
        
        features = pd.DataFrame(index=df.index)
        
        # Demographics
        features['age'] = pd.to_numeric(df['age'] if 'age' in df.columns else np.nan, errors='coerce')
        if 'sex' in df.columns:
            # Map sex strings to numeric regardless of pandas dtype (object vs StringDtype)
            # 'M' -> 1, anything else (including 'F') -> 0
            features['sex'] = (df['sex'].astype(str) == 'M').astype(float)
        else:
            features['sex'] = np.nan
            
        # Previous history
        history_cols = [
            'previous_ed_visits', 'previous_hospital_admissions', 'previous_icu_admissions',
            'cardiovascular_history', 'respiratory_history', 'renal_history',
            'diabetes_history', 'neurological_history', 'malignancy_history'
        ]
        for col in history_cols:
            val = df[col] if col in df.columns else pd.Series([0])
            features[col] = pd.to_numeric(val, errors='coerce').fillna(0)
            
        # Vitals — cast everything to float64 to avoid object dtype crashing XGBoost
        vitals = ['hr', 'rr', 'spo2', 'sbp', 'dbp', 'temp']
        available_vitals = 0
        total_vitals = len(vitals)
        
        for v in vitals:
            arr_raw = df[f'{v}_arrival'].iloc[0] if f'{v}_arrival' in df.columns else np.nan
            cur_raw = df[f'{v}_current'].iloc[0] if f'{v}_current' in df.columns else np.nan
            
            arr = float(arr_raw) if arr_raw is not None and not (isinstance(arr_raw, float) and np.isnan(arr_raw)) else np.nan
            cur = float(cur_raw) if cur_raw is not None and not (isinstance(cur_raw, float) and np.isnan(cur_raw)) else np.nan
            
            features[f'{v}_arrival'] = arr
            features[f'{v}_current'] = cur
            features[f'{v}_delta'] = (cur - arr) if (not np.isnan(cur) and not np.isnan(arr)) else np.nan
            
            is_missing = 1 if np.isnan(cur) else 0
            features[f'{v}_missing'] = is_missing
            available_vitals += (1 - is_missing)
            
        features['information_completeness'] = available_vitals / total_vitals
        features['time_elapsed_minutes'] = float(df['time_elapsed_minutes'].iloc[0]) if 'time_elapsed_minutes' in df.columns else 0.0
        
        # Text
        complaint_series = df['triage_complaint'] if 'triage_complaint' in df.columns else pd.Series([""])
        text_X = self.text_extractor.extract_features(complaint_series, is_training=False)
        
        X = pd.concat([features, text_X], axis=1)
        
        # Ensure exact column ordering; fill any missing columns with NaN
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = np.nan
                
        X = X[self.feature_names]
        
        # Enforce float64 throughout — XGBoost rejects object dtype columns
        X = X.astype(float)
        
        return X

    def predict(self, patient_data: dict) -> dict:
        """
        Returns a dictionary of risks, confidences, raw probabilities,
        and data completeness — everything the router layer needs.

        Output keys
        -----------
        icu_risk_2h, icu_risk_6h, icu_risk_12h, admission_risk   : calibrated [0,1]
        icu_risk_2h_confidence, ...                               : 2*|p-0.5| [0,1]
        icu_risk_2h_raw, ...                                      : uncalibrated XGB prob
        information_completeness                                  : fraction of vitals present [0,1]
        """
        X = self._construct_features(patient_data)

        results = {}
        for t in self.targets:
            model      = self.models[t]
            calibrator = self.calibrators[t]

            raw_prob       = float(model.predict_proba(X)[:, 1][0])
            calibrated_prob = float(calibrator.calibrate(
                                    model.predict_proba(X)[:, 1])[0])
            confidence     = float(2 * abs(calibrated_prob - 0.5))

            results[t]                  = calibrated_prob        # e.g. icu_risk_2h
            results[f"{t}_confidence"]  = confidence             # e.g. icu_risk_2h_confidence
            results[f"{t}_raw"]         = raw_prob               # e.g. icu_risk_2h_raw

        # Expose data completeness so the router knows how much to trust XGBoost
        results["information_completeness"] = float(
            X["information_completeness"].iloc[0]
        )

        return results
