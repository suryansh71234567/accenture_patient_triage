import xgboost as xgb
import pandas as pd
import json
import os

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, random_seed: int = 42) -> xgb.XGBClassifier:
    """
    Trains a single binary XGBoost model.
    """
    print(f"Training XGBoost model on {len(X_train)} samples...")
    
    # We use scikit-learn API of XGBoost for ease of integration
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=random_seed,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    return model

def save_xgboost_model(model: xgb.XGBClassifier, filepath: str):
    """
    Saves the trained XGBoost model to a JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save_model(filepath)
    print(f"Model saved to {filepath}")

def save_feature_names(feature_names: list, filepath: str):
    """
    Saves the exact ordered list of features used for training.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(feature_names, f, indent=4)
    print(f"Feature names saved to {filepath}")
