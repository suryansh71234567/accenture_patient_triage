import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, confusion_matrix
import xgboost as xgb
from .calibration import ProbabilityCalibrator

def evaluate_model(model: xgb.XGBClassifier, calibrator: ProbabilityCalibrator, X: pd.DataFrame, y: pd.Series, split_name: str, target_name: str, df_original: pd.DataFrame = None):
    """
    Evaluates the model and prints metrics.
    If df_original is provided (contains augmentation_pattern), it also evaluates per missingness pattern.
    """
    print(f"\n--- Evaluation for {target_name} ({split_name}) ---")
    
    # Ensure no NaN targets
    valid_idx = y.notna()
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]
    
    if len(y_valid) == 0:
        print("No valid labels for evaluation.")
        return
        
    pos_count = y_valid.sum()
    neg_count = len(y_valid) - pos_count
    print(f"Total: {len(y_valid)}, Positive: {pos_count}, Negative: {neg_count}")
    
    if pos_count == 0 or neg_count == 0:
        print("Only one class present. Cannot compute AUC.")
        return
        
    raw_probs = model.predict_proba(X_valid)[:, 1]
    cal_probs = calibrator.calibrate(raw_probs)
    
    roc_auc = roc_auc_score(y_valid, cal_probs)
    pr_auc = average_precision_score(y_valid, cal_probs)
    ll = log_loss(y_valid, cal_probs)
    brier = brier_score_loss(y_valid, cal_probs)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"LogLoss: {ll:.4f}")
    print(f"Brier:   {brier:.4f}")
    
    # Confusion matrix at 0.5 threshold
    preds = (cal_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_valid, preds)
    print(f"Confusion Matrix (threshold 0.5):\n{cm}")
    
    # Subgroup evaluation based on augmentation_pattern
    if df_original is not None and 'augmentation_pattern' in df_original.columns:
        print(f"\nSubgroup Evaluation by Missingness:")
        patterns = df_original.loc[valid_idx, 'augmentation_pattern'].unique()
        for p in patterns:
            idx = (df_original.loc[valid_idx, 'augmentation_pattern'] == p)
            if idx.sum() == 0 or y_valid[idx].sum() == 0 or (len(y_valid[idx]) - y_valid[idx].sum()) == 0:
                continue
            
            p_roc = roc_auc_score(y_valid[idx], cal_probs[idx])
            p_brier = brier_score_loss(y_valid[idx], cal_probs[idx])
            print(f"  Pattern '{p}' (n={idx.sum()}): ROC-AUC={p_roc:.4f}, Brier={p_brier:.4f}")
            
def plot_feature_importance(model: xgb.XGBClassifier, feature_names: list, target_name: str, save_dir: str):
    """
    Plots and saves feature importance based on gain.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get feature importances (gain is default in xgboost for classifier)
    importances = model.feature_importances_
    
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': importances
    })
    df_imp = df_imp.sort_values(by='importance_gain', ascending=False)
    
    csv_path = os.path.join(save_dir, f"{target_name}_importance.csv")
    df_imp.to_csv(csv_path, index=False)
    
    # Plot top 20
    top_n = min(20, len(df_imp))
    plt.figure(figsize=(10, 8))
    plt.barh(df_imp['feature'].head(top_n)[::-1], df_imp['importance_gain'].head(top_n)[::-1])
    plt.xlabel('Importance (Gain)')
    plt.title(f'Top {top_n} Feature Importances ({target_name})')
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f"{target_name}_importance.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Feature importance saved to {save_dir}")
