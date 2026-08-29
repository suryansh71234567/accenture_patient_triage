import os
import sys
import pandas as pd
import json

# Add project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.text_features import TextFeatureExtractor
from src.features.feature_pipeline import build_feature_matrix
from src.training.train import train_xgboost_model, save_xgboost_model, save_feature_names
from src.training.calibration import calibrate_model, save_calibrator
from src.training.evaluation import evaluate_model, plot_feature_importance

def main():
    print("Starting model training pipeline...")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')
    
    train_path = os.path.join(processed_dir, 'train_states.csv')
    val_path = os.path.join(processed_dir, 'val_states.csv')
    test_path = os.path.join(processed_dir, 'test_states.csv')
    
    # 1. Load splits
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Define targets
    targets = ['icu_risk_2h', 'icu_risk_6h', 'icu_risk_12h', 'admission_risk']
    
    # 2. Build feature matrices
    print("Building features...")
    pca_path = os.path.join(models_dir, 'preprocessing', 'text_pca.joblib')
    text_extractor = TextFeatureExtractor(n_components=8, pca_path=pca_path)
    
    X_train = build_feature_matrix(train_df, text_extractor, is_training=True)
    X_val = build_feature_matrix(val_df, text_extractor, is_training=False)
    X_test = build_feature_matrix(test_df, text_extractor, is_training=False)
    
    # Save feature names
    feature_names = X_train.columns.tolist()
    save_feature_names(feature_names, os.path.join(models_dir, 'preprocessing', 'feature_names.json'))
    
    # 3. Train, Calibrate, Evaluate per target
    for target in targets:
        print(f"\n{'='*40}")
        print(f"Processing Target: {target}")
        print(f"{'='*40}")
        
        y_train = train_df[target]
        y_val = val_df[target]
        y_test = test_df[target]
        
        # Train
        model = train_xgboost_model(X_train, y_train)
        
        # Calibrate on validation
        calibrator = calibrate_model(model, X_val, y_val)
        
        # Evaluate
        evaluate_model(model, calibrator, X_train, y_train, "Train", target, train_df)
        evaluate_model(model, calibrator, X_val, y_val, "Validation", target, val_df)
        evaluate_model(model, calibrator, X_test, y_test, "Test", target, test_df)
        
        # Feature Importance
        imp_dir = os.path.join(models_dir, 'feature_importance')
        plot_feature_importance(model, feature_names, target, imp_dir)
        
        # Save Model and Calibrator
        if target == 'admission_risk':
            model_filename = 'xgb_admission.json'
            calib_filename = 'admission_calibrator.joblib'
        else:
            clean_name = target.replace('icu_risk_', 'icu_')
            model_filename = f'xgb_{clean_name}.json'
            calib_filename = f'{clean_name}_calibrator.joblib'
            
        save_xgboost_model(model, os.path.join(models_dir, model_filename))
        save_calibrator(calibrator, os.path.join(models_dir, 'calibration', calib_filename))
        
    # Save Model Metadata
    metadata = {
        'training_date': pd.Timestamp.now().isoformat(),
        'random_seed': 42,
        'number_of_features': len(feature_names),
        'targets': targets,
        'text_embedding_model': 'all-MiniLM-L6-v2',
        'pca_dimensions': 8,
        'train_instances': len(train_df),
        'val_instances': len(val_df),
        'test_instances': len(test_df)
    }
    meta_path = os.path.join(models_dir, 'preprocessing', 'model_metadata.json')
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {meta_path}")
    
    print("\nModel training pipeline complete.")

if __name__ == "__main__":
    main()
