import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.load_data import load_mimic_data
from src.data.build_patient_states import build_decision_states
from src.data.build_targets import build_targets
from src.data.augmentation import augment_training_data

def main():
    print("Starting data preparation pipeline...")
    
    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_dir = os.path.abspath(os.path.join(project_root, '..', 'dataset'))
    mimic_iv_dir = os.path.join(dataset_dir, 'mimic-iv-clinical-database-demo-2.2')
    mimic_iv_ed_dir = os.path.join(dataset_dir, 'mimic-iv-ed-demo-2.2')
    
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)   #dcmkncJBJK
    
    # 1. Load Data
    data = load_mimic_data(mimic_iv_dir, mimic_iv_ed_dir)
    
    # 2. Build States
    states_df = build_decision_states(data)
    print(f"Generated {len(states_df)} decision states.")
    
    # 3. Build Targets
    states_with_targets = build_targets(states_df, data)
    
    # 4. Patient-level Train/Val/Test Split (70/15/15)
    print("Splitting data by patient...")
    subjects = states_with_targets['subject_id'].unique()
    
    # Train = 70%, Temp = 30%
    train_subj, temp_subj = train_test_split(subjects, test_size=0.3, random_state=42)
    # Val = 15%, Test = 15%
    val_subj, test_subj = train_test_split(temp_subj, test_size=0.5, random_state=42)
    
    # Check for leakage
    assert len(set(train_subj) & set(val_subj)) == 0
    assert len(set(train_subj) & set(test_subj)) == 0
    assert len(set(val_subj) & set(test_subj)) == 0
    
    train_df = states_with_targets[states_with_targets['subject_id'].isin(train_subj)].copy()
    val_df = states_with_targets[states_with_targets['subject_id'].isin(val_subj)].copy()
    test_df = states_with_targets[states_with_targets['subject_id'].isin(test_subj)].copy()
    
    print(f"Train patients: {len(train_subj)}, rows: {len(train_df)}")
    print(f"Val patients: {len(val_subj)}, rows: {len(val_df)}")
    print(f"Test patients: {len(test_subj)}, rows: {len(test_df)}")
    
    # 5. Missing-Information Augmentation (ONLY on Train)
    train_df_aug = augment_training_data(train_df, random_seed=42)
    
    # For val/test, we also add the 'augmentation_pattern' column as 'original' for consistency
    val_df['augmentation_pattern'] = 'original'
    test_df['augmentation_pattern'] = 'original'
    
    # 6. Save Processed Data
    train_path = os.path.join(processed_dir, 'train_states.csv')
    val_path = os.path.join(processed_dir, 'val_states.csv')
    test_path = os.path.join(processed_dir, 'test_states.csv')
    
    train_df_aug.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved processed data to {processed_dir}")
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
