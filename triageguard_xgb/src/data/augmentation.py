import pandas as pd
import numpy as np

def augment_training_data(train_df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    Creates augmented versions of training decision states by simulating missing information.
    Must ONLY be applied to the training split to prevent leakage.
    
    Patterns:
    1 - Complete (original)
    2 - One vital missing
    3 - Two vitals missing
    4 - Sparse triage (approx half missing = 3 missing)
    5 - Very sparse (keep 2 = 4 missing)
    """
    print("Augmenting training data with missing-information patterns...")
    np.random.seed(random_seed)
    
    vitals = ['hr', 'rr', 'spo2', 'sbp', 'dbp', 'temp']
    
    augmented_dfs = []
    
    # Pattern 1: Original
    p1 = train_df.copy()
    p1['augmentation_pattern'] = 'original'
    augmented_dfs.append(p1)
    
    # Function to apply masking
    def mask_vitals(df, num_missing):
        masked_df = df.copy()
        masked_df['augmentation_pattern'] = f'{num_missing}_missing'
        
        # We need to randomly select vitals to mask per row
        # Since doing this row-by-row is slow, we can vectorize by creating random masks
        # Generate random indices to mask for each row
        rand_matrix = np.random.rand(len(masked_df), len(vitals))
        # For each row, the 'num_missing' smallest values in rand_matrix will be masked
        thresholds = np.sort(rand_matrix, axis=1)[:, num_missing - 1]
        
        # Create boolean mask: True if it should be masked
        mask = rand_matrix <= thresholds[:, None]
        
        # Apply mask to each vital
        for i, v in enumerate(vitals):
            is_masked_for_vital = mask[:, i]
            masked_df.loc[is_masked_for_vital, f'{v}_arrival'] = np.nan
            masked_df.loc[is_masked_for_vital, f'{v}_current'] = np.nan
            masked_df.loc[is_masked_for_vital, f'{v}_delta'] = np.nan
            
        return masked_df

    # Pattern 2: One vital missing
    p2 = mask_vitals(train_df, 1)
    augmented_dfs.append(p2)
    
    # Pattern 3: Two vitals missing
    p3 = mask_vitals(train_df, 2)
    augmented_dfs.append(p3)
    
    # Pattern 4: Sparse (3 missing)
    p4 = mask_vitals(train_df, 3)
    augmented_dfs.append(p4)
    
    # Pattern 5: Very sparse (4 missing)
    p5 = mask_vitals(train_df, 4)
    augmented_dfs.append(p5)
    
    final_df = pd.concat(augmented_dfs, ignore_index=True)
    
    # Shuffle the resulting dataframe
    final_df = final_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"Augmentation expanded training data from {len(train_df)} to {len(final_df)} rows.")
    return final_df
