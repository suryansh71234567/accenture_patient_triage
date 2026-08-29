import pandas as pd

def extract_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts structured features and constructs missingness indicators.
    
    Returns a dataframe containing only the final structured feature columns.
    """
    print("Extracting structured features...")
    
    features = pd.DataFrame(index=df.index)
    
    # Demographics
    features['age'] = df['age']
    # Map sex strings to numeric (M=1, F=0)
    features['sex'] = df['sex'].map({'M': 1, 'F': 0}).fillna(0).astype(int)
    
    # Previous history
    history_cols = [
        'previous_ed_visits',
        'previous_hospital_admissions',
        'previous_icu_admissions',
        'cardiovascular_history',
        'respiratory_history',
        'renal_history',
        'diabetes_history',
        'neurological_history',
        'malignancy_history'
    ]
    for col in history_cols:
        if col in df.columns:
            features[col] = df[col]
            
    # Core vitals (arrival, current, delta)
    vitals = ['hr', 'rr', 'spo2', 'sbp', 'dbp', 'temp']
    
    # Missingness indicators and completeness
    available_vitals = 0
    total_vitals = len(vitals)
    
    for v in vitals:
        arrival_col = f'{v}_arrival'
        current_col = f'{v}_current'
        delta_col = f'{v}_delta'
        
        # Keep numeric values as they are (NaNs will be handled by XGBoost)
        features[arrival_col] = df[arrival_col]
        features[current_col] = df[current_col]
        features[delta_col] = df[delta_col]
        
        # Missingness indicator (1 = missing, 0 = available)
        # We base missingness on current_col since it's the value available at decision time
        missing_indicator = df[current_col].isna().astype(int)
        features[f'{v}_missing'] = missing_indicator
        
        # Add to available vitals count (0 if missing, 1 if available)
        available_vitals += (1 - missing_indicator)
        
    features['information_completeness'] = available_vitals / total_vitals
    
    # Time elapsed
    features['time_elapsed_minutes'] = df['time_elapsed_minutes']
    
    return features
