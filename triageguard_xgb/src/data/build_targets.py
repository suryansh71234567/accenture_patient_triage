import pandas as pd

def build_targets(states_df: pd.DataFrame, data: dict) -> pd.DataFrame:
    """
    Builds the prediction targets for each decision state.
    
    Targets:
    - icu_risk_2h: ICU admission within 2 hours after decision_time.
    - icu_risk_6h: ICU admission within 6 hours after decision_time.
    - icu_risk_12h: ICU admission within 12 hours after decision_time.
    - admission_risk: Hospital admission for this ED encounter.
    """
    print("Building targets...")
    icustays = data['icustays']
    
    # We will build targets on a copy
    df = states_df.copy()
    
    # 1. Admission Target
    # In MIMIC-IV, if hadm_id is not null for an ED stay, it means the patient was admitted to the hospital.
    # We will use hadm_id presence as the binary indicator for hospital admission.
    df['admission_risk'] = df['hadm_id'].notna().astype(int)
    
    # 2. ICU Targets
    # We need to find if there is an ICU stay for this subject that starts after decision_time
    # and within the specified horizons.
    
    # To optimize, we can merge all icustays for the subject, then filter
    icu_relevant = icustays[['subject_id', 'intime', 'outtime']].copy()
    
    # Merge states with all icustays for the same subject
    merged = df[['stay_id', 'subject_id', 'decision_time']].merge(icu_relevant, on='subject_id', how='left')
    
    # Check if patient is already in ICU at decision time
    # Rule: If already in ICU, future escalation is 0.
    already_in_icu = (merged['intime'] <= merged['decision_time']) & (merged['outtime'] > merged['decision_time'])
    
    # Check for future ICU admissions
    # We only care about ICU admissions that start AFTER the decision time
    future_icu = merged[merged['intime'] > merged['decision_time']].copy()
    future_icu['time_to_icu'] = (future_icu['intime'] - future_icu['decision_time']).dt.total_seconds() / 3600.0 # in hours
    
    # Find minimum time to next ICU admission for each stay_id and decision_time
    # Since we have unique states identified by stay_id and decision_time, we group by these
    min_time_to_icu = future_icu.groupby(['stay_id', 'decision_time'])['time_to_icu'].min().reset_index()
    
    # Merge back to df
    df = df.merge(min_time_to_icu, on=['stay_id', 'decision_time'], how='left')
    
    # Also find states where patient is already in ICU
    already_in_icu_states = merged[already_in_icu][['stay_id', 'decision_time']].drop_duplicates()
    already_in_icu_states['is_already_in_icu'] = True
    df = df.merge(already_in_icu_states, on=['stay_id', 'decision_time'], how='left')
    df['is_already_in_icu'] = df['is_already_in_icu'].fillna(False)
    
    # Construct binary targets
    df['icu_risk_2h'] = ((df['time_to_icu'] <= 2) & (~df['is_already_in_icu'])).astype(int)
    df['icu_risk_6h'] = ((df['time_to_icu'] <= 6) & (~df['is_already_in_icu'])).astype(int)
    df['icu_risk_12h'] = ((df['time_to_icu'] <= 12) & (~df['is_already_in_icu'])).astype(int)
    
    # Clean up temporary columns
    df = df.drop(columns=['time_to_icu', 'is_already_in_icu'])
    
    return df
