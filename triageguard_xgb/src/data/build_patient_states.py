import pandas as pd
import numpy as np

def build_decision_states(data: dict) -> pd.DataFrame:
    """
    Constructs decision states for each ED encounter at fixed elapsed times (0, 30, 60 mins).
    
    Args:
        data: Dictionary of DataFrames from load_data.py
        
    Returns:
        A DataFrame where each row is a decision state for a patient encounter.
    """
    edstays = data['edstays']
    patients = data['patients']
    triage = data['triage']
    vitalsign = data['vitalsign']
    
    print("Building decision states...")
    
    # We use stay_id (ED stay) as the encounter identifier.
    # We will generate states at 0, 30, and 60 minutes from arrival (edregtime).
    elapsed_minutes_list = [0, 30, 60]
    
    # 1. Base static information per encounter
    # Merge edstays with patients and triage
    base = edstays[['subject_id', 'hadm_id', 'stay_id', 'edregtime', 'edouttime']].copy()
    base = base.rename(columns={'edregtime': 'arrival_time'})
    
    # Attach age and gender
    base = base.merge(patients[['subject_id', 'anchor_age', 'gender']], on='subject_id', how='left')
    base = base.rename(columns={'anchor_age': 'age', 'gender': 'sex'})
    
    # Attach triage information
    # We will assume triage information is available at arrival (t=0)
    triage_info = triage[['stay_id', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'acuity', 'triage_complaint']]
    triage_info = triage_info.rename(columns={
        'temperature': 'temp_arrival',
        'heartrate': 'hr_arrival',
        'resprate': 'rr_arrival',
        'o2sat': 'spo2_arrival',
        'sbp': 'sbp_arrival',
        'dbp': 'dbp_arrival'
    })
    base = base.merge(triage_info, on='stay_id', how='left')
    
    # 2. Expand to multiple decision states
    states = []
    for elapsed in elapsed_minutes_list:
        state_df = base.copy()
        state_df['time_elapsed_minutes'] = elapsed
        state_df['decision_time'] = state_df['arrival_time'] + pd.to_timedelta(elapsed, unit='m')
        states.append(state_df)
    
    states_df = pd.concat(states, ignore_index=True)
    
    # 3. Attach current vitals based on decision_time
    # We need to find the latest vital sign available at or before decision_time for that stay_id
    # Sort vitals by charttime
    vitalsign_sorted = vitalsign.sort_values(by=['stay_id', 'charttime'])
    states_df = states_df.sort_values(by=['stay_id', 'decision_time'])
    
    # We can use pd.merge_asof to match decision_time with the closest past charttime
    # Drop rows where charttime is null for safety
    v_clean = vitalsign_sorted.dropna(subset=['charttime'])
    v_clean = v_clean.rename(columns={
        'temperature': 'temp_current',
        'heartrate': 'hr_current',
        'resprate': 'rr_current',
        'o2sat': 'spo2_current',
        'sbp': 'sbp_current',
        'dbp': 'dbp_current'
    })
    
    states_df = pd.merge_asof(
        states_df.sort_values('decision_time'),
        v_clean[['stay_id', 'charttime', 'temp_current', 'hr_current', 'rr_current', 'spo2_current', 'sbp_current', 'dbp_current']].sort_values('charttime'),
        by='stay_id',
        left_on='decision_time',
        right_on='charttime',
        direction='backward'
    )
    
    # If no vital sign was recorded before decision_time, merge_asof gives NaN for current vitals.
    # In that case, we can fallback to the arrival vitals from triage if they exist and are considered available at t=0.
    # However, sometimes arrival vitals might be NaN too.
    for v in ['temp', 'hr', 'rr', 'spo2', 'sbp', 'dbp']:
        # If current is missing, fallback to arrival
        states_df[f'{v}_current'] = states_df[f'{v}_current'].fillna(states_df[f'{v}_arrival'])
        # Also ensure arrival is correctly stored (merge might not have handled some things, but it's fine)
    
    # Calculate deltas (current - arrival)
    for v in ['temp', 'hr', 'rr', 'spo2', 'sbp', 'dbp']:
        states_df[f'{v}_delta'] = states_df[f'{v}_current'] - states_df[f'{v}_arrival']
        # if arrival was missing, delta is missing
    
    # 4. Attach compact history
    # For each stay, calculate history based on events BEFORE arrival_time
    history_df = build_patient_history(states_df[['subject_id', 'stay_id', 'arrival_time']].drop_duplicates(), data)
    states_df = states_df.merge(history_df, on='stay_id', how='left')
    
    # Restore sorting for readability
    states_df = states_df.sort_values(by=['subject_id', 'stay_id', 'time_elapsed_minutes']).reset_index(drop=True)
    
    return states_df

def build_patient_history(encounters: pd.DataFrame, data: dict) -> pd.DataFrame:
    """
    Computes historical features (previous visits, comorbidities) strictly before arrival.
    """
    print("Building patient history...")
    edstays = data['edstays']
    admissions = data['admissions']
    icustays = data['icustays']
    diagnoses_icd = data['diagnoses_icd']
    
    history_records = []
    
    # Group by subject to optimize
    # Note: iterating rows might be slow for massive datasets, but for the demo (and even full MIMIC if grouped), it's manageable.
    # To fully vectorize:
    
    # Previous ED visits
    ed_merged = encounters.merge(edstays[['subject_id', 'stay_id', 'edregtime']], on='subject_id', suffixes=('', '_prev'))
    prev_ed = ed_merged[ed_merged['edregtime'] < ed_merged['arrival_time']]
    prev_ed_counts = prev_ed.groupby('stay_id').size().reset_index(name='previous_ed_visits')
    
    # Previous admissions
    adm_merged = encounters.merge(admissions[['subject_id', 'hadm_id', 'admittime']], on='subject_id')
    prev_adm = adm_merged[adm_merged['admittime'] < adm_merged['arrival_time']]
    prev_adm_counts = prev_adm.groupby('stay_id').size().reset_index(name='previous_hospital_admissions')
    
    # Previous ICU stays
    icu_merged = encounters.merge(icustays[['subject_id', 'stay_id', 'intime']], on='subject_id', suffixes=('', '_icu'))
    prev_icu = icu_merged[icu_merged['intime'] < icu_merged['arrival_time']]
    prev_icu_counts = prev_icu.groupby('stay_id').size().reset_index(name='previous_icu_admissions')
    
    # Comorbidities based on past hospital admissions ICD codes
    # We only look at diagnoses from hadm_ids that occurred before the current ED stay
    past_diagnoses = prev_adm.merge(diagnoses_icd[['hadm_id', 'icd_code', 'icd_version']], on='hadm_id', how='inner')
    
    # Define simple ICD patterns for broad categories (simplified for demo)
    # Note: In a real system, you would use robust mappings (like Elixhauser). 
    # Here we use prefix matching on common ICD-9 and ICD-10 codes.
    categories = {
        'cardiovascular_history': ['410', '411', '412', '413', '414', '428', 'I20', 'I21', 'I22', 'I23', 'I24', 'I25', 'I50'],
        'respiratory_history': ['490', '491', '492', '493', '494', '495', '496', 'J40', 'J41', 'J42', 'J43', 'J44', 'J45'],
        'renal_history': ['585', 'N18'],
        'diabetes_history': ['250', 'E10', 'E11'],
        'neurological_history': ['430', '431', '432', '433', '434', '435', '436', '437', '438', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'G30'],
        'malignancy_history': ['14', '15', '16', '17', '18', '19', '20', 'C'] # broad prefix matching
    }
    
    # Map each ICD code in past_diagnoses to these categories
    for cat_name, prefixes in categories.items():
        # Check if the code starts with any of the prefixes
        # Vectorized check:
        pattern = '^(' + '|'.join(prefixes) + ')'
        past_diagnoses[cat_name] = past_diagnoses['icd_code'].str.match(pattern).fillna(False).astype(int)
    
    # Aggregate per stay_id (if any past diagnosis matches the category, it's a 1)
    if not past_diagnoses.empty:
        comorbidities = past_diagnoses.groupby('stay_id')[list(categories.keys())].max().reset_index()
    else:
        comorbidities = pd.DataFrame(columns=['stay_id'] + list(categories.keys()))
    
    # Combine all history
    history = encounters[['stay_id']].copy()
    history = history.merge(prev_ed_counts, on='stay_id', how='left').fillna({'previous_ed_visits': 0})
    history = history.merge(prev_adm_counts, on='stay_id', how='left').fillna({'previous_hospital_admissions': 0})
    history = history.merge(prev_icu_counts, on='stay_id', how='left').fillna({'previous_icu_admissions': 0})
    history = history.merge(comorbidities, on='stay_id', how='left').fillna({cat: 0 for cat in categories.keys()})
    
    return history
