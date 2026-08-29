import pandas as pd
import os

def load_mimic_data(mimic_iv_dir: str, mimic_iv_ed_dir: str):
    """
    Loads raw MIMIC-IV and MIMIC-IV-ED CSV files into pandas DataFrames.
    
    Args:
        mimic_iv_dir: Path to the MIMIC-IV core data directory (e.g. dataset/mimic-iv-clinical-database-demo-2.2)
        mimic_iv_ed_dir: Path to the MIMIC-IV-ED data directory (e.g. dataset/mimic-iv-ed-demo-2.2)
        
    Returns:
        A dictionary of pandas DataFrames.
    """
    
    # MIMIC-IV core files
    patients_path = os.path.join(mimic_iv_dir, "hosp", "patients.csv.gz")
    admissions_path = os.path.join(mimic_iv_dir, "hosp", "admissions.csv.gz")
    diagnoses_path = os.path.join(mimic_iv_dir, "hosp", "diagnoses_icd.csv.gz")
    icustays_path = os.path.join(mimic_iv_dir, "icu", "icustays.csv.gz")
    
    # MIMIC-IV-ED files
    edstays_path = os.path.join(mimic_iv_ed_dir, "ed", "edstays.csv.gz")
    triage_path = os.path.join(mimic_iv_ed_dir, "ed", "triage.csv.gz")
    vitalsign_path = os.path.join(mimic_iv_ed_dir, "ed", "vitalsign.csv.gz")
    
    print("Loading MIMIC-IV core data...")
    patients = pd.read_csv(patients_path)
    admissions = pd.read_csv(admissions_path)
    diagnoses_icd = pd.read_csv(diagnoses_path)
    icustays = pd.read_csv(icustays_path)
    
    print("Loading MIMIC-IV-ED data...")
    edstays = pd.read_csv(edstays_path)
    triage = pd.read_csv(triage_path)
    vitalsign = pd.read_csv(vitalsign_path)
    
    # Basic mapping and formatting
    
    # Rename columns to match what the pipeline expects
    edstays = edstays.rename(columns={'intime': 'edregtime', 'outtime': 'edouttime'})
    
    # Ensure datetime columns are properly parsed
    edstays['edregtime'] = pd.to_datetime(edstays['edregtime'])
    edstays['edouttime'] = pd.to_datetime(edstays['edouttime'])
    
    vitalsign['charttime'] = pd.to_datetime(vitalsign['charttime'])
    
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    icustays['outtime'] = pd.to_datetime(icustays['outtime'])
    
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    
    # Rename chiefcomplaint to triage_complaint
    if 'chiefcomplaint' in triage.columns:
        triage = triage.rename(columns={'chiefcomplaint': 'triage_complaint'})
        
    return {
        'patients': patients,
        'admissions': admissions,
        'diagnoses_icd': diagnoses_icd,
        'icustays': icustays,
        'edstays': edstays,
        'triage': triage,
        'vitalsign': vitalsign
    }
