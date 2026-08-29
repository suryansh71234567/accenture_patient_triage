import os
import sys
import json

# Add project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.predict import TriageGuardPredictor

def main():
    print("Testing TriageGuard Inference Pipeline...")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(project_root, 'models')
    
    try:
        predictor = TriageGuardPredictor(models_dir)
    except Exception as e:
        print(f"Failed to load predictor. Ensure models are trained first. Error: {e}")
        return
        
    # Dummy patient data mimicking what might be available at triage
    patient_data = {
        "age": 65.0,
        "sex": "M",
        "previous_ed_visits": 2,
        "previous_hospital_admissions": 1,
        "previous_icu_admissions": 0,
        
        "cardiovascular_history": 1,
        "diabetes_history": 1,
        
        "hr_arrival": 95,
        "hr_current": 105,
        
        "rr_arrival": 18,
        "rr_current": 22,
        
        "spo2_arrival": 96,
        "spo2_current": 92,
        
        "sbp_arrival": 140,
        "sbp_current": 110,
        
        "dbp_arrival": 85,
        "dbp_current": 70,
        
        "temp_arrival": 37.2,
        "temp_current": None, # Missing temperature
        
        "time_elapsed_minutes": 30,
        
        "triage_complaint": "Patient complains of worsening shortness of breath and chest pain."
    }
    
    print("\nPatient Input:")
    print(json.dumps(patient_data, indent=2))
    
    print("\nRunning prediction...")
    results = predictor.predict(patient_data)
    
    print("\nInference Output:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
