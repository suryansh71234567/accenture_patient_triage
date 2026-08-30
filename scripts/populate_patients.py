import json
from pathlib import Path

# Load patient definitions from batch_evaluate_50_patients
from scripts.batch_evaluate_50_patients import PATIENTS

target_ids = {
    'EVAL-001', 'EVAL-002', 'EVAL-003', 'EVAL-005', 'EVAL-006', 'EVAL-007',
    'EVAL-009', 'EVAL-011', 'EVAL-013', 'EVAL-015', 'EVAL-016', 'EVAL-018',
    'EVAL-020', 'EVAL-023', 'EVAL-026', 'EVAL-036', 'EVAL-038', 'EVAL-040'
}

patients_dir = Path('triageguard_agent/data/patients')
patients_dir.mkdir(parents=True, exist_ok=True)

created = []
for p in PATIENTS:
    if p['patient_id'] in target_ids:
        pid = p['patient_id']
        doc = {
            'patient_id': pid,
            'subject_id': pid,
            'age': p['age'],
            'sex': p['sex'],
            'chiefcomplaint': p['chiefcomplaint'],
            'acuity': p['acuity'],
            'heartrate': p['heartrate'],
            'resprate': p['resprate'],
            'o2sat': p['o2sat'],
            'sbp': p['sbp'],
            'dbp': p['dbp'],
            'temperature': p['temperature'],
            'pain': p['pain'],
            'time_elapsed_minutes': 15,
            'previous_ed_visits': p.get('previous_ed_visits', 0),
            'previous_hospital_admissions': p.get('previous_hospital_admissions', 0),
            'previous_icu_admissions': p.get('previous_icu_admissions', 0),
            'cardiovascular_history': p.get('cardiovascular_history', 0),
            'respiratory_history': p.get('respiratory_history', 0),
            'renal_history': p.get('renal_history', 0),
            'diabetes_history': p.get('diabetes_history', 0),
            'neurological_history': p.get('neurological_history', 0),
            'malignancy_history': p.get('malignancy_history', 0),
            'history_text': p.get('history_text', ''),
            'observations': [
                {
                    'encounter_id': f'ENC-{pid}',
                    'timestamp': '2026-08-30T10:00:00Z',
                    'type': 'triage',
                    'heart_rate': p['heartrate'],
                    'spo2': p['o2sat'],
                    'sbp': p['sbp'],
                    'dbp': p['dbp'],
                    'resp_rate': p['resprate'],
                    'pain_score': p['pain'],
                    'note': 'Initial triage assessment'
                }
            ]
        }
        fpath = patients_dir / f'{pid}.json'
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2)
        created.append(pid)

print(f'Successfully wrote {len(created)} patient records.')
