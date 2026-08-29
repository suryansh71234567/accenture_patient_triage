# Data Processing

- `load_data.py`: Loads raw CSVs and performs basic type formatting.
- `build_patient_states.py`: Creates decision states per patient encounter (e.g. at 0, 30, 60 minutes) without forward-leakage of future events.
- `build_targets.py`: Generates truth labels using future events (e.g. ICU entry within 2 hours).
- `augmentation.py`: Applies missing-information masking to create synthetic rows with sparse data, simulating missing vitals. Applied *after* patient-level splitting.
