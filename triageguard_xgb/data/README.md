# Data Directory

- `raw/`: Place raw MIMIC-IV and MIMIC-IV-ED data files here.
- `processed/`: Intermediate processed data, splits, and states are saved here.

## Data Mapping

- `chiefcomplaint` in MIMIC-IV-ED is mapped to `triage_complaint`.
- `anchor_age` in MIMIC-IV is used for patient age.
- `gender` in MIMIC-IV is used for patient sex.
- `disposition` in MIMIC-IV-ED is mapped to hospital admission target.
