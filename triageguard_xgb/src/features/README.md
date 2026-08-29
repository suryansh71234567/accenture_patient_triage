# Feature Extraction

- `structured_features.py`: Extracts demographics, compact previous history (from ICD diagnoses), core vital signs (`_arrival`, `_current`, `_delta`), and missingness flags.
- `text_features.py`: Computes sentence embeddings of the triage complaint and fits a PCA model to reduce dimensionality.
- `feature_pipeline.py`: Assembles structured and text features into a single, compact feature vector.
