import pandas as pd
from .structured_features import extract_structured_features
from .text_features import TextFeatureExtractor

# Allowed feature columns for model (exclude identifiers)
FEATURE_COLUMNS = [
    "age",
    "sex",
    "previous_ed_visits",
    "previous_hospital_admissions",
    "previous_icu_admissions",
    "hr_arrival",
    "hr_current",
    "hr_delta",
    "spo2_arrival",
    # Add other allowed columns as needed
]

FORBIDDEN_COLUMNS = {
    "patient_id",
    "subject_id",
    "encounter_id",
    "hadm_id",
    "stay_id",
}

def build_feature_matrix(df: pd.DataFrame, text_extractor: TextFeatureExtractor, is_training: bool = False) -> pd.DataFrame:
    """
    Combines structured features and text PCA features into the final feature matrix.
    Returns the feature DataFrame (X).
    """
    # Extract structured features and filter to allowed columns
    structured_X = extract_structured_features(df)
    structured_X = structured_X[FEATURE_COLUMNS]

    if "triage_complaint" in df.columns:
        text_X = text_extractor.extract_features(df["triage_complaint"], is_training=is_training)
        X = pd.concat([structured_X, text_X], axis=1)
    else:
        X = structured_X

    # Ensure no forbidden identifier columns are present
    assert not any(col in X.columns for col in FORBIDDEN_COLUMNS), f"Forbidden identifier columns found in feature matrix: {set(X.columns) & FORBIDDEN_COLUMNS}"

    # Ensure column names are strings
    X.columns = X.columns.astype(str)
    return X
