import pandas as pd
from etl.transform.fhir_mapper import fhir_to_model

# List of all columns the model expects
EXPECTED_COLS = [
        "Age", "Duration", "Frequency", "Location", "Intensity", "Nausea", "Vomit",
        "Phonophobia", "Photophobia", "Visual", "Sensory", "Dysphasia", "Dysarthria",
        "Vertigo", "Tinnitus", "Hypoacusis", "Diplopia", "Defect", "Conscience",
        "Paresthesia", "DPF", "Type_Familial hemiplegic migraine", "Type_Migraine without aura",
        "Type_Other", "Type_Sporadic hemiplegic migraine", "Type_Typical aura with migraine",
        "Type_Typical aura without migraine"
    ]


def transform(df):
    """Transform raw EHR data into model-ready format"""
    print("Transforming data...")

    # Add missing columns with default 0
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column order
    df = df[EXPECTED_COLS + ["patient_id"]] if "patient_id" in df.columns else df[EXPECTED_COLS]
    return df