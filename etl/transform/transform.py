import pandas as pd
from etl.transform.fhir_mapper import fhir_to_model

# List of all columns the model expects
EXPECTED_COLS = [
        "Age", "Duration", "Frequency", "Location", "Intensity", "Nausea", "Vomit",
        "Phonophobia", "Photophobia", "Sensory", "Dysphasia", "Dysarthria",
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
    columns = EXPECTED_COLS.copy()
    if "Visual" in df.columns:
        columns.append("Visual")
    if "patient_id" in df.columns:
        columns.append("patient_id")

    df = df[columns]
    
    # Ensure Visual is numeric
    if "Visual" in df.columns:
        df["Visual"] = pd.to_numeric(df["Visual"], errors="coerce")
    return df