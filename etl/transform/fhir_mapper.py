import pandas as pd

FHIR_TO_MODEL_MAP = {
    "patient_age" : "Age",
    "symptom_visual_disturbance": "Visual",
    "symptom_nausea": "Nausea",
    "symptom_photophobia": "Photophobia",
    "symptom_phonophobia": "Phonophobia",
}


def fhir_to_model(df: pd.DataFrame) -> pd.DataFrame:
    ''' Map FHIR-style fields to model feature schema'''

    mapped = {}

    for fhir_field, model_field in FHIR_TO_MODEL_MAP.items():
        if fhir_field in df.columns:
            mapped[model_field] = df[fhir_field]
        else:
            mapped[model_field] = 0
            
    return pd.DataFrame(mapped)