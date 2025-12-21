import pandas as pd

FHIR_TO_MODEL_MAP = {
    "patient_age": "Age",
    "symptom_duration": "Duration",
    "symptom_frequency": "Frequency",
    "symptom_location": "Location",
    "symptom_intensity": "Intensity",
    "symptom_nausea": "Nausea",
    "symptom_vomit": "Vomit",
    "symptom_phonophobia": "Phonophobia",
    "symptom_photophobia": "Photophobia",
    "symptom_visual_disturbance": "Visual",
    "symptom_sensory": "Sensory",
    "symptom_dysphasia": "Dysphasia",
    "symptom_dysarthria": "Dysarthria",
    "symptom_vertigo": "Vertigo",
    "symptom_tinnitus": "Tinnitus",
    "symptom_hypoacusis": "Hypoacusis",
    "symptom_diplopia": "Diplopia",
    "symptom_defect": "Defect",
    "symptom_conscience": "Conscience",
    "symptom_paresthesia": "Paresthesia",
    "symptom_dpf": "DPF",
    "type_familial_hemiplegic_migraine": "Type_Familial hemiplegic migraine",
    "type_migraine_without_aura": "Type_Migraine without aura",
    "type_other": "Type_Other",
    "type_sporadic_hemiplegic_migraine": "Type_Sporadic hemiplegic migraine",
    "type_typical_aura_with_migraine": "Type_Typical aura with migraine",
    "type_typical_aura_without_migraine": "Type_Typical aura without migraine"
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