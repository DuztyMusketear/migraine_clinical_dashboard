import pandas as pd
import sqlalchemy
from data.db import DB_PATH
from pydantic import BaseModel, Field

COLUMN_MAP = {
    "Age": "age",
    "Duration": "duration",
    "Frequency": "frequency",
    "Location": "location",
    "Intensity": "intensity",

    "Nausea": "nausea",
    "Vomit": "vomit",
    "Phonophobia": "phonophobia",
    "Photophobia": "photophobia",

    "Visual": "visual",
    "Sensory": "sensory",
    "Dysphasia": "dysphasia",
    "Dysarthria": "dysarthria",

    "Vertigo": "vertigo",
    "Tinnitus": "tinnitus",
    "Hypoacusis": "hypoacusis",
    "Diplopia": "diplopia",
    "Defect": "defect",

    "Conscience": "conscience",
    "Paresthesia": "paresthesia",
    "DPF": "dpf",

    "Type_Familial hemiplegic migraine": "type_familial_hemiplegic",
    "Type_Migraine without aura": "type_migraine_no_aura",
    "Type_Other": "type_other",
    "Type_Sporadic hemiplegic migraine": "type_sporadic_hemiplegic",
    "Type_Typical aura with migraine": "type_typical_aura_with",
    "Type_Typical aura without migraine": "type_typical_aura_without",
}

class PatientFeatures(BaseModel):
    Age: int = Field(ge=0, le=120)
    Duration: int = Field(ge=0, default=0)
    Frequency: int = Field(ge=0, default=0)
    Location: int = Field(ge=0, default=0)
    Intensity: int = Field(ge=0, default=0)
    Nausea: int = Field(ge=0, le=1, default=0)
    Vomit: int = Field(ge=0, le=1, default=0)
    Phonophobia: int = Field(ge=0, le=1, default=0)
    Photophobia: int = Field(ge=0, le=1, default=0)
    Visual: int = Field(ge=0, le=1, default=0)
    Sensory: int = Field(ge=0, le=1, default=0)
    Dysphasia: int = Field(ge=0, le=1, default=0)
    Dysarthria: int = Field(ge=0, le=1, default=0)
    Vertigo: int = Field(ge=0, le=1, default=0)
    Tinnitus: int = Field(ge=0, le=1, default=0)
    Hypoacusis: int = Field(ge=0, le=1, default=0)
    Diplopia: int = Field(ge=0, le=1, default=0)
    Defect: int = Field(ge=0, le=1, default=0)
    Conscience: int = Field(ge=0, le=1, default=0)
    Paresthesia: int = Field(ge=0, le=1, default=0)
    DPF: int = Field(ge=0, le=1, default=0)
    Type_Familial_hemiplegic_migraine: int = Field(ge=0, le=1, default=0)
    Type_Migraine_without_aura: int = Field(ge=0, le=1, default=0)
    Type_Other: int = Field(ge=0, le=1, default=0)
    Type_Sporadic_hemiplegic_migraine: int = Field(ge=0, le=1, default=0)
    Type_Typical_aura_with_migraine: int = Field(ge=0, le=1, default=0)
    Type_Typical_aura_without_migraine: int = Field(ge=0, le=1, default=0)

    class ConfigDict:
        extra = "forbid"

REQUIRED_COLUMNS = {"patient_id"} | set(COLUMN_MAP.keys())


def validate_schema(df: pd.DataFrame):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def ingest_ehr_dataframe(df: pd.DataFrame):
    # Validate
    validate_schema(df)

    # Normalize column names
    df = df.rename(columns=COLUMN_MAP)

    # Enforce numeric types
    for col in COLUMN_MAP.values():
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    # Write to SQL
    engine = sqlalchemy.create_engine(DB_PATH)
    df.to_sql(
        "patient_records",
        engine,
        if_exists="append",
        index=False
    )