from pydantic import BaseModel, Field

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

    class Config:
        extra = "forbid"