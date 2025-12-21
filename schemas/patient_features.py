from pydantic import BaseModel, Field

class PatientFeatures(BaseModel):
    Age: int = Field(ge=0, le=120)
    Visual: int = Field(ge=0, le=1)
    Nausea: int = Field(ge=0, le=1)
    Photobobia: int = Field(ge=0, le=1)
    Phononphobia: int = Field(ge=0, le=1)

    class Config:
        extra = "forbid"