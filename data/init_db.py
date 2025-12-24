from sqlalchemy import text
from data.db import get_engine

def init_db():
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
             CREATE TABLE IF NOT EXISTS patient_records(
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          patient_id INTEGER NOT NULL,

                          Age INTEGER,
                          Duration INTEGER,
                          Frequency INTEGER,
                          Location INTEGER,
                          Intensity INTEGER,
                          Nausea INTEGER,
                          Vomit INTEGER,
                          Phonophobia INTEGER,
                          Photophobia INTEGER,
                          Visual INTEGER,
                          Sensory INTEGER,
                          Dysphasia INTEGER,
                          Dysarthria INTEGER,
                          Vertigo INTEGER,
                          Tinnitus INTEGER,
                          Hypoacusis INTEGER,
                          Diplopia INTEGER,
                          Defect INTEGER,
                          Conscience INTEGER,
                          Paresthesia INTEGER,
                          DPF INTEGER,
                          
                          Type_Familial_hemiplegic_migraine INTEGER,
                          Type_Migraine_no_aura INTEGER,
                          Type_Other INTEGER,
                          Type_Sporadic_hemiplegic_migraine INTEGER,
                          Type_Typical_aura_with_migraine INTEGER,
                          Type_Typical_aura_without_migraine INTEGER,

                          ingestion_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                          )
"""))

if __name__ == "__main__":
    init_db()