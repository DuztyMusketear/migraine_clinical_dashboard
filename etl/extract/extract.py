import os
import pandas as pd
from data.db import get_engine, DB_PATH

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "synthetic_ehr.csv")


def extract_from_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    engine = get_engine()
    df = pd.read_sql("SELECT * FROM patient_records", engine)
    print("Extracted data from Database")
    return df

def extract_from_csv():
    """Extract from raw CSV"""
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        print(f"Extracted data from CSV: {CSV_PATH}")
        return df
    return None

def extract_dummy():
    """Generate dummy EHR data"""
    print("Generating dummy data...")
    df = pd.DataFrame({
        "patient_id" : [1, 2, 3, 4],
        "Age": [25, 47, 33, 18],
        "Duration": [3, 6, 4, 2],
        "Frequency": [2, 5, 3, 1],
        "Location": [1, 2, 1, 3],
        "Intensity": [2, 3, 2, 1],
        "Nausea": [1, 0, 1, 0],
        "Vomit": [0, 0, 1, 0],
        "Phonophobia": [1, 1, 0, 0],
        "Photophobia": [1, 1, 1, 0],
        "Visual": [1, 0, 1, 0],
        "Sensory": [0, 1, 0, 0],
        "Dysphasia": [0, 0, 1, 0],
        "Dysarthria": [0, 0, 0, 0],
        "Vertigo": [0, 1, 0, 0],
        "Tinnitus": [0, 0, 0, 0],
        "Hypoacusis": [0, 0, 0, 0],
        "Diplopia": [0, 0, 0, 0],
        "Defect": [0, 0, 0, 0],
        "Conscience": [1, 1, 1, 1],
        "Paresthesia": [0, 1, 0, 0],
        "DPF": [0, 0, 0, 0],
        "Type_Familial hemiplegic migraine": [0, 0, 0, 0],
        "Type_Migraine without aura": [1, 0, 1, 1],
        "Type_Other": [0, 0, 0, 0],
        "Type_Sporadic hemiplegic migraine": [0, 0, 0, 0],
        "Type_Typical aura with migraine": [0, 0, 0, 0],
        "Type_Typical aura without migraine": [0, 1, 0, 0]
    })
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


def extract():
    """Unified Extraction Function
    1. SQL (production-line)
    2. CSV (batch simulation)
    3. Dummy (testing)
    """
    try: 
        return extract_from_db()
    except FileNotFoundError:
        df = extract_from_csv()
        if df is not None:
            return df
        return extract_dummy

