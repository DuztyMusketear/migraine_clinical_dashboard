import os
import pandas as pd
import sqlalchemy

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "synthetic_ehr.csv")
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "patient_data.db")
DB_URI = f"sqlite:///{DB_PATH}"

def extract_from_db():
    """Extract from Database"""
    if not os.path.exists("data/processed/patient_data.db"):
        raise FileNotFoundError("Database not found at data/processed/patient_data.db")
    engine = sqlalchemy.create_engine(DB_URI)
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
            "Visual": [1, 0, 1, 0]
    })
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


def extract():
    """Unified Extraction Function"""
    df = extract_from_csv()
    if df is not None:
        return df
    
    try:
        return extract_from_db()
    except FileNotFoundError:
        return extract_dummy()

