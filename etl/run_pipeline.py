import pandas as pd
import os


#File Paths
RAW_DATA_PATH = os.path.join("data", "raw", "synthetic_ehr.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "processed_ehr.csv")


def extract():
    """Read Raw Data"""
    print("Extracting data...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"No raw data found at {RAW_DATA_PATH}. Creating dummy data")
        #Create dummy EHR data
        df = pd.DataFrame({
            "patient_id" : [1, 2, 3],
            "Age": [25, 47, 33],
            "Visual": [1, 0, 1]
        })
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
    else:
        df = pd.read_csv(RAW_DATA_PATH)
    return df

def transform(df):
    """Perform Transformation"""
    print("Transforming data...")

    # List of all columns the model expects
    expected_cols = [
        "Age", "Duration", "Frequency", "Location", "Intensity", "Nausea", "Vomit",
        "Phonophobia", "Photophobia", "Visual", "Sensory", "Dysphasia", "Dysarthria",
        "Vertigo", "Tinnitus", "Hypoacusis", "Diplopia", "Defect", "Conscience",
        "Paresthesia", "DPF", "Type_Familial hemiplegic migraine", "Type_Migraine without aura",
        "Type_Other", "Type_Sporadic hemiplegic migraine", "Type_Typical aura with migraine",
        "Type_Typical aura without migraine"
    ]
    # Add missing columns with default 0
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column order
    df = df[expected_cols + ["patient_id"]]

    return df

def load(df):
    """Saved Processed Data"""
    print("Loading data...")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

def run_pipeline():
    df = extract()
    df = transform(df)
    load(df)
    return df

if __name__ == "__main__":
    run_pipeline()