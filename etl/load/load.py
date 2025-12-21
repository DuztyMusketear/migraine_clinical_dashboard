import pandas as pd
import os

PROCESSED_DATA_PATH = os.path.join("data", "processed", "processed_ehr.csv")

def load(df):
    """Saved Processed Data"""
    print("Loading data...")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

