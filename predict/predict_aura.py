import pandas as pd
import os
import sys
import joblib

BASE_DIR = BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from etl.run_pipeline import run_pipeline
from schemas.patient_features import PatientFeatures
from predict.model_loader import load_active_model

#File paths
PREDICTIONS_PATH = os.path.join(BASE_DIR,"data", "processed", "predictions.csv")

def validate_row(row, feature_cols):
    validated = PatientFeatures(**row.to_dict())
    return pd.DataFrame([validated.dict()])[feature_cols]

def predict_all():
    """Predict visual aura for all patients"""
    df = run_pipeline()

    #Load versioned model via loader
    model = load_active_model()
    FEATURE_COLS = list(model.feature_names_in_)

    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"ETL output missing required features: {missing}")
    
    predictions = []
    probabilities = []

    for _, row in df.iterrows():
        X = validate_row(row, FEATURE_COLS)
        predictions.append(model.predict(X)[0])
        probabilities.append(model.predict_proba(X)[0][1])

    df["predicted_aura"] = predictions
    df["predicted_aura_prob"] = probabilities

    df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")
    return df


def predict_patient(patient_id):
    """Predict visual aura for a single patient"""
    df = run_pipeline()
    model = load_active_model() #Use loader
    FEATURE_COLS = list(model.feature_names_in_)

    patient_row = df[df['patient_id'] == patient_id]
    if patient_row.empty:
        raise ValueError(f"No patient found with ID {patient_id}")

    X = validate_row(patient_row.iloc[0], FEATURE_COLS)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return prediction, probability, patient_row


if __name__ == "__main__":
    predict_all()