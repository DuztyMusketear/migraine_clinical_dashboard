import pandas as pd
import os
import joblib

BASE_DIR = BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from etl.run_pipeline import run_pipeline

#File paths
MODEL_PATH = os.path.join(BASE_DIR,"model", "logistic_model.pk1")
PREDICTIONS_PATH = os.path.join(BASE_DIR,"data", "processed", "predictions.csv")

def predict_all():
    """Predict visual aura for all patients"""
    df = run_pipeline()
    model = joblib(MODEL_PATH)

    FEATURE_COLS = model.feature_names_in_

    df["predicted_aura"] = model.predict(df[FEATURE_COLS])
    df["predicted_aura_prob"] = model.predict_proba(df[FEATURE_COLS])[:,1]

    df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")
    return df


def predict_patient(patient_id):
    """Predict visual aura for a single patient"""
    df = run_pipeline()
    model = joblib.load(MODEL_PATH)

    patient_row = df[df['patient_id'] == patient_id]
    if patient_row.empty:
        raise ValueError(f"No patient found with ID {patient_id}")

    FEATURE_COLS = model.feature_names_in_
    prediction = model.predict(patient_row[FEATURE_COLS])[0]
    probability = model.predict_proba(patient_row[FEATURE_COLS])[0][1]

    return prediction, probability, patient_row

if __name__ == "__main__":
    predict_all()