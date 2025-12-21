import pandas as pd
import os
import joblib
BASE_DIR = BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#File paths
PROCESSED_DATA_PATH = os.path.join(BASE_DIR,"data", "processed", "processed_ehr.csv")
PREDICTIONS_PATH = os.path.join(BASE_DIR,"data", "processed", "predictions.csv")
MODEL_PATH = os.path.join(BASE_DIR,"model", "logistic_model.pk1")

def predict_all(df=None, model=None):
    """Predict visual aura for all patients"""
    if df is None:
        df = pd.read_csv(PROCESSED_DATA_PATH)

    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Train the model first.")
        model = joblib.load(MODEL_PATH)

    feature_cols = df.drop(columns=["patient_id", "Visual"], errors="ignore").columns
    df["predicted_aura"] = model.predict(df[feature_cols])

    df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")
    return df

def predict_patient(patient_id, df=None, model=None):
    """Predict visual aura for a single patient"""
    if df is None:
        df = pd.read_csv(PROCESSED_DATA_PATH)

    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Train the model first.")
        model = joblib.load(MODEL_PATH)

    patient_row = df[df['patient_id'] == patient_id]
    if patient_row.empty:
        raise ValueError(f"No patient found with ID {patient_id}")

    feature_cols = patient_row.drop(columns=["patient_id", "Visual"], errors="ignore").columns
    prediction = model.predict(patient_row[feature_cols])[0]
    return prediction, patient_row

if __name__ == "__main__":
    # Default behavior: predict all
    predict_all()