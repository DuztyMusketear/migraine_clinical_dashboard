import pandas as pd
import os
import sys
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from etl.run_pipeline import run_pipeline
from schemas.patient_features import PatientFeatures
from predict.model_loader import load_active_model, get_active_version
from model.metrics import log_metrics, load_metrics
from model.drift import population_stability_index
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



#File paths
PREDICTIONS_PATH = os.path.join(BASE_DIR,"data", "processed", "predictions.csv")

def validate_row(row, feature_cols):
    validated = PatientFeatures(**row.to_dict())
    return pd.DataFrame([validated.dict()])[feature_cols]

def update_metrics(version, y_true, y_pred, y_prob):
    """Calculate and Log Metrics"""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_0": precision_score(y_true, y_pred, pos_label=0),
        "Precision_1": precision_score(y_true, y_pred, pos_label=1),
        "Recall_0": recall_score(y_true, y_pred, pos_label=0),
        "Recall_1": recall_score(y_true, y_pred, pos_label=1),
        "F1_0": f1_score(y_true, y_pred, pos_label=0),
        "F1_1": f1_score(y_true, y_pred, pos_label=1),
        "AUC": roc_auc_score(y_true, y_prob),
    }

    log_metrics(version, metrics)
    return metrics

def predict_all():
    """Predict visual aura for all patients"""
    df = run_pipeline()

    #Load versioned model via loader
    model = load_active_model()
    version = get_active_version()
    FEATURE_COLS = list(model.feature_names_in_)

    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"ETL output missing required features: {missing}")
    
    predictions, probabilities = [], []

    for _, row in df.iterrows():
        X = validate_row(row, FEATURE_COLS)
        predictions.append(model.predict(X)[0])
        probabilities.append(model.predict_proba(X)[0][1])

    df["predicted_aura"] = predictions
    df["predicted_aura_prob"] = probabilities
    df.to_csv(PREDICTIONS_PATH, index=False)
   

    metrics = update_metrics(
        version, df["Visual"], 
        df["predicted_aura"], 
        df["predicted_aura_prob"], 
    )

    # Drift Metrics (PSI)
    baseline = df[FEATURE_COLS].iloc[: len(df)//2]
    current = df[FEATURE_COLS].iloc[len(df)//2 :]

    psi_scores = {
        f"PSI_{col}": population_stability_index(
            baseline[col], current[col]
        )
        for col in FEATURE_COLS
    }
    
    metrics.update(psi_scores)
    log_metrics(version, metrics)
    
    return df


def predict_patient(patient_id):
    """Predict visual aura for a single patient"""
    df = run_pipeline()
    model = load_active_model() #Use loader
    version = get_active_version()

    FEATURE_COLS = list(model.feature_names_in_)
    patient_row = df[df['patient_id'] == patient_id]

    if patient_row.empty:
        raise ValueError(f"No patient found with ID {patient_id}")

    X = validate_row(patient_row.iloc[0], FEATURE_COLS)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    update_metrics(
        version, 
        [patient_row["Visual"].values[0]], 
        [prediction], 
        [probability], 
        patient_ids=[patient_id]
    )

    return prediction, probability, patient_row


if __name__ == "__main__":
    predict_all()