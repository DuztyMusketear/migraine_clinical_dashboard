import os
import json
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from model.drift import population_stability_index
from model.metrics import log_metrics

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")
REGISTRY_PATH = os.path.join(MODEL_DIR, "registry.json")

def retrain_model(df):
    """Retrain model on current data and save as a new version."""
    
    TARGET_COL = "Visual"
    FEATURE_COLS = [c for c in df.columns if c != TARGET_COL and c != "patient_id"]

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Drop rows with missing target
    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if X.empty:
        print("No valid training data available! Skipping retrain.")
        return None

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)

    # Create new version folder
    version = datetime.now().strftime("%Y%m%d%H%M%S")
    version_dir = os.path.join(MODEL_DIR, version)
    os.makedirs(version_dir, exist_ok=True)

    # Attach metadata
    model._version = version
    model._trained_at = datetime.utcnow().isoformat()
    model.feature_names_in_ = X.columns.to_list()

    #Save Model
    model_path = os.path.join(version_dir, "logistic_model.joblib")
    joblib.dump(model, model_path)

    # Drift (PSI) Calculation
    baseline = X.iloc[: len(X)//2]
    current = X.iloc[len(X)//2 :]

    psi_scores = {
        f"PSI_{col}": population_stability_index(
            baseline[col],
            current[col]
        )
        for col in X.columns
    }

    metrics = {
        "Accuracy": round(accuracy, 4),
        "AUC": round(auc, 4),
        **psi_scores
    }

    log_metrics(version, metrics)


    # Update registry.json
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
    else:
        registry = {}

    registry.setdefault("versions", [])

    if version not in registry["versions"]:
        registry["versions"].append(version)

    registry["active"] = version
    registry["last_updated"] = datetime.utcnow().isoformat()

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)

    print(f"Registry path: {REGISTRY_PATH}")
    print("Registry contents after retrain:", json.dumps(registry, indent=4))
    print(f"Model retrained and activated: {version}")
    return model