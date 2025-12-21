import os
import json
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from model.drift import population_stability_index
from model.metrics import log_metrics

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")
REGISTRY_PATH = os.path.join(MODEL_DIR, "registry.json")

def retrain_model(df):
    """Retrain model on current data and save as a new version."""
    # Split features & target
    X = df.drop(columns=["Visual", "patient_id"], errors="ignore")
    y = df["Visual"]

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

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

    log_metrics(version, psi_scores)


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

    print(f"Model retrained and activated: {version}")
    return model