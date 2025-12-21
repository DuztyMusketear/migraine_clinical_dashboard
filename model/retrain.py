import os
import json
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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

    model_path = os.path.join(version_dir, "logistic_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved new model version: {model_path}")

    # Update registry.json
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
    else:
        registry = {}

    registry.setdefault("versions", [])
    registry["versions"].append(version)
    registry["active"] = version

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)

    print(f"Updated registry.json with active version: {version}")
    return model