import json
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REGISTRY_PATH = os.path.join(BASE_DIR, "model", "registry.json")


def get_active_version(default="legacy"):
    if not os.path.exists(REGISTRY_PATH):
        return default
    try:
        with open(REGISTRY_PATH) as f:
            data = json.load(f)
        return data.get("active", default)
    except Exception:
        return default

def load_active_model():
    version = get_active_version()
    model_path = os.path.join(BASE_DIR, "model", version, "logistic_model.joblib")

    # fallback to legacy
    if not os.path.exists(model_path):
        print(f"Warning: Active model '{version}' not found. Falling back to legacy.")
        version = "legacy"
        model_path = os.path.join(BASE_DIR, "model", version, "logistic_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at '{model_path}'")

    return joblib.load(model_path)