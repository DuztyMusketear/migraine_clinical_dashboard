import json
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REGISTRY_PATH = os.path.join(BASE_DIR, "model", "registry.json")


def get_active_version(default="legacy"):
    """Return the active model version form registry.json"""
    if not os.path.exists(REGISTRY_PATH):
        return default

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    return registry.get("active", default)

def load_active_model():
    """Load the active model"""
    version = get_active_version()

    if version == "legacy":
        raise FileNotFoundError("No versioned model yet")

    model_path = os.path.join(
        BASE_DIR, "model", version, "logistic_model.joblib"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)