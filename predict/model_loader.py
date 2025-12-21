import json
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def load_active_model():
    registry_path = os.path.join(BASE_DIR, "model", "registry.json")
    with open(registry_path) as f:
        registry = json.load(f)
    
    version = registry["active"]
    model_path = os.path.join(BASE_DIR, "model", version, "logistic_model.joblib")
    return joblib.load(model_path)