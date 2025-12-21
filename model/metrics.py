import os
import json
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

def log_metrics(version, metrics_dict):
    """Log model metrics for a given version and timestamp"""
    snapshot = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        **metrics_dict
    }

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics_registry = json.load(f)
    else:
        metrics_registry = {}

    metrics_registry.setdefault(version, []).append(snapshot)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_registry, f, indent=4)
    
    print(f"Metrics logged for model version {version} at {snapshot['timestamp']}")

def load_metrics(version=None):
    """Load metrics for a specific version, or all versions"""
    if not os.path.exists(METRICS_PATH):
        return {}
    
    with open(METRICS_PATH, "r") as f:
        metrics_registry = json.load(f)
    
    if version:
        return metrics_registry.get(version, {})
    return metrics_registry