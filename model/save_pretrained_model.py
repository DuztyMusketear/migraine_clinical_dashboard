import os
import joblib
import json
import shutil
import sys

# --- CONFIG ---
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1"

# Paths (relative to model/)
BASE_DIR = os.path.dirname(__file__)  # model/ folder
OLD_MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
VERSIONED_FOLDER = os.path.join(BASE_DIR, VERSION)
NEW_MODEL_PATH = os.path.join(VERSIONED_FOLDER, "logistic_model.joblib")
REGISTRY_PATH = os.path.join(BASE_DIR, "registry.json")

# --- SAFETY CHECK ---
if not os.path.exists(OLD_MODEL_PATH):
    raise FileNotFoundError(
        f"{OLD_MODEL_PATH} not found. Train the model first."
    )

# --- CREATE VERSION FOLDER ---
os.makedirs(VERSIONED_FOLDER, exist_ok=True)

# --- COPY MODEL ---
shutil.copy2(OLD_MODEL_PATH, NEW_MODEL_PATH)

# --- UPDATE REGISTRY ---
with open(REGISTRY_PATH, "w") as f:
    json.dump({"active": VERSION}, f, indent=2)

print(f"Model registered as {VERSION}")
print(f"Saved to {NEW_MODEL_PATH}")
print(f"Registry updated → active={VERSION}")