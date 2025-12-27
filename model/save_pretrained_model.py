import os
import json
import shutil
from .save_best_model import train_and_save_model  # relative import

def main(version="v1"):
    BASE_DIR = os.path.dirname(__file__)
    OLD_MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
    VERSIONED_FOLDER = os.path.join(BASE_DIR, version)
    NEW_MODEL_PATH = os.path.join(VERSIONED_FOLDER, "logistic_model.joblib")
    REGISTRY_PATH = os.path.join(BASE_DIR, "registry.json")
    LEGACY_FOLDER = os.path.join(BASE_DIR, "legacy")

    # Train model if .pkl doesn't exist
    if not os.path.exists(OLD_MODEL_PATH):
        print("No trained model found. Training now...")
        train_and_save_model()

    # Create version folders
    os.makedirs(VERSIONED_FOLDER, exist_ok=True)
    os.makedirs(LEGACY_FOLDER, exist_ok=True)  # ensure legacy exists

    # Copy model to versioned folder
    shutil.copy2(OLD_MODEL_PATH, NEW_MODEL_PATH)

    # Copy model to legacy if missing
    LEGACY_MODEL_PATH = os.path.join(LEGACY_FOLDER, "logistic_model.joblib")
    if not os.path.exists(LEGACY_MODEL_PATH):
        shutil.copy2(OLD_MODEL_PATH, LEGACY_MODEL_PATH)
        print("Legacy model created.")

    # Update registry
    if not os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "w") as f:
            json.dump({"active": version}, f, indent=2)
        print(f"Registry created → active={version}")
    else:
        with open(REGISTRY_PATH, "r+") as f:
            data = json.load(f)
            data["active"] = version
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
        print(f"Registry updated → active={version}")


# CLI support
if __name__ == "__main__":
    import sys
    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    main(ver)