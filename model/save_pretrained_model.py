import os
import json
import shutil
from .save_best_model import train_and_save_model  # relative import

def ensure_model(version="v1"):
    """
    Train model if needed, ensure legacy + versioned folder exist, and update registry.
    """

    BASE_DIR = os.path.dirname(__file__)
    MODEL_PKL = os.path.join(BASE_DIR, "logistic_model.pkl")
    VERSION_FOLDER = os.path.join(BASE_DIR, version)
    VERSION_MODEL = os.path.join(VERSION_FOLDER, "logistic_model.joblib")
    LEGACY_FOLDER = os.path.join(BASE_DIR, "legacy")
    LEGACY_MODEL = os.path.join(LEGACY_FOLDER, "logistic_model.joblib")
    REGISTRY_PATH = os.path.join(BASE_DIR, "registry.json")

    # Step 1: Train model if missing
    if not os.path.exists(MODEL_PKL):
        print("No trained model found. Training now...")
        train_and_save_model()

    # Step 2: Create folders if missing
    os.makedirs(VERSION_FOLDER, exist_ok=True)
    os.makedirs(LEGACY_FOLDER, exist_ok=True)

    # Step 3: Copy model to version folder
    shutil.copy2(MODEL_PKL, VERSION_MODEL)

    # Step 4: Ensure legacy model exists
    if not os.path.exists(LEGACY_MODEL):
        shutil.copy2(MODEL_PKL, LEGACY_MODEL)
        print("Legacy model created.")

    # Step 5: Update registry
    registry = {"active": version}
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r") as f:
                data = json.load(f)
            data["active"] = version
            registry = data
        except Exception:
            pass  # fallback to default registry
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Model '{version}' ensured. Registry updated → active={version}")


# CLI support
if __name__ == "__main__":
    import sys
    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    ensure_model(ver)