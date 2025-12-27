import os
import json
import shutil
import sys

def register_model(version="v1"):
    """Register a pre-trained model under a version and update registry.json"""
    
    # Paths (relative to this script in model/)
    BASE_DIR = os.path.dirname(__file__)  # model/ folder
    OLD_MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
    VERSIONED_FOLDER = os.path.join(BASE_DIR, version)
    NEW_MODEL_PATH = os.path.join(VERSIONED_FOLDER, "logistic_model.joblib")
    REGISTRY_PATH = os.path.join(BASE_DIR, "registry.json")

    # Safety check
    if not os.path.exists(OLD_MODEL_PATH):
        raise FileNotFoundError(
            f"{OLD_MODEL_PATH} not found. Train the model first."
        )

    # Create version folder
    os.makedirs(VERSIONED_FOLDER, exist_ok=True)

    # Copy model to versioned folder
    shutil.copy2(OLD_MODEL_PATH, NEW_MODEL_PATH)

    # Update registry
    with open(REGISTRY_PATH, "w") as f:
        json.dump({"active": version}, f, indent=2)

    print(f"Model registered as {version}")
    print(f"Saved to {NEW_MODEL_PATH}")
    print(f"Registry updated → active={version}")


# Optional: allow running from command line
if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    register_model(ver)