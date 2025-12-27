import os
import json
import shutil
from .save_best_model import train_and_save_model  # import the function

def main(version="v1"):
    """Ensure model is trained, then register version"""
    
    BASE_DIR = os.path.dirname(__file__)
    OLD_MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
    VERSIONED_FOLDER = os.path.join(BASE_DIR, version)
    NEW_MODEL_PATH = os.path.join(VERSIONED_FOLDER, "logistic_model.joblib")
    REGISTRY_PATH = os.path.join(BASE_DIR, "registry.json")

    # Train model if .pkl doesn't exist
    if not os.path.exists(OLD_MODEL_PATH):
        print("No trained model found. Training now...")
        train_and_save_model()

    # Create version folder
    os.makedirs(VERSIONED_FOLDER, exist_ok=True)

    # Copy to versioned folder
    shutil.copy2(OLD_MODEL_PATH, NEW_MODEL_PATH)

    # Update registry
    with open(REGISTRY_PATH, "w") as f:
        json.dump({"active": version}, f, indent=2)

    print(f"Model registered as {version}")
    print(f"Saved to {NEW_MODEL_PATH}")
    print(f"Registry updated → active={version}")

# CLI support
if __name__ == "__main__":
    import sys
    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    main(ver)