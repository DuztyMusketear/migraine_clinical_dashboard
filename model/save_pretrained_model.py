import os
import joblib
import json
import shutil

# --- Paths (relative to this script in model/) ---
old_model_path = "logistic_model.pk1"  # pre-trained model in the same folder as this script

if not os.path.exists(old_model_path):
    raise FileNotFoundError(f"{old_model_path} not found. Make sure you have trained the model first.")

# Versioned folder
version = "v1"
versioned_folder = os.path.join("v1")  # creates model/v1 relative to current folder
os.makedirs(versioned_folder, exist_ok=True)

# New model path (expected by load_active_model)
new_model_path = os.path.join(versioned_folder, "logistic_model.joblib")

# Copy the model to the versioned folder and rename
shutil.copy2(old_model_path, new_model_path)

# Create registry.json pointing to the active version
registry_path = "registry.json"  # stays in model/ folder
with open(registry_path, "w") as f:
    json.dump({"active": version}, f, indent=2)

print(f"Model copied to {new_model_path}")
print(f"Registry created at {registry_path} pointing to version '{version}'")