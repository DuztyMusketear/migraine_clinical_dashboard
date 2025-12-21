
import joblib
import os
import sys
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from etl.run_pipeline import run_pipeline

# Run ETL pipeline
processed_df = run_pipeline()

# Load model
MODEL_PATH = os.path.join(BASE_DIR, "model", "logistic_model.pk1")
model = joblib.load(MODEL_PATH)

# Simulate selecting a patient
selected_patient = 1
patient_row = processed_df[processed_df['patient_id'] == selected_patient]

# Prepare features
X_patient = patient_row.drop(columns=['patient_id', 'Visual'])

# Make prediction
prediction = model.predict(X_patient)[0]

print(f"Predicted Visual Aura for patient {selected_patient}: {prediction}")
print(patient_row)
