import streamlit as st
import pandas as pd
import joblib
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from etl.run_pipeline import run_pipeline

#File Paths
MODEL_PATH = os.path.join(BASE_DIR, "model", "logistic_model.pk1")
PROCESSED_DATA_PATH =  os.path.join(BASE_DIR, "data", "processed", "processed_ehr.csv")

#Run ETL pipeline
processed_df = run_pipeline()
if processed_df is None:
    processed_df = pd.read_csv(PROCESSED_DATA_PATH)

#Load Model
model = joblib.load(MODEL_PATH)

#Streamlit App
st.title("Migraine Prediction Dashboard")

#Patient Selection
patient_options = processed_df['patient_id'].tolist()
selected_patient = st.selectbox("Select a patient", patient_options)

#Get Patient Row
patient_row = processed_df[processed_df['patient_id'] == selected_patient]

#Features for prediction
X_patinet = patient_row.drop(columns=['patient_id', 'Visual'], errors='ignore')

#Make prediction
prediction = model.predict(X_patinet)[0]
st.write(f"Predicted Visual Aura: {prediction}")

st.subheader("Patient Data")
st.dataframe(patient_row)