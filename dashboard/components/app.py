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
FEATURE_COLS = model.feature_names_in_
X_patient = patient_row[FEATURE_COLS]

#Make prediction
prediction = model.predict(X_patient)[0]
prediction_proba = model.predict_proba(X_patient)[0][1]


st.metric("Predicted Visual Aura", int(prediction))
st.progress(float(prediction_proba))

st.subheader("Patient Data")
st.dataframe(patient_row)