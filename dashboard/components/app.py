import streamlit as st
import pandas as pd
import joblib
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from etl.run_pipeline import run_pipeline
from model.retrain import retrain_model
from predict.model_loader import load_active_model

#File Paths
MODEL_PATH = os.path.join(BASE_DIR, "model", "logistic_model.pk1")
PROCESSED_DATA_PATH =  os.path.join(BASE_DIR, "data", "processed", "processed_ehr.csv")

#Run ETL pipeline
processed_df = run_pipeline()

#Load Model
try: 
    model = load_active_model()
except FileNotFoundError:
    model = joblib.load(MODEL_PATH)

#Streamlit UI
st.title("Migraine Prediction Dashboard")


#Admin Authentication
admin_username = "admin"
admin_password = "password123"

st.sidebar.header("Admin Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
is_admin = False

if username == admin_username and password == admin_password:
    is_admin = True
    st.sidebar.success("Admin Authenticated")
elif username or password:
    st.sidebar.error("Invalid Credentials")

#Sidebar Admin Controls
if is_admin:
    st.sidebar.header("Admin Controls")
    if st.sidebar.checkbox("Enable Live Retraining"):
        if st.sidebar.button("Retrain Model"):
            st.sidebar.info("Retraining Model...")
            new_model = retrain_model(processed_df)
            st.sidebar.success("Model retrained (v2)")


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