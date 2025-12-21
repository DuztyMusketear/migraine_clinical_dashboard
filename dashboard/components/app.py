import streamlit as st
import pandas as pd
import joblib
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from dotenv import load_dotenv
from etl.run_pipeline import run_pipeline
from model.retrain import retrain_model
from model.metrics import load_metrics
from predict.model_loader import load_active_model, get_active_version
from predict.feature_summary import global_feature_summary, patient_feature_contribution
from dashboard.components.metrics_tab import render_metrics_tab

load_dotenv()
admin_username = os.getenv("ADMIN_USERNAME")
admin_password = os.getenv("ADMIN_PASSWORD")
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

#File Paths
MODEL_PATH = os.path.join(BASE_DIR, "model", "logistic_model.pk1")

#Run ETL pipeline
@st.cache_data
def load_processed_df():
    return run_pipeline()

processed_df = load_processed_df()

#Load Model
@st.cache_data
def load_model():
    try: 
        return load_active_model()
    except FileNotFoundError:
        return joblib.load(MODEL_PATH)
model = load_model()

FEATURE_COLS = model.feature_names_in_


# ----------------------------
# UI Header
# --------------------------
st.title("Migraine Prediction Dashboard")
st.caption(f"Active Model Version: `{get_active_version()}`")



# ----------------------------
# Admin Controls
# ----------------------------
if st.session_state.is_admin:
    st.sidebar.header("Admin Controls")
    if st.sidebar.checkbox("Enable Live Retraining"):
        if st.sidebar.button("Retrain Model"):
            st.sidebar.info("Retraining Model...")
            new_model = retrain_model(processed_df)
            st.sidebar.success("Model retrained (v2)")
else:
    st.sidebar.info("Login as admin to access controls")
# ----------------------------
# Patient Selection (global)
# ----------------------------
patient_options = processed_df['patient_id'].tolist()
selected_patient = st.selectbox("Select a patient", patient_options)

patient_row = processed_df[processed_df['patient_id'] == selected_patient]
X_patient = patient_row[FEATURE_COLS]

prediction = model.predict(X_patient)[0]
prediction_proba = model.predict_proba(X_patient)[0][1]

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab_login = st.tabs([
    "Patient Prediction",
    "Feature Analysis",
    "Model Metrics",
    "Admin"
])

# --- Tab 1: Patient Prediction ---
with tab1:
    st.metric("Predicted Visual Aura", int(prediction))
    st.progress(float(prediction_proba))
    st.subheader("Patient Data")
    st.dataframe(patient_row)

# --- Tab 2: Patient Feature Analysis ---
with tab2:
    st.subheader(f"Feature Contributions for Patient {selected_patient}")
    st.dataframe(
        patient_feature_contribution(model, X_patient),
        use_container_width=True
    )

    st.subheader("Global Feature Importance (Odds Ratios)")
    st.dataframe(
        global_feature_summary(model),
        use_container_width=True
    )

# --- Tab 3: Model Metrics ---
with tab3:
    if st.session_state.is_admin:
        version = get_active_version()
        metrics = load_metrics(version)

      
        if metrics:
            latest_metrics = metrics[-1]  # get most recent snapshot
            psi_values = {k: v for k, v in latest_metrics.items() if k.startswith("PSI_")}

            if psi_values:
                max_psi = max(psi_values.values())
                if max_psi > 0.2:
                    st.warning(f"⚠️ Data drift detected (max PSI = {max_psi:.2f}).")
                else:
                    st.success(f"✅ No significant drift detected (max PSI = {max_psi:.2f}).")

                with st.expander("View feature-level drift (PSI)"):
                    st.dataframe(
                        pd.DataFrame.from_dict(psi_values, orient="index", columns=["PSI"])
                        .sort_values("PSI", ascending=False),
                        use_container_width=True
                    )

        render_metrics_tab()
    else:
        st.info("Admin access required to view model metrics.")

# --- Tab Login: Admin Login ---
with tab_login:
    st.subheader("Admin")

   

    if st.session_state.is_admin:
        st.success("You are logged in as Admin")
        if st.button("Logout"):
            st.session_state.is_admin = False
            st.rerun()
    else:
        # Display demo credentials
        st.markdown(
        f"**Demo Credentials:**  \n"
        f"- Username: `{admin_username}`  \n"
        f"- Password: `{admin_password}`"
    )
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == admin_username and password == admin_password:
                st.session_state.is_admin = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")
