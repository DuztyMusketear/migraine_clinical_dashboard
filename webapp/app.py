from flask import Flask, render_template, request, redirect, url_for, session
from markupsafe import Markup
import sys, os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import plotly.express as px
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from etl.run_pipeline import run_pipeline
from predict.model_loader import load_active_model, get_active_version
from predict.feature_summary import global_feature_summary, patient_feature_contribution
from model.metrics import load_metrics
from model.retrain import retrain_model

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# Load model & data
processed_df = run_pipeline()
model = load_active_model()
FEATURE_COLS = model.feature_names_in_

# Admin credentials from .env
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password123")

@app.route("/metrics")
def metrics():
    is_admin = session.get("is_admin", False)
    if not is_admin:
        return redirect(url_for("index"))


    

    # Flatten metrics into DataFrame-like structure for HTML table
    metrics_table = []
    max_psi = 0

    metrics_registry = load_metrics()  # load all versions
    if metrics_registry:
        for version, snapshots in metrics_registry.items():
            for snap in snapshots:
                row = {"version": version, **snap}  # include version in table
                metrics_table.append(row)
                # calculate max PSI
                psi_keys = [k for k in snap.keys() if k.startswith("PSI_")]
                if psi_keys:
                    max_psi = max(max_psi, max(snap[k] for k in psi_keys))

    return render_template(
        "metrics.html",
        metrics_table=metrics_table,
        max_psi=max_psi,
        metrics_available=bool(metrics_table),
        metric_options=[k for k in metrics_table[0].keys() if k not in ("timestamp", "version")] if metrics_table else [],
        metrics_chart=None
    )



@app.route("/", methods=["GET", "POST"])
def index():
    # --- Admin session ---
    is_admin = session.get("is_admin", False)

    # --- Patient selection ---
    patient_ids = processed_df["patient_id"].tolist()
    patient_id = request.form.get("patient_id", session.get("selected_patient"))
    patient_row = None
    prediction = None
    probability = None
    feature_contrib = None

    if patient_id is not None:
        patient_id = int(patient_id)
        session["selected_patient"] = patient_id
        patient_row = processed_df[processed_df["patient_id"] == patient_id]
        if not patient_row.empty:
            X_patient = patient_row[FEATURE_COLS]
            prediction = model.predict(X_patient)[0]
            probability = model.predict_proba(X_patient)[0][1]
            feature_contrib = patient_feature_contribution(model, X_patient)

    global_feature_imp = global_feature_summary(model)
    active_version = get_active_version()

    # --- Metrics / PSI ---
    metrics_table = []
    max_psi = 0
    metrics_available = False
    metric_options = []
    metrics_chart = None
    psi_values = {}  # define this here so it's always available

    if is_admin:
        metrics_registry = load_metrics()  # load all versions
        all_keys = set()  # collect all keys (Accuracy, AUC, PSI_...)

        if metrics_registry:
            # Flatten all snapshots
            for version, snapshots in metrics_registry.items():
                for snap in snapshots:
                    row = {"version": version, **snap}
                    metrics_table.append(row)
                    all_keys.update(row.keys())

            metrics_available = len(metrics_table) > 0

            # Build psi_values from latest snapshot per version
            all_psi_keys = sorted([k for k in all_keys if k.startswith("PSI_")])
            for version, snapshots in metrics_registry.items():
                last_snap = snapshots[-1]
                for k in all_psi_keys:
                    psi_values[k] = last_snap.get(k, 0.0)

            max_psi = max(psi_values.values()) if psi_values else 0

            # Make sure every row has all keys (fill missing with 0.0)
            for row in metrics_table:
                for key in all_keys:
                    if key not in row:
                        row[key] = 0.0

            #Build sorted metric dropdown
            metric_options = sorted([k for k in all_keys if k not in ("timestamp", "version")])

            # Default metric to plot
            metric_to_plot = request.args.get("metric", metric_options[0])

            # Convert to DataFrame for plotting
            df = pd.DataFrame(metrics_table)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.reindex(columns=metric_options + ["version", "timestamp"])

            fig = px.line(
                df,
                x="timestamp",
                y=metric_to_plot,
                color="version",
                title=f"{metric_to_plot} Over Time"
            )
            metrics_chart = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
    

    return render_template(
        "index.html",
        patient_ids=patient_ids,
        selected_patient=patient_id,
        patient_row=patient_row,
        has_patient=patient_row is not None and not patient_row.empty,
        prediction=prediction,
        probability=probability,
        feature_contrib=feature_contrib,
        global_feature_imp=global_feature_imp,
        active_version=active_version,
        is_admin=is_admin,
        psi_values=psi_values,
        metrics_table=metrics_table,
        max_psi=max_psi,
        metrics_available=metrics_available,
        admin_username=ADMIN_USERNAME,
        admin_password=ADMIN_PASSWORD,
        metric_options=metric_options,
        metrics_chart=Markup(metrics_chart) if metrics_chart else None,
    )

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session["is_admin"] = True
    return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session["is_admin"] = False
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)