import pandas as pd
import streamlit as st
from model.metrics import load_metrics

def render_metrics_tab():
    st.subheader("Model Performance Over Time")

    metrics_registry = load_metrics()
    if not metrics_registry:
        st.info("No metrics logged yet.")
        return

    # Flatten JSON → DataFrame
    rows = []
    for version, snapshots in metrics_registry.items():
        for snap in snapshots:
            row = {"version": version, **snap}
            rows.append(row)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    st.dataframe(df, use_container_width=True)

    metric = st.selectbox(
        "Select metric to visualize",
        [c for c in df.columns if c not in ("timestamp", "version")]
    )

    st.line_chart(
        df.pivot(index="timestamp", columns="version", values=metric)
    )