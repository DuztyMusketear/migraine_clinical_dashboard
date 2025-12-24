import os
import pandas as pd
from etl.extract.extract import extract_dummy
from etl.transform.transform import transform
from etl.load import load as load_module

# -----------------------------
# Test extract()
# -----------------------------
def test_extract_dummy():
    df = extract_dummy()
    assert not df.empty
    assert "patient_id" in df.columns

# -----------------------------
# Test transform()
# -----------------------------
def test_transform_adds_missing_columns():
    df = extract_dummy()
    df_transformed = transform(df.drop(columns=["Age"]))  # Drop a column to test adding
    assert "Age" in df_transformed.columns

# -----------------------------
# Test load()
# -----------------------------
def test_load_creates_file(tmp_path, monkeypatch):
    df = extract_dummy()

    # Patch PROCESSED_DATA_PATH in the load module
    monkeypatch.setattr(load_module, "PROCESSED_DATA_PATH", tmp_path / "processed_ehr.csv")

    # Call load
    load_module.load(df)

    # Assert the file exists
    assert (tmp_path / "processed_ehr.csv").exists()