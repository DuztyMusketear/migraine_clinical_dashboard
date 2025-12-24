import os
import json
import joblib
import tempfile
import pytest
from predict import model_loader

# -----------------------------
# Fixtures for temporary registry and model
# -----------------------------
@pytest.fixture
def temp_registry_and_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Paths
        registry_path = os.path.join(tmpdir, "registry.json")
        model_dir = os.path.join(tmpdir, "v1")
        os.makedirs(model_dir)
        model_file = os.path.join(model_dir, "logistic_model.joblib")

        # Dummy scikit-learn model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        joblib.dump(model, model_file)

        # Registry pointing to the dummy model
        registry = {"active": "v1"}
        with open(registry_path, "w") as f:
            json.dump(registry, f)

        yield tmpdir, registry_path, model_file, model

# -----------------------------
# Test get_active_version()
# -----------------------------
def test_get_active_version_default(monkeypatch):
    # Registry doesn't exist
    monkeypatch.setattr(model_loader, "REGISTRY_PATH", "/non/existent/path.json")
    assert model_loader.get_active_version() == "legacy"

def test_get_active_version_from_registry(temp_registry_and_model, monkeypatch):
    tmpdir, registry_path, _, _ = temp_registry_and_model
    monkeypatch.setattr(model_loader, "REGISTRY_PATH", registry_path)
    assert model_loader.get_active_version() == "v1"

# -----------------------------
# Test load_active_model()
# -----------------------------
def test_load_active_model_success(temp_registry_and_model, monkeypatch):
    tmpdir, registry_path, _, model = temp_registry_and_model

    # Create the model directory and save the model file
    model_dir = os.path.join(tmpdir, "model", "v1")
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, "logistic_model.joblib")
    joblib.dump(model, model_file)

    # Patch REGISTRY_PATH to use the temp registry
    monkeypatch.setattr(model_loader, "REGISTRY_PATH", registry_path)

    # Patch BASE_DIR so load_active_model looks inside tmpdir
    monkeypatch.setattr(model_loader, "BASE_DIR", tmpdir)

    loaded_model = model_loader.load_active_model()
    assert loaded_model is not None


def test_load_active_model_missing_file(monkeypatch):
    # Point to a registry that has a version but no model file
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "registry.json")
        with open(registry_path, "w") as f:
            json.dump({"active": "v1"}, f)
        monkeypatch.setattr(model_loader, "REGISTRY_PATH", registry_path)
        with pytest.raises(FileNotFoundError):
            model_loader.load_active_model()

def test_load_active_model_legacy(monkeypatch):
    # Registry missing => default legacy
    monkeypatch.setattr(model_loader, "REGISTRY_PATH", "/non/existent/path.json")
    with pytest.raises(FileNotFoundError):
        model_loader.load_active_model()