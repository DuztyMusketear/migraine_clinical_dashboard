import pytest
import pandas as pd
from webapp.app import app, ADMIN_USERNAME, ADMIN_PASSWORD

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        with app.app_context():
            yield client

@pytest.fixture
def admin_session(client):
    response = client.post(
        "/login",
        data={
            "username": ADMIN_USERNAME,
            "password": ADMIN_PASSWORD
        },
        follow_redirects=True
    )

    with client.session_transaction() as sess:
        assert sess.get("is_admin") is True

    return client

@pytest.fixture
def sample_df():
    """Small sample for testing predictions"""
    df = pd.DataFrame({
        "patient_id": [1, 2],
        "Age": [25, 30],
        "Duration": [3, 5],
        "Frequency": [2, 3],
        "Location": [1, 2],
        "Intensity": [2, 3],
        "Nausea": [0, 1],
        "Vomit": [0, 1],
        "Phonophobia": [0, 1],
        "Photophobia": [0, 1],
        "Visual": [1, 0],
        "Sensory": [0, 1],
        "Dysphasia": [0, 0],
        "Dysarthria": [0, 0],
        "Vertigo": [0, 1],
        "Tinnitus": [0, 0],
        "Hypoacusis": [0, 0],
        "Diplopia": [0, 0],
        "Defect": [0, 0],
        "Conscience": [1, 1],
        "Paresthesia": [0, 1],
        "DPF": [0, 0],
        "Type_Familial hemiplegic migraine": [0, 0],
        "Type_Migraine without aura": [1, 0],
        "Type_Other": [0, 0],
        "Type_Sporadic hemiplegic migraine": [0, 0],
        "Type_Typical aura with migraine": [0, 0],
        "Type_Typical aura without migraine": [0, 0]
    })
    return df

# -----------------------------
# Basic endpoint tests
# -----------------------------
def test_index_get(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Migraine Prediction Dashboard" in response.data

def test_login_success(client):
    response = client.post('/login', data={
        "username": ADMIN_USERNAME,
        "password": ADMIN_PASSWORD
    }, follow_redirects=True)
    assert response.status_code == 200
    with client.session_transaction() as sess:
        assert sess.get("is_admin") is True

def test_login_failure(client):
    response = client.post('/login', data={
        "username": "wrong",
        "password": "wrong"
    }, follow_redirects=True)
    assert response.status_code == 200
    with client.session_transaction() as sess:
        assert sess.get("is_admin") is None

def test_logout(client):
    with client.session_transaction() as sess:
        sess['is_admin'] = True
    response = client.get('/logout', follow_redirects=True)
    assert response.status_code == 200
    with client.session_transaction() as sess:
        assert sess.get("is_admin") is False

def test_login_logout_flow(client):
    response = client.post("/login", data={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}, follow_redirects=True)
    with client.session_transaction() as sess:
        assert sess.get("is_admin") is True
    response = client.get("/logout", follow_redirects=True)
    with client.session_transaction() as sess:
        assert sess.get("is_admin") is False

def test_invalid_login(client):
    response = client.post("/login", data={"username": "wrong", "password": "wrong"}, follow_redirects=True)
    with client.session_transaction() as sess:
        assert sess.get("is_admin") is None

# -----------------------------
# Patient prediction tests
# -----------------------------
def test_patient_prediction(client, monkeypatch, sample_df):
    # Patch run_pipeline
    monkeypatch.setattr("webapp.app.run_pipeline", lambda: sample_df)
    # Mock model
    class MockModel:
        feature_names_in_ = sample_df.columns[1:]
        coef_ = [[0.1] * len(feature_names_in_)]
        def predict(self, X): return [1]*len(X)
        def predict_proba(self, X): return [[0.2, 0.8]]*len(X)
    monkeypatch.setattr("webapp.app.load_active_model", lambda: MockModel())
    monkeypatch.setattr(
    "webapp.app.patient_feature_contribution",
    lambda model, X: pd.DataFrame({
        "Feature": X.columns.tolist(),
        "Contribution": [0.1]*len(X.columns),
        "Coefficient": [0.1]*len(X.columns)  # float
    })
)
    monkeypatch.setattr("webapp.app.global_feature_summary", lambda model: pd.DataFrame({"feature": model.feature_names_in_, "coef": model.coef_[0]}))

    response = client.post("/", data={"patient_id": 1})
    assert response.status_code == 200
    assert b"Prediction" in response.data or b"prediction" in response.data

def test_invalid_patient(client, monkeypatch):
    monkeypatch.setattr("webapp.app.run_pipeline", lambda: pd.DataFrame({"patient_id":[1]}))
    class DummyModel:
        feature_names_in_ = ["dummy"]
        coef_ = [[0.0]]
        def predict(self, X): return [0]
        def predict_proba(self, X): return [[0.5,0.5]]
    monkeypatch.setattr("webapp.app.load_active_model", lambda: DummyModel())
    monkeypatch.setattr("webapp.app.global_feature_summary", lambda model: pd.DataFrame())
    response = client.post("/", data={"patient_id": 999})
    assert response.status_code == 200

# -----------------------------
# Metrics tests
# -----------------------------
def test_metrics_access_control(client):
    response = client.get("/metrics", follow_redirects=True)
    assert response.status_code == 200
    assert b"Migraine Prediction Dashboard" in response.data

def test_metrics_page(admin_session, monkeypatch):
    fake_metrics = {
        "v1": [{
            "timestamp": "2024-01-20T19:12:03",
            "Accuracy": 0.94,
            "AUC": 0.947,
        }]
    }

    monkeypatch.setattr("webapp.app.load_metrics", lambda: fake_metrics)

    response = admin_session.get("/metrics", follow_redirects=False)

    assert response.status_code == 200
    data = response.data.lower()
    assert b"accuracy" in data or b"auc" in data
        
def test_empty_metrics_file(admin_session, monkeypatch):
    monkeypatch.setattr("webapp.app.load_metrics", lambda: {"v1": []})
    response = admin_session.get("/metrics", follow_redirects=True)
    assert response.status_code == 200
    # Page should handle empty metrics gracefully
    assert b"no metrics" in response.data.lower() or b"metrics" in response.data.lower()