from fastapi.testclient import TestClient
from churn.serve import app
from churn.train import train_and_save

def test_predict_endpoint(tmp_path, monkeypatch):
    # Force model artifacts into temp dir for test isolation
    import churn.train as t
    import churn.serve as s

    monkeypatch.setattr(t, "ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(t, "MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setattr(s, "MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setattr(s, "_model", None)

    train_and_save()

    client = TestClient(app)
    r = client.post("/predict", json={
        "tenure_months": 6,
        "monthly_charges": 110,
        "contract_type": "month-to-month",
        "has_internet": True,
        "has_streaming": True
    })
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_label"], bool)
