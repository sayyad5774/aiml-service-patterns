from churn.train import train_and_save

def test_train_saves_model(tmp_path, monkeypatch):
    # point artifact dir into tmp
    import churn.train as t
    monkeypatch.setattr(t, "ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(t, "MODEL_PATH", str(tmp_path / "model.joblib"))
    p = train_and_save()
    assert p.endswith("model.joblib")
