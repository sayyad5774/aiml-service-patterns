import os
import joblib
from fastapi import FastAPI
from churn.schemas import ChurnFeatures, ChurnPrediction

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")

app = FastAPI(title="Churn Service", version="0.1.0")

_model = None

def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Run: python -m churn.train")
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=ChurnPrediction)
def predict(features: ChurnFeatures):
    model = _load_model()
    X = [{
        "tenure_months": features.tenure_months,
        "monthly_charges": features.monthly_charges,
        "contract_type": features.contract_type,
        "has_internet": features.has_internet,
        "has_streaming": features.has_streaming,
    }]
    proba = float(model.predict_proba(X)[0][1])
    return ChurnPrediction(churn_probability=proba, churn_label=(proba >= 0.5))
@app.post("/predict", response_model=ChurnPrediction)
def predict(features: ChurnFeatures):
    import pandas as pd

    model = _load_model()

    row = {
        "tenure_months": features.tenure_months,
        "monthly_charges": features.monthly_charges,
        "contract_type": features.contract_type,
        "has_internet": features.has_internet,
        "has_streaming": features.has_streaming,
    }

    X = pd.DataFrame([row])
    proba = float(model.predict_proba(X)[0][1])

    return ChurnPrediction(
        churn_probability=proba,
        churn_label=(proba >= 0.5),
    )

def main():
    import uvicorn
    uvicorn.run("churn.serve:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
from fastapi import FastAPI, HTTPException
import os
import joblib
import traceback

from churn.schemas import ChurnFeatures, ChurnPrediction

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")

app = FastAPI(title="Churn Service", version="0.1.0")
_model = None

def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Run: python -m churn.train")
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=ChurnPrediction)
def predict(features: ChurnFeatures):
    try:
        import pandas as pd

        model = _load_model()

        X = pd.DataFrame([{
            "tenure_months": features.tenure_months,
            "monthly_charges": features.monthly_charges,
            "contract_type": features.contract_type,
            "has_internet": features.has_internet,
            "has_streaming": features.has_streaming,
        }])

        proba = float(model.predict_proba(X)[0][1])
        return ChurnPrediction(churn_probability=proba, churn_label=(proba >= 0.5))

    except Exception as e:
        # Print traceback to server logs for fast diagnosis
        traceback.print_exc()
        # Return JSON error to client
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
