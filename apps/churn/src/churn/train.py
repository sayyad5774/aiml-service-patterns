import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")

@dataclass(frozen=True)
class TrainConfig:
    n_rows: int = 2000
    random_state: int = 42

def _make_synthetic_data(cfg: TrainConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state)
    tenure = rng.integers(0, 72, size=cfg.n_rows)
    monthly = rng.normal(80, 30, size=cfg.n_rows).clip(10, 200)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], size=cfg.n_rows, p=[0.65, 0.2, 0.15])
    internet = rng.random(cfg.n_rows) < 0.75
    streaming = (rng.random(cfg.n_rows) < 0.55) & internet

    # simple churn signal: short tenure + high monthly + month-to-month increases churn
    logit = (
        -0.03 * tenure
        + 0.015 * (monthly - 80)
        + 0.9 * (contract == "month-to-month")
        + 0.25 * internet
        + 0.2 * streaming
        - 0.7
    )
    prob = 1 / (1 + np.exp(-logit))
    y = rng.random(cfg.n_rows) < prob

    df = pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "contract_type": contract,
        "has_internet": internet,
        "has_streaming": streaming,
        "churn": y.astype(int),
    })
    return df

def train_and_save(cfg: TrainConfig = TrainConfig()) -> str:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = _make_synthetic_data(cfg)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    cat_cols = ["contract_type"]
    num_cols = ["tenure_months", "monthly_charges", "has_internet", "has_streaming"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    clf = LogisticRegression(max_iter=200)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)

    joblib.dump(pipe, MODEL_PATH)
    return MODEL_PATH

def main() -> None:
    path = train_and_save()
    print(f"saved_model={path}")

if __name__ == "__main__":
    main()
