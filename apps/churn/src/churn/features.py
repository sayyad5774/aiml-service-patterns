import pandas as pd

CONTRACT_MAP = {"month-to-month": 0, "one-year": 1, "two-year": 2}

def featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["contract_code"] = out["contract_type"].map(CONTRACT_MAP).fillna(0).astype(int)
    out["has_internet"] = out["has_internet"].astype(int)
    out["has_streaming"] = out["has_streaming"].astype(int)
    return out[["tenure_months", "monthly_charges", "contract_code", "has_internet", "has_streaming"]]
