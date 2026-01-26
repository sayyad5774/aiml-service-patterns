from pydantic import BaseModel, Field

class ChurnFeatures(BaseModel):
    tenure_months: int = Field(ge=0, le=240)
    monthly_charges: float = Field(ge=0, le=5000)
    contract_type: str  # "month-to-month", "one-year", "two-year"
    has_internet: bool
    has_streaming: bool

class ChurnPrediction(BaseModel):
    churn_probability: float
    churn_label: bool
