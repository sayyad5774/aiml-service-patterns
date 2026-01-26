from dataclasses import dataclass
import os

@dataclass(frozen=True)
class CommonSettings:
    env: str = os.getenv("ENV", "local")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

COMMON = CommonSettings()
