"""Loads the saved pipeline once and exposes a framework-independent predict function."""
import json
from pathlib import Path

import joblib
import pandas as pd

from app.features import RAW_COLUMNS

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


class ModelNotLoadedError(RuntimeError):
    pass


class Predictor:
    def __init__(self, artifacts_dir: Path = ARTIFACTS):
        model_path = artifacts_dir / "model.joblib"
        if not model_path.exists():
            raise ModelNotLoadedError(f"Model file not found at {model_path}. Run `python train.py` first.")
        self.pipeline = joblib.load(model_path)
        meta_path = artifacts_dir / "metadata.json"
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def predict(self, records: list[dict]) -> list[dict]:
        df = pd.DataFrame(records).reindex(columns=RAW_COLUMNS)  # missing keys -> NaN
        proba = self.pipeline.predict_proba(df)[:, 1]
        return [{"transported": bool(p >= 0.5), "probability": round(float(p), 4)} for p in proba]
