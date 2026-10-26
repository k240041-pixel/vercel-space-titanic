"""Feature engineering shared by training AND the API.

Keeping this in one module guarantees the model sees identical features
at training time and prediction time (avoids train/serve skew).
"""
import numpy as np
import pandas as pd

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
RAW_COLUMNS = [
    "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP",
    *SPEND_COLS,
]
CATEGORICAL = ["HomePlanet", "Destination", "Deck", "Side", "CryoSleep", "VIP"]
NUMERIC = ["Age", "CabinNum", *SPEND_COLS, "TotalSpend", "NoSpend", "IsChild"]
MODEL_COLUMNS = CATEGORICAL + NUMERIC


def _to_bool_str(s: pd.Series) -> pd.Series:
    """Map True/False/NaN (or 'True'/'False') -> 'True'/'False'/'Unknown'."""
    def conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            return "Unknown"
        return str(bool(v)) if not isinstance(v, str) else v
    return s.map(conv)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Raw passenger columns -> model-ready columns."""
    out = pd.DataFrame(index=df.index)

    # Cabin "Deck/Num/Side" -> three separate features
    cabin = df["Cabin"].astype("object").where(df["Cabin"].notna(), None)
    parts = cabin.map(lambda c: c.split("/") if isinstance(c, str) and c.count("/") == 2 else [None, None, None])
    out["Deck"] = parts.map(lambda p: p[0]).fillna("Unknown")
    out["Side"] = parts.map(lambda p: p[2]).fillna("Unknown")
    out["CabinNum"] = pd.to_numeric(parts.map(lambda p: p[1]), errors="coerce")

    out["HomePlanet"] = df["HomePlanet"].astype("object").fillna("Unknown")
    out["Destination"] = df["Destination"].astype("object").fillna("Unknown")
    out["CryoSleep"] = _to_bool_str(df["CryoSleep"])
    out["VIP"] = _to_bool_str(df["VIP"])
    out["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Domain rule found in EDA: passengers in cryosleep never spend money
    cryo = out["CryoSleep"] == "True"
    for c in SPEND_COLS:
        v = pd.to_numeric(df[c], errors="coerce")
        out[c] = v.mask(cryo & v.isna(), 0.0)

    out["TotalSpend"] = out[SPEND_COLS].sum(axis=1, min_count=1)
    out["NoSpend"] = (out["TotalSpend"].fillna(-1) == 0).astype(int)
    out["IsChild"] = (out["Age"] <= 12).astype(int)  # NaN -> 0
    return out[MODEL_COLUMNS]
