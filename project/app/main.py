"""Spaceship Titanic prediction API.   Run:  uvicorn app.main:app --reload"""
import logging
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

# The saved model pickle references `app.features`, so the project root must be importable
# no matter how the host (uvicorn, gunicorn, Vercel) loads this file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.predictor import ModelNotLoadedError, Predictor

logger = logging.getLogger("spaceship")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["predictor"] = Predictor()      # load once at startup, not per request
        logger.info("Model loaded: %s", state["predictor"].metadata.get("model"))
    except ModelNotLoadedError as e:
        logger.error(str(e))                  # API still starts; /health reports the problem
    yield
    state.clear()


app = FastAPI(title="Spaceship Titanic API", version="1.0.0", lifespan=lifespan)

CABIN_RE = re.compile(r"^[A-Z]/\d{1,4}/[PS]$")


class Passenger(BaseModel):
    """One passenger. Optional fields may be omitted/null (the model imputes them)."""
    model_config = ConfigDict(extra="forbid")  # reject typos like "CryoSleeep"

    HomePlanet: Optional[Literal["Earth", "Europa", "Mars"]] = None
    CryoSleep: Optional[bool] = None
    Cabin: Optional[str] = Field(None, examples=["B/0/P"], description="Deck/Number/Side, e.g. F/12/S")
    Destination: Optional[Literal["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"]] = None
    Age: Optional[float] = Field(None, ge=0, le=120)
    VIP: Optional[bool] = None
    RoomService: Optional[float] = Field(None, ge=0)
    FoodCourt: Optional[float] = Field(None, ge=0)
    ShoppingMall: Optional[float] = Field(None, ge=0)
    Spa: Optional[float] = Field(None, ge=0)
    VRDeck: Optional[float] = Field(None, ge=0)

    @field_validator("Cabin")
    @classmethod
    def check_cabin(cls, v):
        if v is not None and not CABIN_RE.match(v):
            raise ValueError("Cabin must look like 'Deck/Number/Side', e.g. 'F/12/S' (side is P or S)")
        return v


class Prediction(BaseModel):
    transported: bool
    probability: float = Field(description="Probability the passenger was transported")


class BatchRequest(BaseModel):
    passengers: list[Passenger] = Field(min_length=1, max_length=500)


def get_predictor() -> Predictor:
    p = state.get("predictor")
    if p is None:  # lazy fallback (e.g. serverless cold start without lifespan)
        try:
            p = state["predictor"] = Predictor()
        except ModelNotLoadedError:
            raise HTTPException(status_code=503, detail="Model is not loaded. Run `python train.py` and redeploy.")
    return p


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [{"field": ".".join(str(x) for x in e["loc"] if x != "body"), "problem": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "Invalid input", "details": errors})


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": "Internal server error. Please try again or contact support."})


@app.get("/", include_in_schema=False)
def root():
    return {"service": "Spaceship Titanic API", "docs": "/docs", "health": "/health", "predict": "POST /predict"}


@app.get("/health")
def health():
    p = state.get("predictor")
    return {"status": "ok" if p else "model_not_loaded", "model": p.metadata.get("model") if p else None}


@app.post("/predict", response_model=Prediction)
def predict(passenger: Passenger):
    predictor = get_predictor()
    try:
        return predictor.predict([passenger.model_dump()])[0]
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed for this input.")


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(req: BatchRequest):
    predictor = get_predictor()
    try:
        return predictor.predict([p.model_dump() for p in req.passengers])
    except Exception:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Batch prediction failed.")
