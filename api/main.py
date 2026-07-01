from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
import os

# Import our modular utilities
from src.predict import load_artifacts, predict_single, get_risk_level

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a customer is likely to churn using an XGBoost model.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load artifacts (Modularized) ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

try:
    model, threshold, feat_names = load_artifacts(MODELS_DIR)
except Exception as e:
    # Use fallback path for local development if directory structure differs
    try:
        model, threshold, feat_names = load_artifacts("models")
    except:
        raise RuntimeError(f"Failed to load model artifacts from {MODELS_DIR}: {e}")


# ── Request schema ────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    # Demographics
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")

    # Account info
    tenure: int = Field(..., example=12, ge=0)
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=65.5, ge=0)
    TotalCharges: Optional[float] = Field(None, example=786.0)

    # Services
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")

    @validator("gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling")
    def validate_binary(cls, v):
        if v not in ["Yes", "No", "Male", "Female"]:
             raise ValueError("Check allowed values (Yes/No or Male/Female)")
        return v


# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    threshold_used: float
    model_version: str = "1.1.0"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"message": "Customer Churn Prediction API is online.", "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": True, "threshold": round(threshold, 4)}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    try:
        # Use modular prediction logic from src.predict
        result = predict_single(customer.dict(), model, threshold, feat_names)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerFeatures]):
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Batch size limit is 100.")

    results = []
    for customer in customers:
        try:
            res = predict_single(customer.dict(), model, threshold, feat_names)
            results.append(res)
        except Exception as e:
            results.append({"error": str(e)})

    return {"predictions": results, "count": len(results)}
