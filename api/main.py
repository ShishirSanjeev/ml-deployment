from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a customer is likely to churn using an XGBoost model trained on the Telco Customer Churn dataset.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load artifacts ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

try:
    model     = joblib.load(os.path.join(MODELS_DIR, "churn_model.joblib"))
    threshold = joblib.load(os.path.join(MODELS_DIR, "best_threshold.joblib"))
    feat_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    print(f"[OK] Model loaded | Threshold: {threshold:.2f}")
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")


# ── Request schema ────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    # Demographics
    gender: str = Field(..., example="Male", description="Male or Female")
    SeniorCitizen: int = Field(..., example=0, description="1 if senior citizen, else 0")
    Partner: str = Field(..., example="Yes", description="Yes or No")
    Dependents: str = Field(..., example="No", description="Yes or No")

    # Account info
    tenure: int = Field(..., example=12, ge=0, description="Months with the company")
    Contract: str = Field(..., example="Month-to-month", description="Month-to-month / One year / Two year")
    PaperlessBilling: str = Field(..., example="Yes", description="Yes or No")
    PaymentMethod: str = Field(..., example="Electronic check",
                               description="Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)")
    MonthlyCharges: float = Field(..., example=65.5, ge=0)
    TotalCharges: Optional[float] = Field(None, example=786.0, ge=0,
                                          description="Leave null for new customers")

    # Services
    PhoneService: str = Field(..., example="Yes", description="Yes or No")
    MultipleLines: str = Field(..., example="No", description="Yes / No / No phone service")
    InternetService: str = Field(..., example="Fiber optic", description="DSL / Fiber optic / No")
    OnlineSecurity: str = Field(..., example="No", description="Yes / No / No internet service")
    OnlineBackup: str = Field(..., example="No", description="Yes / No / No internet service")
    DeviceProtection: str = Field(..., example="No", description="Yes / No / No internet service")
    TechSupport: str = Field(..., example="No", description="Yes / No / No internet service")
    StreamingTV: str = Field(..., example="Yes", description="Yes / No / No internet service")
    StreamingMovies: str = Field(..., example="Yes", description="Yes / No / No internet service")

    @validator("gender")
    def validate_gender(cls, v):
        if v not in ["Male", "Female"]:
            raise ValueError("gender must be Male or Female")
        return v

    @validator("Contract")
    def validate_contract(cls, v):
        valid = ["Month-to-month", "One year", "Two year"]
        if v not in valid:
            raise ValueError(f"Contract must be one of {valid}")
        return v

    @validator("InternetService")
    def validate_internet(cls, v):
        valid = ["DSL", "Fiber optic", "No"]
        if v not in valid:
            raise ValueError(f"InternetService must be one of {valid}")
        return v


# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_prediction: int = Field(..., description="1 = Churn, 0 = No Churn")
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    risk_level: str = Field(..., description="Low / Medium / High")
    threshold_used: float = Field(..., description="Decision threshold applied")
    model_version: str = "1.0.0"


# ── Feature engineering (mirrors notebook) ───────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    df = df.copy()

    # Fix TotalCharges for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Engineered features
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"]
    ).astype(str)

    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    df["has_streaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)

    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_addons"] = df[addon_cols].apply(lambda row: (row == "Yes").sum(), axis=1)

    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_elec_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    return df


def get_risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    return "Low"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "Customer Churn Prediction API is running.",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "threshold": round(threshold, 4)
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([customer.dict()])

        # Apply feature engineering
        input_df = engineer_features(input_df)

        # Ensure column order matches training
        input_df = input_df[feat_names]

        # Predict
        prob = float(model.predict_proba(input_df)[0][1])
        prediction = int(prob >= threshold)
        risk = get_risk_level(prob)

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(prob, 4),
            risk_level=risk,
            threshold_used=round(threshold, 4)
        )

    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing or unexpected feature: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerFeatures]):
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Batch size limit is 100.")
    results = []
    for customer in customers:
        try:
            input_df = pd.DataFrame([customer.dict()])
            input_df = engineer_features(input_df)
            input_df = input_df[feat_names]
            prob = float(model.predict_proba(input_df)[0][1])
            prediction = int(prob >= threshold)
            results.append({
                "churn_prediction": prediction,
                "churn_probability": round(prob, 4),
                "risk_level": get_risk_level(prob)
            })
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results, "count": len(results)}
