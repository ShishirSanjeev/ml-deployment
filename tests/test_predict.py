"""
Basic unit tests for the prediction logic.
Run with: pytest tests/test_predict.py -v
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.main import engineer_features, get_risk_level


# ── Sample payload ────────────────────────────────────────────────────────────
SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
}


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_engineer_features_runs():
    df = pd.DataFrame([SAMPLE_CUSTOMER])
    result = engineer_features(df)
    assert "tenure_group" in result.columns
    assert "avg_monthly_spend" in result.columns
    assert "has_streaming" in result.columns
    assert "num_addons" in result.columns
    assert "is_month_to_month" in result.columns
    assert "is_elec_check" in result.columns


def test_has_streaming_true():
    df = pd.DataFrame([SAMPLE_CUSTOMER])
    result = engineer_features(df)
    assert result["has_streaming"].iloc[0] == 1


def test_is_month_to_month_true():
    df = pd.DataFrame([SAMPLE_CUSTOMER])
    result = engineer_features(df)
    assert result["is_month_to_month"].iloc[0] == 1


def test_is_elec_check_true():
    df = pd.DataFrame([SAMPLE_CUSTOMER])
    result = engineer_features(df)
    assert result["is_elec_check"].iloc[0] == 1


def test_num_addons_zero():
    customer = SAMPLE_CUSTOMER.copy()
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]:
        customer[col] = "No"
    df = pd.DataFrame([customer])
    result = engineer_features(df)
    assert result["num_addons"].iloc[0] == 0


def test_total_charges_null_handled():
    customer = SAMPLE_CUSTOMER.copy()
    customer["TotalCharges"] = None
    df = pd.DataFrame([customer])
    result = engineer_features(df)
    assert result["TotalCharges"].iloc[0] == 0.0


def test_risk_level_high():
    assert get_risk_level(0.85) == "High"


def test_risk_level_medium():
    assert get_risk_level(0.55) == "Medium"


def test_risk_level_low():
    assert get_risk_level(0.20) == "Low"


def test_risk_level_boundary():
    assert get_risk_level(0.70) == "High"
    assert get_risk_level(0.40) == "Medium"
    assert get_risk_level(0.39) == "Low"
