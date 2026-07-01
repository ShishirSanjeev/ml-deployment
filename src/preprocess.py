"""
src/preprocess.py
-----------------
All data loading, cleaning, and feature engineering logic.
Mirrors the transformations in 02_modeling.ipynb so that
training and inference are always consistent.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────
ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_group"
]

NUMERICAL_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "avg_monthly_spend", "has_streaming", "num_addons",
    "is_month_to_month", "is_elec_check"
]

TARGET_COL = "Churn"


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn CSV.

    Args:
        filepath: Path to telco_churn.csv

    Returns:
        Raw DataFrame
    """
    df = pd.read_csv(filepath)
    print(f"[load_raw_data] Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


# ── Cleaning ──────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning steps:
    - Drop customerID (not a feature)
    - Fix TotalCharges (whitespace -> NaN -> fill with 0)
    - Encode target: Churn Yes/No -> 1/0

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()

    # Drop ID column
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Fix TotalCharges — some rows have whitespace strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_nulls = df["TotalCharges"].isna().sum()
    if n_nulls > 0:
        print(f"[clean_data] Filling {n_nulls} NaN TotalCharges with 0.")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Encode target
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
        print(f"[clean_data] Target distribution:\n{df[TARGET_COL].value_counts()}")

    print(f"[clean_data] Done. Shape: {df.shape}")
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 6 derived features that improve model performance.
    Must be applied identically during training AND inference.

    Features created:
        tenure_group       : Binned tenure (0-1yr, 1-2yr, 2-4yr, 4+yr)
        avg_monthly_spend  : TotalCharges / (tenure + 1)
        has_streaming      : 1 if StreamingTV or StreamingMovies == Yes
        num_addons         : Count of active add-on services (0-6)
        is_month_to_month  : 1 if Contract == Month-to-month
        is_elec_check      : 1 if PaymentMethod == Electronic check

    Args:
        df: Cleaned DataFrame (with or without target column)

    Returns:
        DataFrame with engineered features appended
    """
    df = df.copy()

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"]
    ).astype(str)

    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    df["has_streaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)

    df["num_addons"] = df[ADDON_COLS].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_elec_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    print(f"[engineer_features] Done. Shape: {df.shape}")
    return df


# ── Full Pipeline ─────────────────────────────────────────────────────────────
def run_preprocessing_pipeline(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    End-to-end preprocessing: load -> clean -> engineer features -> split X/y.

    Args:
        filepath: Path to raw CSV

    Returns:
        X (features DataFrame), y (target Series)
    """
    df = load_raw_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    print(f"[run_preprocessing_pipeline] X: {X.shape}, y: {y.shape}")
    return X, y
