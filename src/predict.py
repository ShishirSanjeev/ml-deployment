"""
src/predict.py
--------------
Inference utilities.
Loads saved model artifacts and runs predictions on new data.
Used by api/main.py and can also be run as a standalone script.

Usage (standalone):
    python -m src.predict --input data/new_customers.csv --output data/predictions.csv
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocess import engineer_features


# ── Artifact loader ───────────────────────────────────────────────────────────
def load_artifacts(models_dir: str) -> tuple:
    """
    Load model pipeline, threshold, and feature names from disk.

    Args:
        models_dir: Directory containing .joblib files

    Returns:
        (model, threshold, feature_names)
    """
    models_dir = Path(models_dir)

    model         = joblib.load(models_dir / "churn_model.joblib")
    threshold     = joblib.load(models_dir / "best_threshold.joblib")
    feature_names = joblib.load(models_dir / "feature_names.joblib")

    print(f"[load_artifacts] Model loaded | Threshold: {threshold:.4f}")
    return model, threshold, feature_names


# ── Risk level helper ─────────────────────────────────────────────────────────
def get_risk_level(prob: float) -> str:
    """
    Map churn probability to a human-readable risk tier.

    Thresholds:
        >= 0.70 -> High
        >= 0.40 -> Medium
        <  0.40 -> Low
    """
    if prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    return "Low"


# ── Single prediction ─────────────────────────────────────────────────────────
def predict_single(
    customer: dict,
    model,
    threshold: float,
    feature_names: list
) -> dict:
    """
    Run inference on a single customer record.

    Args:
        customer:      Dict of raw customer features (matches API schema)
        model:         Loaded sklearn pipeline
        threshold:     Decision threshold
        feature_names: Expected column order

    Returns:
        Dict with churn_prediction, churn_probability, risk_level
    """
    df = pd.DataFrame([customer])
    df = engineer_features(df)
    df = df[feature_names]

    prob = float(model.predict_proba(df)[0][1])
    prediction = int(prob >= threshold)

    return {
        "churn_prediction":  prediction,
        "churn_probability": round(prob, 4),
        "risk_level":        get_risk_level(prob),
        "threshold_used":    round(threshold, 4)
    }


# ── Batch prediction ──────────────────────────────────────────────────────────
def predict_batch(
    df: pd.DataFrame,
    model,
    threshold: float,
    feature_names: list
) -> pd.DataFrame:
    """
    Run inference on a DataFrame of customer records.

    Args:
        df:            Raw customer DataFrame (no target column)
        model:         Loaded sklearn pipeline
        threshold:     Decision threshold
        feature_names: Expected column order

    Returns:
        Original DataFrame with 3 new columns appended:
            churn_probability, churn_prediction, risk_level
    """
    df_feat = engineer_features(df.copy())
    df_feat = df_feat[feature_names]

    probas      = model.predict_proba(df_feat)[:, 1]
    predictions = (probas >= threshold).astype(int)
    risk_levels = [get_risk_level(p) for p in probas]

    result = df.copy()
    result["churn_probability"] = np.round(probas, 4)
    result["churn_prediction"]  = predictions
    result["risk_level"]        = risk_levels

    churn_count = predictions.sum()
    print(f"[predict_batch] {len(df)} customers | {churn_count} predicted to churn "
          f"({churn_count/len(df)*100:.1f}%)")

    return result


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch churn predictions.")
    parser.add_argument("--input",  type=str, required=True,                    help="Path to input CSV")
    parser.add_argument("--output", type=str, default="data/predictions.csv",   help="Path to save predictions")
    parser.add_argument("--models", type=str, default="models/",                help="Directory with model artifacts")
    args = parser.parse_args()

    model, threshold, feature_names = load_artifacts(args.models)

    df_input = pd.read_csv(args.input)
    print(f"[main] Loaded {len(df_input)} rows from {args.input}")

    df_output = predict_batch(df_input, model, threshold, feature_names)
    df_output.to_csv(args.output, index=False)
    print(f"[main] Predictions saved to {args.output}")
