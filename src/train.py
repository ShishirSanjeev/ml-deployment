"""
src/train.py
------------
Model training script.
Builds the sklearn preprocessing pipeline, trains XGBoost,
tunes the decision threshold, and saves all artifacts to models/.

Usage:
    python -m src.train --data data/telco_churn.csv --models models/
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, roc_curve
)
from xgboost import XGBClassifier

from src.preprocess import (
    run_preprocessing_pipeline,
    CATEGORICAL_COLS,
    NUMERICAL_COLS
)


# ── Preprocessing pipeline ────────────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - StandardScales numerical columns
    - OneHotEncodes categorical columns (handle_unknown='ignore' for safety)
    """
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, NUMERICAL_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ])
    return preprocessor


# ── Threshold tuning ──────────────────────────────────────────────────────────
def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Find the decision threshold that maximises F1 score on the validation set.

    Args:
        y_true:  True binary labels
        y_proba: Predicted probabilities for class 1

    Returns:
        Optimal threshold (float)
    """
    thresholds = np.arange(0.20, 0.80, 0.01)
    best_thresh, best_f1 = 0.5, 0.0

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"[tune_threshold] Best threshold: {best_thresh:.2f} | F1: {best_f1:.4f}")
    return float(best_thresh)


# ── Main training function ────────────────────────────────────────────────────
def train(data_path: str, models_dir: str) -> None:
    """
    Full training run:
    1. Preprocess data
    2. Train/test split
    3. Build pipeline
    4. Cross-validate
    5. Fit on full train set
    6. Tune threshold on test set
    7. Evaluate and print report
    8. Save artifacts

    Args:
        data_path:  Path to raw CSV
        models_dir: Directory to save model artifacts
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Preprocess ─────────────────────────────────────────────────────────
    X, y = run_preprocessing_pipeline(data_path)

    # ── 2. Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[train] Train: {X_train.shape} | Test: {X_test.shape}")

    # ── 3. Build pipeline ─────────────────────────────────────────────────────
    preprocessor = build_preprocessor()

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb)
    ])

    # ── 4. Cross-validate ─────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"[train] CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 5. Fit ────────────────────────────────────────────────────────────────
    pipeline.fit(X_train, y_train)
    print("[train] Model fitted.")

    # ── 6. Tune threshold ─────────────────────────────────────────────────────
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    best_threshold = tune_threshold(y_test.values, y_proba)

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    y_pred = (y_proba >= best_threshold).astype(int)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"\n[train] === Final Evaluation ===")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    # ── 8. Save artifacts ─────────────────────────────────────────────────────
    model_path     = Path(models_dir) / "churn_model.joblib"
    thresh_path    = Path(models_dir) / "best_threshold.joblib"
    features_path  = Path(models_dir) / "feature_names.joblib"

    joblib.dump(pipeline,        model_path)
    joblib.dump(best_threshold,  thresh_path)
    joblib.dump(list(X.columns), features_path)

    print(f"\n[train] Artifacts saved to {models_dir}/")
    print(f"  {model_path.name}")
    print(f"  {thresh_path.name}")
    print(f"  {features_path.name}")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the churn prediction model.")
    parser.add_argument("--data",   type=str, default="data/telco_churn.csv", help="Path to raw CSV")
    parser.add_argument("--models", type=str, default="models/",              help="Directory to save artifacts")
    args = parser.parse_args()

    train(data_path=args.data, models_dir=args.models)
