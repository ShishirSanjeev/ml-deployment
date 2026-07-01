"""
src/explain.py
--------------
SHAP-based model explainability utilities.
Generates global and local explanations for the XGBoost model.

Usage (standalone):
    python -m src.explain --data data/telco_churn.csv --models models/ --output reports/
"""

import argparse
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.preprocess import run_preprocessing_pipeline, engineer_features
from src.predict import load_artifacts


# ── SHAP explainer setup ──────────────────────────────────────────────────────
def build_explainer(model, X_transformed: np.ndarray):
    """
    Build a SHAP TreeExplainer for the XGBoost classifier.

    Args:
        model:         Fitted sklearn Pipeline
        X_transformed: Preprocessed feature matrix (numpy array)

    Returns:
        shap.TreeExplainer
    """
    classifier = model.named_steps["classifier"]
    explainer  = shap.TreeExplainer(classifier)
    print("[build_explainer] SHAP TreeExplainer ready.")
    return explainer


def get_transformed_features(model, X: pd.DataFrame) -> tuple[np.ndarray, list]:
    """
    Apply the preprocessing step of the pipeline and retrieve feature names.

    Args:
        model: Fitted sklearn Pipeline
        X:     Raw feature DataFrame (after engineer_features)

    Returns:
        (X_transformed array, feature_names list)
    """
    preprocessor = model.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)

    # Get feature names from ColumnTransformer
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        # Fallback for older sklearn
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    return X_transformed, feature_names


# ── Global explanations ───────────────────────────────────────────────────────
def plot_global_summary(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list,
    output_dir: str,
    max_display: int = 20
) -> None:
    """
    Generate and save a SHAP summary (beeswarm) plot showing global feature importance.

    Args:
        shap_values:   SHAP values array (n_samples x n_features)
        X_transformed: Preprocessed feature matrix
        feature_names: List of feature names
        output_dir:    Directory to save the plot
        max_display:   Number of top features to show
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.title("Global Feature Importance (SHAP Beeswarm)", fontsize=14, pad=15)
    plt.tight_layout()
    save_path = Path(output_dir) / "shap_global_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_global_summary] Saved: {save_path}")


def plot_bar_importance(
    shap_values: np.ndarray,
    feature_names: list,
    output_dir: str,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Generate a bar chart of mean absolute SHAP values and return a ranked DataFrame.

    Args:
        shap_values:   SHAP values array
        feature_names: List of feature names
        output_dir:    Directory to save the plot
        top_n:         Number of top features to display

    Returns:
        DataFrame with feature importance ranking
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "mean_shap":  mean_abs_shap
    }).sort_values("mean_shap", ascending=False).reset_index(drop=True)

    top_df = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df["feature"][::-1], top_df["mean_shap"][::-1], color="#e74c3c")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(f"Top {top_n} Features by Mean |SHAP Value|", fontsize=14)
    plt.tight_layout()

    save_path = Path(output_dir) / "shap_bar_importance.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_bar_importance] Saved: {save_path}")

    return importance_df


# ── Local explanations ────────────────────────────────────────────────────────
def explain_single_prediction(
    customer: dict,
    model,
    threshold: float,
    feature_names_raw: list,
    output_dir: str,
    customer_id: str = "customer_0"
) -> dict:
    """
    Generate a SHAP waterfall plot explaining a single prediction.

    Args:
        customer:          Raw customer feature dict
        model:             Fitted sklearn Pipeline
        threshold:         Decision threshold
        feature_names_raw: Raw feature names (before preprocessing)
        output_dir:        Directory to save the plot
        customer_id:       Label for the output file

    Returns:
        Dict with prediction details and top contributing features
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Preprocess
    df = pd.DataFrame([customer])
    df = engineer_features(df)
    df = df[feature_names_raw]

    X_transformed, feat_names_transformed = get_transformed_features(model, df)

    # SHAP values
    classifier = model.named_steps["classifier"]
    explainer  = shap.TreeExplainer(classifier)
    shap_vals  = explainer.shap_values(X_transformed)

    prob       = float(model.predict_proba(df)[0][1])
    prediction = int(prob >= threshold)

    # Waterfall plot
    explanation = shap.Explanation(
        values         = shap_vals[0],
        base_values    = explainer.expected_value,
        data           = X_transformed[0],
        feature_names  = feat_names_transformed
    )

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, max_display=15, show=False)
    plt.title(f"Local Explanation — {customer_id} | Churn Prob: {prob:.2%}", fontsize=13)
    plt.tight_layout()

    save_path = Path(output_dir) / f"shap_waterfall_{customer_id}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain_single_prediction] Saved: {save_path}")

    # Top contributing features
    contrib_df = pd.DataFrame({
        "feature": feat_names_transformed,
        "shap_value": shap_vals[0]
    }).reindex(pd.Series(shap_vals[0]).abs().sort_values(ascending=False).index)

    return {
        "customer_id":       customer_id,
        "churn_probability": round(prob, 4),
        "churn_prediction":  prediction,
        "top_features":      contrib_df.head(5).to_dict(orient="records")
    }


# ── Full explain run ──────────────────────────────────────────────────────────
def run_explain_pipeline(data_path: str, models_dir: str, output_dir: str) -> None:
    """
    End-to-end explainability run:
    1. Load data and artifacts
    2. Transform features
    3. Compute SHAP values
    4. Save global summary and bar importance plots

    Args:
        data_path:  Path to raw CSV
        models_dir: Directory with model artifacts
        output_dir: Directory to save plots
    """
    # Load
    model, threshold, feature_names = load_artifacts(models_dir)
    X, y = run_preprocessing_pipeline(data_path)
    X = X[feature_names]

    # Sample for speed (SHAP on full dataset can be slow)
    sample_size = min(500, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    print(f"[run_explain_pipeline] Using {sample_size} samples for SHAP.")

    # Transform
    X_transformed, feat_names_transformed = get_transformed_features(model, X_sample)

    # SHAP values
    classifier = model.named_steps["classifier"]
    explainer  = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)
    print(f"[run_explain_pipeline] SHAP values computed. Shape: {shap_values.shape}")

    # Plots
    plot_global_summary(shap_values, X_transformed, feat_names_transformed, output_dir)
    importance_df = plot_bar_importance(shap_values, feat_names_transformed, output_dir)

    print("\n[run_explain_pipeline] Top 10 features by mean |SHAP|:")
    print(importance_df.head(10).to_string(index=False))


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP explanations.")
    parser.add_argument("--data",   type=str, default="data/telco_churn.csv", help="Path to raw CSV")
    parser.add_argument("--models", type=str, default="models/",              help="Directory with model artifacts")
    parser.add_argument("--output", type=str, default="reports/",             help="Directory to save plots")
    args = parser.parse_args()

    run_explain_pipeline(
        data_path  = args.data,
        models_dir = args.models,
        output_dir = args.output
    )
