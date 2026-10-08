"""CLI entry point for reproducible model training and benchmarking.

Usage:
    py -3.11 train_model.py

Outputs:
    - models/best_model.joblib (Champion 15-feature pipeline)
    - models/all_models.joblib (All 7 candidate 15-feature pipelines)
    - models/metrics_summary.json (Verified evaluation metrics)
    - models/feature_metadata.json (Feature stats, ranges, units)
    - assets/ (ROC curves, confusion matrices, correlation plots)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import (
    load_raw_data,
    get_stratified_split,
    save_dataset_splits,
    LABEL_NAMES,
    MALIGNANT_LABEL,
    BENIGN_LABEL,
)
from src.preprocessing import (
    select_features_on_training_data,
    compute_feature_metadata,
)
from src.train import run_full_training_suite


def ensure_directories():
    for d in ["models", "assets", "data/raw", "data/processed"]:
        os.makedirs(d, exist_ok=True)


def plot_and_save_artifacts(results: dict, X_train: pd.DataFrame, y_train: pd.Series):
    """Generates and saves visual charts to the assets/ directory."""
    champion_name = results["champion_name"]
    test_metrics = results["production_test_metrics"]
    curves_data = results["curves_data"]

    # 1. Multi-Model ROC Curve Comparison
    plt.figure(figsize=(10, 7))
    for name, curve in curves_data.items():
        auc_score = test_metrics[name].get("roc_auc", 0.0)
        plt.plot(
            curve["roc"]["fpr"],
            curve["roc"]["tpr"],
            label=f"{name} (AUC = {auc_score:.4f})",
            linewidth=2 if name == champion_name else 1.2,
            linestyle="-" if name == champion_name else "--",
        )
    plt.plot([0, 1], [0, 1], "k:", label="Random Chance (AUC = 0.50)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Malignant Recall)", fontsize=12)
    plt.title("ROC Curves Comparison - 15-Feature Production Pipeline", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("assets/roc_curves_comparison.png", dpi=300)
    plt.close()

    # 2. Champion Model Confusion Matrix Heatmap
    cm_raw = test_metrics[champion_name]["confusion_matrix"]["raw_matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Malignant (0)", "Pred Benign (1)"],
        yticklabels=["True Malignant (0)", "True Benign (1)"],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(f"Confusion Matrix: {champion_name} (Holdout Test Set)", fontsize=13, fontweight="bold")
    plt.ylabel("True Diagnosis", fontsize=11)
    plt.xlabel("Predicted Diagnosis", fontsize=11)
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix_champion.png", dpi=300)
    plt.close()

    # 3. Correlation Heatmap of Selected Features
    selected_features = results["selected_features"]
    corr = X_train[selected_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Matrix of Top 15 Selected Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/feature_correlation_matrix.png", dpi=300)
    plt.close()

    print("\nVisual diagnostic plots saved to assets/")


def main():
    print("=" * 70)
    print("CANCER PREDICTION SYSTEM - REPRODUCIBLE ML PIPELINE RUNNER")
    print("Target Convention: 0 = Malignant (Positive Class), 1 = Benign")
    print("=" * 70)

    ensure_directories()
    save_dataset_splits()

    X_train, X_test, y_train, y_test = get_stratified_split(test_size=0.20, random_state=42)

    # Leak-free feature selection on X_train only
    print("\n[Step 1] Running leak-free ANOVA F-statistic feature selection on X_train...")
    selected_features, feature_scores = select_features_on_training_data(X_train, y_train, k=15)
    print(f"Top 15 features selected without data leakage.")

    # Compute feature metadata
    print("\n[Step 2] Computing feature distribution metadata for UI and presets...")
    feature_meta = compute_feature_metadata(X_train, y_train, selected_features)
    with open("models/feature_metadata.json", "w") as f:
        json.dump(feature_meta, f, indent=2)
    print("models/feature_metadata.json saved.")

    # Train all candidate models following strict leak-free protocol
    print("\n[Step 3] Training, tuning, and selecting champion strictly on X_train...")
    results = run_full_training_suite(X_train, X_test, y_train, y_test, selected_features)

    champion_name = results["champion_name"]
    best_pipeline = results["production_models"][champion_name]

    # Save artifacts
    print("\n[Step 4] Persisting production model pipelines and metrics...")
    joblib.dump(best_pipeline, "models/best_model.joblib")
    joblib.dump(results["production_models"], "models/all_models.joblib")

    # Serialize metrics summary (clean json)
    serializable_summary = {
        "champion_model": champion_name,
        "selection_protocol": "Selected strictly via Stratified 5-Fold Cross-Validation on X_train before touching test set",
        "label_mapping": {"0": "Malignant", "1": "Benign"},
        "positive_class": "Malignant (label 0)",
        "selected_features_15": selected_features,
        "production_15_cv_metrics": results["production_cv_metrics"],
        "production_15_metrics": results["production_test_metrics"],
        "experimental_30_cv_metrics": results["baseline_30_cv_metrics"],
        "experimental_30_metrics": results["baseline_30_test_metrics"],
        "curves_data": results["curves_data"],
    }
    with open("models/metrics_summary.json", "w") as f:
        json.dump(serializable_summary, f, indent=2)
    print("models/best_model.joblib, all_models.joblib, and metrics_summary.json saved.")

    # Generate charts
    print("\n[Step 5] Generating artifact visualizations...")
    plot_and_save_artifacts(results, X_train, y_train)

    # Print summary table
    print("\n" + "=" * 90)
    print("VERIFIED MODEL BENCHMARK RESULTS (CV ON X_train VS HOLDOUT TEST SET N=114)")
    print("=" * 90)
    headers = f"{'Model':<25} | {'15-Feat CV AUC':<14} | {'Test AUC':<10} | {'Test Recall':<12} | {'Test Acc':<10} | {'30-Feat Test AUC':<16}"
    print(headers)
    print("-" * len(headers))
    for name in results["production_test_metrics"]:
        cv15 = results["production_cv_metrics"][name]
        test15 = results["production_test_metrics"][name]
        test30 = results["baseline_30_test_metrics"][name]
        champ_mark = " (CHAMPION)" if name == champion_name else ""
        print(
            f"{name + champ_mark:<25} | "
            f"{cv15.get('roc_auc', 0.0):<14.4f} | "
            f"{test15.get('roc_auc', 0.0):<10.4f} | "
            f"{test15['malignant_recall']:<12.4f} | "
            f"{test15['accuracy']:<10.4f} | "
            f"{test30.get('roc_auc', 0.0):<16.4f}"
        )
    print("=" * 90)
    print("Training run completed successfully.")


if __name__ == "__main__":
    main()
