"""Preprocessing and feature selection module.

This module enforces the critical separation between:
    1. Experimental 30-feature baseline pipeline
    2. Production 15-feature reduced pipeline

DATA LEAKAGE PREVENTION:
    Feature selection is fitted STRICTLY on X_train. The resulting selected
    feature names are persisted and used to define the production feature schema.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import MALIGNANT_LABEL, BENIGN_LABEL

# Human-friendly feature metadata (descriptions and physical units)
FEATURE_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "mean radius": {
        "description": "Mean distance from center to perimeter points",
        "unit": "mm",
        "category": "Size",
    },
    "mean perimeter": {
        "description": "Mean nuclear perimeter",
        "unit": "mm",
        "category": "Size",
    },
    "mean area": {
        "description": "Mean nuclear surface area",
        "unit": "mm²",
        "category": "Size",
    },
    "mean compactness": {
        "description": "Mean compactness: (perimeter² / area - 1.0)",
        "unit": "dimensionless ratio",
        "category": "Shape",
    },
    "mean concavity": {
        "description": "Mean severity of concave portions of the contour",
        "unit": "dimensionless ratio",
        "category": "Shape",
    },
    "mean concave points": {
        "description": "Mean number of concave portions along the contour",
        "unit": "count",
        "category": "Shape",
    },
    "radius error": {
        "description": "Standard error of radius measurements",
        "unit": "mm",
        "category": "Variability",
    },
    "perimeter error": {
        "description": "Standard error of perimeter measurements",
        "unit": "mm",
        "category": "Variability",
    },
    "area error": {
        "description": "Standard error of area measurements",
        "unit": "mm²",
        "category": "Variability",
    },
    "worst radius": {
        "description": "Mean of the 3 largest radius measurements",
        "unit": "mm",
        "category": "Size",
    },
    "worst perimeter": {
        "description": "Mean of the 3 largest perimeter measurements",
        "unit": "mm",
        "category": "Size",
    },
    "worst area": {
        "description": "Mean of the 3 largest area measurements",
        "unit": "mm²",
        "category": "Size",
    },
    "worst compactness": {
        "description": "Worst compactness measurement",
        "unit": "dimensionless ratio",
        "category": "Shape",
    },
    "worst concavity": {
        "description": "Worst concavity measurement",
        "unit": "dimensionless ratio",
        "category": "Shape",
    },
    "worst concave points": {
        "description": "Worst number of concave portions along contour",
        "unit": "count",
        "category": "Shape",
    },
}


def select_features_on_training_data(
    X_train: pd.DataFrame, y_train: pd.Series, k: int = 15
) -> Tuple[List[str], pd.DataFrame]:
    """Determines top k features using ANOVA F-value strictly on the training set.

    Guarantees zero data leakage by never accessing test data.

    Args:
        X_train: Training features.
        y_train: Training target.
        k: Number of features to select (default: 15).

    Returns:
        selected_features (list): Names of the top k features.
        score_df (pd.DataFrame): All features ranked by ANOVA F-statistic and p-value.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)

    score_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "f_statistic": selector.scores_,
            "p_value": selector.pvalues_,
            "selected": selector.get_support(),
        }
    ).sort_values("f_statistic", ascending=False)

    selected_features = list(X_train.columns[selector.get_support()])
    return selected_features, score_df


def create_production_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Builds the 15-feature production pipeline.

    Input expected at fit and predict: EXACTLY the 15 selected features.
    Pipeline structure:
        StandardScaler() -> Classifier
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def create_experimental_30_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Builds the 30-feature experimental benchmark pipeline.

    Input expected at fit and predict: ALL 30 original features.
    Pipeline structure:
        StandardScaler() -> Classifier
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def compute_feature_metadata(
    X_train: pd.DataFrame, y_train: pd.Series, selected_features: List[str]
) -> Dict[str, Any]:
    """Computes comprehensive feature statistics for UI input sliders and presets.

    Calculates:
        - Global min, max, mean, std, 25%, median, 75%
        - Benign-specific median and IQR
        - Malignant-specific median and IQR
    """
    metadata: Dict[str, Any] = {}

    for feat in selected_features:
        series = X_train[feat]
        benign_series = series[y_train == BENIGN_LABEL]
        malignant_series = series[y_train == MALIGNANT_LABEL]

        info = FEATURE_DESCRIPTIONS.get(
            feat,
            {"description": feat, "unit": "unit", "category": "General"},
        )

        metadata[feat] = {
            "description": info["description"],
            "unit": info["unit"],
            "category": info["category"],
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(series.median()),
            "benign_median": float(benign_series.median()),
            "benign_mean": float(benign_series.mean()),
            "benign_std": float(benign_series.std()),
            "malignant_median": float(malignant_series.median()),
            "malignant_mean": float(malignant_series.mean()),
            "malignant_std": float(malignant_series.std()),
        }

    return metadata


if __name__ == "__main__":
    from src.data_loader import get_stratified_split

    X_train, X_test, y_train, y_test = get_stratified_split()
    selected_features, scores = select_features_on_training_data(X_train, y_train, k=15)
    print("Selected 15 Features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")
    meta = compute_feature_metadata(X_train, y_train, selected_features)
    print(f"\nMetadata computed for {len(meta)} features.")
