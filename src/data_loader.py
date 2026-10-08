"""Data loading, validation, and splitting module.

Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)
Source: UCI Machine Learning Repository / scikit-learn datasets

CRITICAL TARGET LABEL CONVENTION (sklearn):
    - 0 = 'malignant'
    - 1 = 'benign'
Throughout this project, Malignant is treated as the primary target class of interest (safety-critical).
"""

from typing import Tuple, Dict, Any
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Target label definitions
MALIGNANT_LABEL: int = 0
BENIGN_LABEL: int = 1
LABEL_NAMES: Dict[int, str] = {
    MALIGNANT_LABEL: "Malignant",
    BENIGN_LABEL: "Benign",
}


def load_raw_data() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Loads the Wisconsin Breast Cancer dataset from scikit-learn.

    Returns:
        X (pd.DataFrame): 30 continuous clinical features.
        y (pd.Series): Target labels (0 = malignant, 1 = benign).
        metadata (dict): Metadata including feature names, target names, description.
    """
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    y = raw.target.copy()

    metadata = {
        "feature_names": list(raw.feature_names),
        "target_names": list(raw.target_names),
        "target_mapping": {0: "malignant", 1: "benign"},
        "n_samples": len(X),
        "n_features": X.shape[1],
        "class_distribution": {
            "malignant_count": int((y == MALIGNANT_LABEL).sum()),
            "benign_count": int((y == BENIGN_LABEL).sum()),
            "malignant_ratio": float((y == MALIGNANT_LABEL).mean()),
            "benign_ratio": float((y == BENIGN_LABEL).mean()),
        },
    }

    validate_dataset(X, y)
    return X, y, metadata


def validate_dataset(X: pd.DataFrame, y: pd.Series) -> None:
    """Performs integrity checks on the raw dataset.

    Raises:
        ValueError: If nulls, shape mismatches, or unexpected labels are found.
    """
    if X.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values.")
    if len(X) != len(y):
        raise ValueError(f"Feature count ({len(X)}) does not match target count ({len(y)}).")
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"Unexpected target labels found: {set(y.unique())}. Expected {{0, 1}}.")
    if X.shape[1] != 30:
        raise ValueError(f"Expected 30 features, but got {X.shape[1]}.")


def get_stratified_split(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the dataset into train and test sets using stratified sampling.

    Args:
        test_size: Proportion of dataset allocated to the holdout test set (default: 0.20).
        random_state: Random seed for exact reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X, y, _ = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def save_dataset_splits(
    output_dir: str = "data", test_size: float = 0.2, random_state: int = 42
) -> None:
    """Saves raw CSV and train/test splits to the data directory."""
    raw_dir = os.path.join(output_dir, "raw")
    proc_dir = os.path.join(output_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    X, y, _ = load_raw_data()
    full_df = X.copy()
    full_df["target"] = y
    full_df["target_label"] = y.map(LABEL_NAMES)
    full_df.to_csv(os.path.join(raw_dir, "breast_cancer_raw.csv"), index=False)

    X_train, X_test, y_train, y_test = get_stratified_split(
        test_size=test_size, random_state=random_state
    )

    train_df = X_train.copy()
    train_df["target"] = y_train
    train_df.to_csv(os.path.join(proc_dir, "train.csv"), index=False)

    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df.to_csv(os.path.join(proc_dir, "test.csv"), index=False)


if __name__ == "__main__":
    X, y, meta = load_raw_data()
    print(f"Dataset successfully loaded and verified.")
    print(f"Features: {meta['n_features']}, Samples: {meta['n_samples']}")
    print(f"Class distribution: {meta['class_distribution']}")
    save_dataset_splits()
    print("Raw and split datasets successfully written to data/ directory.")
