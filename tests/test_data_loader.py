"""Unit tests for src.data_loader."""

import pytest
import pandas as pd
from src.data_loader import (
    load_raw_data,
    get_stratified_split,
    MALIGNANT_LABEL,
    BENIGN_LABEL,
)


def test_load_raw_data_shape_and_types():
    X, y, meta = load_raw_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (569, 30)
    assert len(y) == 569
    assert X.isnull().sum().sum() == 0
    assert set(y.unique()) == {MALIGNANT_LABEL, BENIGN_LABEL}


def test_target_label_convention():
    _, y, meta = load_raw_data()
    # In sklearn: 0 is malignant, 1 is benign
    assert meta["target_mapping"][0] == "malignant"
    assert meta["target_mapping"][1] == "benign"
    assert meta["class_distribution"]["malignant_count"] == 212
    assert meta["class_distribution"]["benign_count"] == 357


def test_stratified_split_ratios():
    X_train, X_test, y_train, y_test = get_stratified_split(test_size=0.20, random_state=42)
    assert len(X_train) == 455
    assert len(X_test) == 114

    # Check stratification preservation
    train_ratio = (y_train == MALIGNANT_LABEL).mean()
    test_ratio = (y_test == MALIGNANT_LABEL).mean()
    assert pytest.approx(train_ratio, abs=0.01) == test_ratio
