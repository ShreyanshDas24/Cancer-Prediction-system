"""Unit tests for src.preprocessing."""

import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.data_loader import get_stratified_split
from src.preprocessing import (
    select_features_on_training_data,
    create_production_pipeline,
    create_experimental_30_pipeline,
    compute_feature_metadata,
)


def test_leak_free_feature_selection():
    X_train, _, y_train, _ = get_stratified_split()
    selected_15, score_df = select_features_on_training_data(X_train, y_train, k=15)

    assert len(selected_15) == 15
    assert len(score_df) == 30
    assert all(feat in X_train.columns for feat in selected_15)


def test_production_and_baseline_pipeline_creation():
    clf = LogisticRegression()
    prod_pipe = create_production_pipeline(clf)
    base_pipe = create_experimental_30_pipeline(clf)

    assert isinstance(prod_pipe, Pipeline)
    assert isinstance(base_pipe, Pipeline)
    assert "scaler" in prod_pipe.named_steps
    assert "classifier" in prod_pipe.named_steps


def test_compute_feature_metadata():
    X_train, _, y_train, _ = get_stratified_split()
    selected_15, _ = select_features_on_training_data(X_train, y_train, k=15)
    metadata = compute_feature_metadata(X_train, y_train, selected_15)

    assert len(metadata) == 15
    for feat in selected_15:
        m = metadata[feat]
        assert "min" in m and "max" in m
        assert "benign_median" in m and "malignant_median" in m
        assert m["min"] <= m["median"] <= m["max"]
