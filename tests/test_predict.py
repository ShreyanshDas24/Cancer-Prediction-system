"""Unit tests for src.predict."""

import pytest
from src.predict import CancerPredictor


@pytest.fixture
def predictor():
    return CancerPredictor()


def test_predictor_initialization(predictor):
    assert len(predictor.selected_features) == 15
    assert predictor.pipeline is not None
    assert predictor.metadata is not None


def test_predict_benign_median(predictor):
    sample = {f: predictor.metadata[f]["benign_median"] for f in predictor.selected_features}
    res = predictor.predict_sample(sample)

    assert "predicted_class" in res
    assert "proba_malignant" in res
    assert "model_risk_category" in res
    assert res["predicted_class"] == "Benign"
    assert res["proba_malignant"] < 0.50
    assert res["model_risk_category"] in ["Very Low", "Low"]


def test_predict_malignant_median(predictor):
    sample = {f: predictor.metadata[f]["malignant_median"] for f in predictor.selected_features}
    res = predictor.predict_sample(sample)

    assert res["predicted_class"] == "Malignant"
    assert res["proba_malignant"] > 0.50
    assert res["model_risk_category"] in ["Elevated", "High"]


def test_predict_validation_missing_feature(predictor):
    sample = {f: predictor.metadata[f]["benign_median"] for f in predictor.selected_features}
    del sample[predictor.selected_features[0]]

    with pytest.raises(ValueError, match="Missing required features"):
        predictor.predict_sample(sample)


def test_predict_validation_non_numeric(predictor):
    sample = {f: predictor.metadata[f]["benign_median"] for f in predictor.selected_features}
    sample[predictor.selected_features[0]] = "invalid_string"

    with pytest.raises(ValueError, match="contains invalid or non-numeric values"):
        predictor.predict_sample(sample)
