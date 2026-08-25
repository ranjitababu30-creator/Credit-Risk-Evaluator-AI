import pytest

from src.prediction import predict_application, ArtifactBundle
from src import scoring


def sample_input():
    bundle = ArtifactBundle.get()
    # Build a plausible input using the first numeric/categorical columns
    values = {}
    for col in bundle.numeric_columns:
        values[col] = 1000
    for col in bundle.categorical_columns:
        values[col] = "other"
    return values


def test_predict_application_returns_expected_shape():
    result = predict_application(sample_input(), explain=False)
    assert 0 <= result["risk_score"] <= 1000
    assert 0 <= result["probability_of_default"] <= 1
    assert result["risk_category"] in [b[2] for b in __import__("config").RISK_BANDS]
    assert result["recommendation"] in ("APPROVE", "REVIEW", "REJECT")
    assert "anomaly" in result


def test_predict_application_with_explanation():
    result = predict_application(sample_input(), explain=True)
    assert "explanation" in result
    assert "risk_increasing_factors" in result["explanation"]
    assert "risk_reducing_factors" in result["explanation"]


def test_predict_application_handles_missing_fields_gracefully():
    result = predict_application({}, explain=False)
    assert "risk_score" in result


def test_predict_application_ignores_unknown_fields():
    inputs = sample_input()
    inputs["this_field_does_not_exist"] = 42
    result = predict_application(inputs, explain=False)
    assert any("Ignored unrecognised fields" in w for w in result["warnings"])


def test_scoring_probability_to_score_bounds():
    assert scoring.probability_to_score(0.0) == 1000
    assert scoring.probability_to_score(1.0) == 0
    assert scoring.probability_to_score(0.5) == 500


def test_scoring_score_to_band():
    assert scoring.score_to_band(900) == "LOW"
    assert scoring.score_to_band(100) == "VERY HIGH"


def test_decision_engine_thresholds():
    d = scoring.decide(800, 0.1, is_anomalous=False)
    assert d["recommendation"] == "APPROVE"
    d = scoring.decide(200, 0.9, is_anomalous=False)
    assert d["recommendation"] == "REJECT"
    d = scoring.decide(600, 0.4, is_anomalous=False)
    assert d["recommendation"] == "REVIEW"


def test_decision_engine_anomaly_forces_review():
    d = scoring.decide(900, 0.05, is_anomalous=True)
    assert d["recommendation"] == "REVIEW"
    assert d["overridden_by_review_rules"] is True
