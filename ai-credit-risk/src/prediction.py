"""
src/prediction.py

Loads the persisted model/preprocessor/anomaly-detector artifacts once
and exposes `predict_application(input_dict)` which runs the SAME
preprocessing + SAME trained model used everywhere else in the platform
(assessment page, API, and What-If scenario simulator all call this).
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

import config
from src import scoring, anomaly_detection, explainability


class ModelNotTrainedError(Exception):
    pass


class ArtifactBundle:
    """Lazily-loaded, cached set of trained artifacts."""
    _instance = None

    def __init__(self):
        required = [config.MODEL_PATH, config.PREPROCESSOR_PATH,
                    config.ANOMALY_MODEL_PATH, config.METADATA_PATH]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise ModelNotTrainedError(
                "Model has not been trained yet. Run "
                "'python -m src.train_model' first. Missing files: "
                f"{missing}"
            )

        self.model = joblib.load(config.MODEL_PATH)
        self.preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        self.anomaly_model = joblib.load(config.ANOMALY_MODEL_PATH)

        raw_model_path = os.path.join(config.MODELS_DIR, "raw_model.pkl")
        self.raw_model = joblib.load(raw_model_path) if os.path.exists(raw_model_path) else self.model

        with open(config.METADATA_PATH) as f:
            self.metadata = json.load(f)

        self.feature_columns = self.metadata["feature_columns"]
        self.numeric_columns = self.metadata["numeric_columns"]
        self.categorical_columns = self.metadata["categorical_columns"]

        from src.preprocessing import get_output_feature_names
        self.feature_names_out = get_output_feature_names(
            self.preprocessor, self.numeric_columns, self.categorical_columns
        )

        # Small background sample for SHAP, taken from the holdout snapshot
        # (already-known, non-secret demo data) if available.
        self.background = None
        holdout_path = os.path.join(config.MODELS_DIR, "holdout_snapshot.csv")
        if os.path.exists(holdout_path):
            try:
                holdout_df = pd.read_csv(holdout_path)
                cols = [c for c in self.feature_columns if c in holdout_df.columns]
                sample = holdout_df[cols].head(50)
                self.background = self.preprocessor.transform(sample)
                if hasattr(self.background, "toarray"):
                    self.background = self.background.toarray()
            except Exception:  # noqa: BLE001
                self.background = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reload(cls):
        cls._instance = None
        return cls.get()


def _build_input_row(input_dict, feature_columns):
    """
    Builds a single-row DataFrame matching the training feature columns.
    Missing optional fields are filled with NaN (the preprocessing
    pipeline's imputers handle them safely) rather than crashing.
    """
    row = {}
    for col in feature_columns:
        row[col] = input_dict.get(col, np.nan)
    return pd.DataFrame([row], columns=feature_columns)


def predict_application(input_dict: dict, explain: bool = True) -> dict:
    """
    Runs the full pipeline for ONE application:
      preprocessing -> calibrated probability -> risk score -> band ->
      anomaly check -> decision -> (optional) explanation.
    Never raises for missing/invalid individual fields - reports problems
    in the returned dict's "warnings" list instead where safely possible.
    """
    bundle = ArtifactBundle.get()
    warnings = []

    unknown_fields = [k for k in input_dict.keys() if k not in bundle.feature_columns]
    if unknown_fields:
        warnings.append(f"Ignored unrecognised fields not used by the model: {unknown_fields}")

    row_df = _build_input_row(input_dict, bundle.feature_columns)

    try:
        processed = bundle.preprocessor.transform(row_df)
        if hasattr(processed, "toarray"):
            processed = processed.toarray()
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Could not process the supplied input: {e}") from e

    prob_default = float(bundle.model.predict_proba(processed)[0, 1])
    risk_score = scoring.probability_to_score(prob_default)
    risk_band = scoring.score_to_band(risk_score)

    is_anomalous, anomaly_score = anomaly_detection.score_application(bundle.anomaly_model, processed)

    decision = scoring.decide(risk_score, prob_default, is_anomalous)

    result = {
        "probability_of_default": round(prob_default, 4),
        "risk_score": risk_score,
        "risk_category": risk_band,
        "recommendation": decision["recommendation"],
        "decision_reasons": decision["reasons"],
        "anomaly": {
            "is_anomalous": is_anomalous,
            "anomaly_score": round(anomaly_score, 4),
            "label": "UNUSUAL APPLICATION" if is_anomalous else "NORMAL",
        },
        "model_version": bundle.metadata.get("model_version"),
        "selected_model": bundle.metadata.get("selected_model"),
        "warnings": warnings,
        "disclaimer": config.GENERAL_DISCLAIMER,
    }

    if explain:
        result["explanation"] = explainability.explain_instance(
            bundle.raw_model, bundle.feature_names_out, processed, bundle.background
        )

    return result
