"""
src/anomaly_detection.py

Isolation Forest-based anomaly detection. Trained on preprocessed
training features. An "anomalous" application is treated purely as a
MONITORING / REVIEW signal - never as an automatic fraud accusation.
"""

from sklearn.ensemble import IsolationForest
import numpy as np

import config


def train_anomaly_model(X_train_processed):
    model = IsolationForest(
        contamination=config.ANOMALY_CONTAMINATION,
        random_state=config.RANDOM_STATE,
        n_estimators=200,
    )
    model.fit(X_train_processed)
    return model


def score_application(model, X_processed_row):
    """
    Returns (is_anomalous: bool, anomaly_score: float).
    IsolationForest.decision_function: higher = more normal, lower/negative
    = more anomalous. We invert and rescale to an intuitive 0-1 "unusualness"
    score where higher = more unusual.
    """
    raw_pred = model.predict(X_processed_row)  # -1 = anomaly, 1 = normal
    decision = model.decision_function(X_processed_row)  # roughly -0.5..0.5
    unusualness = np.clip(0.5 - decision, 0, 1)
    is_anomalous = bool(raw_pred[0] == -1)
    return is_anomalous, float(unusualness[0])
