"""
src/calibration.py

Wraps scikit-learn's CalibratedClassifierCV so raw model scores become
genuine probability estimates, and provides calibration-quality metrics
(Brier score, and a simple Expected Calibration Error) used both during
training (model selection) and on the admin monitoring page.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def calibrate_model(fitted_estimator, X_train, y_train, method="isotonic", cv=5):
    """
    Wraps an ALREADY-FITTED estimator's raw scores in a calibrator.
    Uses cv='prefit' is deprecated in newer sklearn, so instead we fit a
    fresh CalibratedClassifierCV wrapping an unfitted clone via cross
    validation on the training data only (never on test data).
    """
    from sklearn.base import clone
    calibrated = CalibratedClassifierCV(clone(fitted_estimator), method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated


def evaluate_calibration(y_true, y_prob, n_bins=10):
    """Returns Brier score and Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    brier = brier_score_loss(y_true, y_prob)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    quality = "Good" if ece < 0.05 else ("Review" if ece < 0.12 else "Poor")
    return {"brier_score": float(brier), "ece": float(ece), "quality": quality}
