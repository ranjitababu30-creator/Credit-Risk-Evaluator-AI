"""
src/scoring.py

Converts a calibrated probability of default into a 0-1000 demonstration
risk score, and looks up the configured (prototype) risk band and
decision recommendation. Also contains the transparent decision engine.
"""

import config


def probability_to_score(prob_default: float) -> int:
    score = config.RISK_SCORE_MAX * (1 - prob_default)
    score = max(config.RISK_SCORE_MIN, min(config.RISK_SCORE_MAX, score))
    return int(round(score))


def score_to_band(score: int) -> str:
    for lo, hi, label in config.RISK_BANDS:
        if lo <= score <= hi:
            return label
    return "UNKNOWN"


def decide(score: int, prob_default: float, is_anomalous: bool) -> dict:
    """
    Transparent, configurable policy engine.
      score >= APPROVE_SCORE_THRESHOLD -> APPROVE
      score <  REJECT_SCORE_THRESHOLD  -> REJECT
      otherwise                        -> REVIEW
    Overrides (anomaly / low model confidence) can force REVIEW even when
    the score alone would say APPROVE or REJECT.
    """
    reasons = []

    if score >= config.APPROVE_SCORE_THRESHOLD:
        recommendation = "APPROVE"
    elif score < config.REJECT_SCORE_THRESHOLD:
        recommendation = "REJECT"
    else:
        recommendation = "REVIEW"
        reasons.append(f"Risk score ({score}) falls in the REVIEW band "
                        f"({config.REJECT_SCORE_THRESHOLD}-{config.APPROVE_SCORE_THRESHOLD - 1}).")

    confidence_margin = abs(prob_default - 0.5)
    low_confidence = confidence_margin < config.MIN_CONFIDENCE_MARGIN

    overridden = False
    if config.ANOMALY_FORCES_REVIEW and is_anomalous and recommendation != "REVIEW":
        overridden = True
        reasons.append(
            "This application was flagged as statistically UNUSUAL compared "
            "to the training population, so it is routed to manual REVIEW "
            "rather than being auto-decided. This is a monitoring signal, "
            "not an accusation of wrongdoing."
        )
        recommendation = "REVIEW"

    if (config.LOW_MODEL_CONFIDENCE_FORCES_REVIEW and low_confidence
            and recommendation != "REVIEW"):
        overridden = True
        reasons.append(
            "The model's estimated probability of default is close to its "
            "decision boundary, indicating low confidence, so this "
            "application is routed to manual REVIEW."
        )
        recommendation = "REVIEW"

    if not reasons:
        reasons.append(f"Risk score ({score}) meets the '{recommendation}' policy threshold.")

    return {
        "recommendation": recommendation,
        "reasons": reasons,
        "overridden_by_review_rules": overridden,
        "low_confidence": bool(low_confidence),
    }
