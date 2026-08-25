"""
src/fairness.py

Responsible AI fairness/bias audit.

Sensitive attributes are NEVER used as model features (see
preprocessing.py / train_model.py, which explicitly exclude them). This
module uses them ONLY to audit group-level outcomes of a model that was
trained without them.

If no protected/audit attribute is present in the dataset, this module
reports that fairness auditing is unavailable rather than fabricating
one.
"""

import numpy as np
import pandas as pd

import config


def audit_fairness(df_with_protected, y_true_col, y_pred_col, protected_col):
    """
    df_with_protected must contain: the true label column, the predicted
    label column (1 = predicted bad credit/default, i.e. NOT approved),
    and the protected attribute column.
    """
    groups = df_with_protected[protected_col].dropna().unique().tolist()
    if len(groups) < 2:
        return {
            "available": False,
            "reason": f"Only one group value found for '{protected_col}'; "
                      "at least two groups are required for a comparison.",
        }

    metrics = {}
    for g in groups:
        subset = df_with_protected[df_with_protected[protected_col] == g]
        y_true = subset[y_true_col]
        y_pred = subset[y_pred_col]

        n = len(subset)
        selection_rate = float((y_pred == 0).mean())  # 0 = approved/good

        positives = subset[y_true == 1]  # actual bad credit
        negatives = subset[y_true == 0]  # actual good credit

        tpr = float((positives[y_pred_col] == 1).mean()) if len(positives) else None
        fpr = float((negatives[y_pred_col] == 1).mean()) if len(negatives) else None
        fnr = float((positives[y_pred_col] == 0).mean()) if len(positives) else None

        metrics[str(g)] = {
            "n": int(n),
            "approval_rate": round(selection_rate, 4),
            "true_positive_rate": round(tpr, 4) if tpr is not None else None,
            "false_positive_rate": round(fpr, 4) if fpr is not None else None,
            "false_negative_rate": round(fnr, 4) if fnr is not None else None,
        }

    approval_rates = [m["approval_rate"] for m in metrics.values()]
    dpd = max(approval_rates) - min(approval_rates)

    tprs = [m["true_positive_rate"] for m in metrics.values() if m["true_positive_rate"] is not None]
    eod = (max(tprs) - min(tprs)) if len(tprs) >= 2 else None

    if dpd >= config.FAIRNESS_DEMOGRAPHIC_PARITY_REVIEW or (
            eod is not None and eod >= config.FAIRNESS_EQUAL_OPPORTUNITY_REVIEW):
        status = "REVIEW"
    elif dpd >= config.FAIRNESS_DEMOGRAPHIC_PARITY_WARN or (
            eod is not None and eod >= config.FAIRNESS_EQUAL_OPPORTUNITY_WARN):
        status = "WARNING"
    else:
        status = "NORMAL"

    return {
        "available": True,
        "protected_attribute": protected_col,
        "group_metrics": metrics,
        "demographic_parity_difference": round(float(dpd), 4),
        "equal_opportunity_difference": round(float(eod), 4) if eod is not None else None,
        "status": status,
        "disclaimer": config.FAIRNESS_DISCLAIMER,
    }


def audit_all_protected_attributes(df_with_protected, y_true_col, y_pred_col, protected_cols):
    if not protected_cols:
        return {
            "available": False,
            "reason": "Fairness audit unavailable for this dataset because "
                      "appropriate audit attributes are not provided.",
        }
    results = {}
    for col in protected_cols:
        if col in df_with_protected.columns:
            results[col] = audit_fairness(df_with_protected, y_true_col, y_pred_col, col)
    return {"available": True, "audits": results}
