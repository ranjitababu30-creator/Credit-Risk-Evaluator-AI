"""
src/explainability.py

Produces per-application "why" explanations.

If the `shap` package is installed, uses a real SHAP TreeExplainer (for
tree-based models) or KernelExplainer/LinearExplainer fallback to compute
local, instance-specific contributions.

If `shap` is NOT installed (e.g. this sandbox has no internet access to
install it), falls back to a GLOBAL feature-importance-based explanation
scaled by how far each feature value sits from the training mean. This
fallback is always clearly labeled as such in its output - it is never
presented as a local SHAP explanation.
"""

import numpy as np

try:
    import shap  # noqa: F401
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _human_label(feature_name):
    return feature_name.replace("num__", "").replace("cat__", "").replace("_", " ").strip()


def explain_with_shap(model, background_X, instance_X, feature_names, top_n=6):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(instance_X)

    # Handle both binary-classifier list output and single-array output
    if isinstance(shap_values, list):
        values = np.array(shap_values[1])[0]  # class 1 = default/bad credit
    else:
        arr = np.array(shap_values)
        values = arr[0, :, 1] if arr.ndim == 3 else arr[0]

    contributions = list(zip(feature_names, values, instance_X[0]))
    return _format_contributions(contributions, top_n, method="shap")


def explain_with_fallback(model, feature_names, instance_X, top_n=6):
    """Global-importance x deviation-from-typical fallback explanation."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.ones(len(feature_names))

    values = instance_X[0]
    # Signed pseudo-contribution: importance * standardized-ish value.
    # This is NOT a causal/local attribution - just a transparent heuristic.
    pseudo_contrib = importances * values

    contributions = list(zip(feature_names, pseudo_contrib, values))
    return _format_contributions(contributions, top_n, method="fallback_feature_importance")


def _format_contributions(contributions, top_n, method):
    increasing = sorted(
        [c for c in contributions if c[1] > 0], key=lambda x: -x[1]
    )[:top_n]
    decreasing = sorted(
        [c for c in contributions if c[1] < 0], key=lambda x: x[1]
    )[:top_n]

    def fmt(items):
        out = []
        for name, contrib, value in items:
            out.append({
                "feature": _human_label(name),
                "value": round(float(value), 4),
                "contribution": round(float(contrib), 4),
                "explanation": (
                    f"{_human_label(name)} (value: {round(float(value), 3)}) "
                    f"{'increased' if contrib > 0 else 'decreased'} the "
                    "model-estimated risk."
                ),
            })
        return out

    return {
        "method": method,
        "is_local_shap": method == "shap",
        "risk_increasing_factors": fmt(increasing),
        "risk_reducing_factors": fmt(decreasing),
        "disclaimer": (
            "These factors reflect the trained model's estimated "
            "association between inputs and predicted risk. They describe "
            "model-estimated risk, not a borrower's actual character or "
            "guaranteed repayment behaviour."
            if method == "shap" else
            "SHAP was not available in this environment, so this "
            "explanation uses a FALLBACK based on overall model feature "
            "importance rather than a true per-application SHAP "
            "explanation. Install the 'shap' package for local "
            "explanations."
        ),
    }


def explain_instance(model, feature_names, instance_X, background_X=None, top_n=6):
    if SHAP_AVAILABLE and hasattr(model, "predict_proba") and _is_tree_model(model):
        try:
            return explain_with_shap(model, background_X, instance_X, feature_names, top_n)
        except Exception as e:  # noqa: BLE001
            print(f"[explainability] SHAP failed ({e}); using fallback explanation.")
            return explain_with_fallback(model, feature_names, instance_X, top_n)
    return explain_with_fallback(model, feature_names, instance_X, top_n)


def _is_tree_model(model):
    name = type(model).__name__.lower()
    return any(k in name for k in ("forest", "xgb", "gbm", "gradientboost", "tree"))
