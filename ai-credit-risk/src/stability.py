"""
src/stability.py

Feature distribution / drift monitoring using the Population Stability
Index (PSI). Compares a "current" set of feature values against the
reference distribution captured from the training data.

PSI is a monitoring signal only - it does not by itself mean the model
is invalid.
"""

import numpy as np
import pandas as pd

import config


def build_reference_buckets(train_series: pd.Series, n_buckets=config.PSI_BUCKETS):
    """Builds quantile-based bucket edges from the training distribution."""
    series = train_series.dropna().astype(float)
    if series.nunique() < 2:
        return None
    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = np.unique(series.quantile(quantiles).values)
    if len(edges) < 3:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.tolist()


def compute_reference_distribution(train_series: pd.Series):
    edges = build_reference_buckets(train_series)
    if edges is None:
        return None
    counts, _ = np.histogram(train_series.dropna().astype(float), bins=edges)
    props = counts / max(counts.sum(), 1)
    return {"edges": edges, "proportions": props.tolist()}


def compute_psi(reference: dict, current_series: pd.Series) -> float:
    edges = np.array(reference["edges"])
    ref_props = np.array(reference["proportions"])
    counts, _ = np.histogram(current_series.dropna().astype(float), bins=edges)
    cur_props = counts / max(counts.sum(), 1)

    ref_props = np.clip(ref_props, 1e-4, None)
    cur_props = np.clip(cur_props, 1e-4, None)

    psi = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
    return psi


def psi_status(psi_value: float) -> str:
    if psi_value < config.PSI_STABLE_MAX:
        return "Stable"
    if psi_value < config.PSI_WARNING_MAX:
        return "Warning"
    return "Unstable"


def audit_stability(reference_stats: dict, current_df: pd.DataFrame) -> dict:
    """reference_stats: {column_name: {"edges": [...], "proportions": [...]}}"""
    results = {}
    for col, ref in reference_stats.items():
        if col not in current_df.columns:
            continue
        psi = compute_psi(ref, current_df[col])
        results[col] = {"psi": round(psi, 4), "status": psi_status(psi)}

    if not results:
        return {"available": False, "reason": "No reference distributions available for comparison."}

    overall = "Stable"
    if any(r["status"] == "Unstable" for r in results.values()):
        overall = "Unstable"
    elif any(r["status"] == "Warning" for r in results.values()):
        overall = "Warning"

    return {"available": True, "overall_status": overall, "features": results}
