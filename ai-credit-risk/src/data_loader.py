"""
src/data_loader.py

Responsible for:
  * Generating a clearly-labeled synthetic demonstration dataset when a
    real dataset is not supplied.
  * Loading whichever CSV is being used.
  * Inspecting it (shape, dtypes, missing values, unique values, target
    candidates + distribution) and printing a readable report.
  * Auto-detecting (or accepting a manual override for) the target column,
    and normalising it to {0, 1} where 1 = "bad credit / default".
  * Identifying which protected/audit attributes are present, WITHOUT ever
    inventing them.

No hidden assumptions: if something can't be safely auto-detected, this
module raises a clear, actionable error instead of guessing.
"""

import os
import numpy as np
import pandas as pd

import config


class DatasetError(Exception):
    """Raised when the dataset cannot be safely loaded or interpreted."""


# ---------------------------------------------------------------------------
# Synthetic data generation (clearly labeled, NOT real financial data)
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(n_rows: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic, clearly-labeled demonstration credit dataset.
    This is NOT real financial data and must never be presented as such.
    The generating process bakes in a genuine (if simplified) statistical
    relationship between the inputs and default risk so that a model
    trained on it produces meaningful, non-random results.
    """
    rng = np.random.default_rng(random_state)

    income = rng.normal(45000, 18000, n_rows).clip(8000, 250000)
    loan_amount = rng.normal(12000, 8000, n_rows).clip(500, 80000)
    existing_debt = (rng.beta(2, 5, n_rows) * income * 0.6).clip(0, None)
    employment_duration_years = rng.exponential(4, n_rows).clip(0, 40)
    credit_history_years = rng.exponential(6, n_rows).clip(0, 45)
    credit_limit = (income * rng.uniform(0.1, 0.5, n_rows)).clip(500, None)
    credit_used = (credit_limit * rng.beta(2, 3, n_rows)).clip(0, credit_limit)
    late_payments_last_2y = rng.poisson(0.8, n_rows)
    num_open_accounts = rng.integers(1, 12, n_rows)
    checking_balance = rng.normal(2000, 3000, n_rows)
    savings_balance = rng.normal(4000, 6000, n_rows).clip(0, None)
    age = rng.integers(18, 75, n_rows)
    sex = rng.choice(["male", "female"], n_rows)
    purpose = rng.choice(
        ["car", "furniture", "electronics", "education", "business", "medical", "other"],
        n_rows,
    )

    monthly_income = income / 12
    monthly_debt = existing_debt / 12
    dti = monthly_debt / monthly_income.clip(min=1)
    lti = loan_amount / income.clip(min=1)
    utilization = credit_used / credit_limit.clip(min=1)

    # Simplified, transparent "true" risk-generating process (logit scale).
    logit = (
        -1.6
        + 2.6 * dti
        + 1.3 * lti
        + 1.8 * utilization
        + 0.35 * late_payments_last_2y
        - 0.05 * employment_duration_years
        - 0.04 * credit_history_years
        - 0.00002 * savings_balance
        + rng.normal(0, 0.6, n_rows)  # noise
    )
    prob_default = 1 / (1 + np.exp(-logit))
    target = (rng.uniform(0, 1, n_rows) < prob_default).astype(int)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "income": income.round(2),
        "employment_duration_years": employment_duration_years.round(2),
        "loan_amount": loan_amount.round(2),
        "purpose": purpose,
        "existing_debt": existing_debt.round(2),
        "credit_history_years": credit_history_years.round(2),
        "credit_limit": credit_limit.round(2),
        "credit_used": credit_used.round(2),
        "late_payments_last_2y": late_payments_last_2y,
        "num_open_accounts": num_open_accounts,
        "checking_balance": checking_balance.round(2),
        "savings_balance": savings_balance.round(2),
        "target": target,  # 1 = bad credit / default, 0 = good credit
    })
    return df


def ensure_dataset_available():
    """
    Ensures a usable CSV exists in data/. Preference order:
      1. data/german_credit.csv, if the user has supplied it.
      2. data/synthetic_credit_profiles.csv, generated automatically if
         missing.
    Returns (path, is_synthetic: bool).
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if os.path.exists(config.GERMAN_CREDIT_CSV):
        return config.GERMAN_CREDIT_CSV, False

    if not os.path.exists(config.SYNTHETIC_CSV):
        df = generate_synthetic_dataset()
        df.to_csv(config.SYNTHETIC_CSV, index=False)
        print(
            "[data_loader] NOTE: data/german_credit.csv was not found.\n"
            "[data_loader] Generated a SYNTHETIC, clearly-labeled demo "
            f"dataset instead: {config.SYNTHETIC_CSV}\n"
            "[data_loader] This is NOT real financial data. To use a real "
            "dataset, place a CSV at data/german_credit.csv and re-run "
            "training."
        )
    return config.SYNTHETIC_CSV, True


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------
def inspect_dataset(df: pd.DataFrame) -> dict:
    """Builds a structured inspection report and prints a readable summary."""
    report = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "unique_values": {c: int(df[c].nunique()) for c in df.columns},
        "target_candidates": [c for c in config.TARGET_CANDIDATES if c in df.columns]
        or [c for c in df.columns if c.lower() in config.TARGET_CANDIDATES],
    }

    print("=" * 70)
    print("DATASET INSPECTION REPORT")
    print("=" * 70)
    print(f"Shape: {report['shape'][0]} rows x {report['shape'][1]} columns")
    print(f"Columns: {report['columns']}")
    print("Missing values per column:")
    for c, n in report["missing_values"].items():
        if n:
            print(f"  - {c}: {n}")
    if not any(report["missing_values"].values()):
        print("  (none)")
    print(f"Target candidates found: {report['target_candidates']}")
    print("=" * 70)
    return report


# ---------------------------------------------------------------------------
# Target detection & normalisation
# ---------------------------------------------------------------------------
def detect_target_column(df: pd.DataFrame) -> str:
    if config.MANUAL_TARGET_COLUMN:
        if config.MANUAL_TARGET_COLUMN not in df.columns:
            raise DatasetError(
                f"MANUAL_TARGET_COLUMN='{config.MANUAL_TARGET_COLUMN}' set in "
                f"config.py but that column does not exist in the dataset. "
                f"Available columns: {list(df.columns)}"
            )
        return config.MANUAL_TARGET_COLUMN

    lower_map = {c.lower(): c for c in df.columns}
    for candidate in config.TARGET_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]

    raise DatasetError(
        "Could not automatically detect a target column. Looked for any "
        f"of {config.TARGET_CANDIDATES} (case-insensitive) in columns "
        f"{list(df.columns)}. Fix this by setting MANUAL_TARGET_COLUMN in "
        "config.py to the correct column name."
    )


def normalise_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Converts the target column to {0,1} where 1 = bad credit / default.
    Documents the exact mapping applied. Raises DatasetError if the target
    has more than 2 distinct non-null values (unsupported automatically).
    """
    series = df[target_col]
    unique_vals = sorted(series.dropna().unique().tolist(), key=lambda v: str(v))

    if series.dtype == bool:
        mapped = series.astype(int)
        mapping_doc = "boolean True/False -> 1/0"

    elif set(str(v).lower() for v in unique_vals) <= {"0", "1"}:
        mapped = series.astype(int)
        mapping_doc = "already {0,1} -> used as-is (1 = bad credit/default)"

    elif set(str(v).lower() for v in unique_vals) <= {"1", "2"}:
        # Classic UCI German Credit encoding: 1 = Good, 2 = Bad
        mapped = series.astype(int).map({1: 0, 2: 1})
        mapping_doc = "1=Good->0, 2=Bad->1 (standard German Credit encoding)"

    elif set(str(v).lower() for v in unique_vals) <= {"good", "bad"}:
        mapped = series.astype(str).str.lower().map({"good": 0, "bad": 1})
        mapping_doc = "'good'->0, 'bad'->1"

    elif set(str(v).lower() for v in unique_vals) <= {"yes", "no"}:
        mapped = series.astype(str).str.lower().map({"no": 0, "yes": 1})
        mapping_doc = "'no'->0, 'yes'(default)->1"

    elif len(unique_vals) == 2:
        # Generic binary fallback: sort values, treat the "larger looking"
        # one as the positive (bad credit) class only when values are
        # numeric; otherwise this is ambiguous and must be reported.
        try:
            numeric_vals = sorted(float(v) for v in unique_vals)
            mapped = series.astype(float).map({numeric_vals[0]: 0, numeric_vals[1]: 1})
            mapping_doc = f"{numeric_vals[0]}->0 (good), {numeric_vals[1]}->1 (bad) [inferred numeric ordering]"
        except (ValueError, TypeError):
            raise DatasetError(
                f"Target column '{target_col}' has 2 unique non-numeric "
                f"values {unique_vals} that don't match any known mapping "
                "(good/bad, yes/no, 0/1, 1/2). Automatic mapping is unsafe. "
                "Please recode the target column manually before training."
            )
    else:
        raise DatasetError(
            f"Target column '{target_col}' has {len(unique_vals)} unique "
            f"values {unique_vals}. This platform supports binary "
            "(good/bad credit) targets only. Automatic mapping is unsafe."
        )

    if mapped.isnull().any():
        raise DatasetError(
            f"Target mapping for column '{target_col}' produced null "
            "values for some rows - the mapping did not cover every "
            "observed value. Please check the target column contents."
        )

    print(f"[data_loader] Target column: '{target_col}'")
    print(f"[data_loader] Target mapping applied: {mapping_doc}")
    df = df.copy()
    df[target_col] = mapped.astype(int)
    print(f"[data_loader] Target distribution:\n{df[target_col].value_counts().to_string()}")
    return df


def detect_protected_attributes(df: pd.DataFrame) -> list:
    lower_map = {c.lower(): c for c in df.columns}
    found = [lower_map[c] for c in config.PROTECTED_ATTRIBUTE_CANDIDATES if c in lower_map]
    return found


def load_dataset():
    """
    Full load pipeline: ensure a CSV exists, load it, inspect it, detect
    and normalise the target, detect protected attributes.

    Returns dict with: df, target_col, protected_attrs, is_synthetic, path
    """
    path, is_synthetic = ensure_dataset_available()
    df = pd.read_csv(path)
    if df.empty:
        raise DatasetError(f"Dataset at {path} loaded but contains 0 rows.")

    inspect_dataset(df)
    target_col = detect_target_column(df)
    df = normalise_target(df, target_col)
    protected = detect_protected_attributes(df)
    if protected:
        print(f"[data_loader] Protected/audit attributes detected: {protected} "
              "(excluded from model features, retained only for fairness audit)")
    else:
        print("[data_loader] No protected/audit attributes detected in this dataset.")

    return {
        "df": df,
        "target_col": target_col,
        "protected_attrs": protected,
        "is_synthetic": is_synthetic,
        "path": path,
    }
