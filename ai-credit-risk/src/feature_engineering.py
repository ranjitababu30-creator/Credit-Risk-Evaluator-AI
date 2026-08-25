"""
src/feature_engineering.py

Creates derived financial features ONLY when their required source
columns exist. Every skipped feature is reported explicitly - nothing is
silently invented.

Recognised source columns (case-insensitive, first match wins):
  income                -> income, annual_income, monthly_income
  loan_amount           -> loan_amount, credit_amount
  existing_debt         -> existing_debt, debt, monthly_debt
  credit_limit          -> credit_limit
  credit_used           -> credit_used, credit_balance
  late_payments         -> late_payments_last_2y, late_payments, num_late_payments
  employment_duration   -> employment_duration_years, employment_duration
  credit_history_years  -> credit_history_years, credit_history_length
  successful_repayments -> successful_repayments
  total_repayments      -> total_repayments
"""

ALIASES = {
    "income": ["income", "annual_income", "monthly_income"],
    "loan_amount": ["loan_amount", "credit_amount"],
    "existing_debt": ["existing_debt", "debt", "monthly_debt"],
    "credit_limit": ["credit_limit"],
    "credit_used": ["credit_used", "credit_balance"],
    "late_payments": ["late_payments_last_2y", "late_payments", "num_late_payments"],
    "employment_duration": ["employment_duration_years", "employment_duration"],
    "credit_history_years": ["credit_history_years", "credit_history_length"],
    "successful_repayments": ["successful_repayments"],
    "total_repayments": ["total_repayments"],
}


def _find_column(df, logical_name):
    lower_map = {c.lower(): c for c in df.columns}
    for alias in ALIASES[logical_name]:
        if alias in lower_map:
            return lower_map[alias]
    return None


def engineer_features(df):
    """
    Returns (df_with_new_features, report) where report lists which
    derived features were created and which were skipped (and why).
    """
    df = df.copy()
    report = {"created": [], "skipped": []}

    income_col = _find_column(df, "income")
    loan_col = _find_column(df, "loan_amount")
    debt_col = _find_column(df, "existing_debt")
    limit_col = _find_column(df, "credit_limit")
    used_col = _find_column(df, "credit_used")
    success_col = _find_column(df, "successful_repayments")
    total_col = _find_column(df, "total_repayments")

    # Debt-to-Income Ratio
    if income_col and debt_col:
        monthly_income = df[income_col].clip(lower=1) / 12.0
        df["debt_to_income_ratio"] = (df[debt_col] / 12.0) / monthly_income
        report["created"].append("debt_to_income_ratio")
    else:
        report["skipped"].append(
            ("debt_to_income_ratio", "requires income and existing_debt columns")
        )

    # Loan-to-Income Ratio
    if income_col and loan_col:
        df["loan_to_income_ratio"] = df[loan_col] / df[income_col].clip(lower=1)
        report["created"].append("loan_to_income_ratio")
    else:
        report["skipped"].append(
            ("loan_to_income_ratio", "requires income and loan_amount columns")
        )

    # Credit utilization
    if limit_col and used_col:
        df["credit_utilization"] = (df[used_col] / df[limit_col].clip(lower=1)).clip(0, 3)
        report["created"].append("credit_utilization")
    else:
        report["skipped"].append(
            ("credit_utilization", "requires credit_limit and credit_used columns")
        )

    # Repayment ratio
    if success_col and total_col:
        df["repayment_ratio"] = df[success_col] / df[total_col].clip(lower=1)
        report["created"].append("repayment_ratio")
    else:
        report["skipped"].append(
            ("repayment_ratio", "requires successful_repayments and total_repayments columns")
        )

    print("[feature_engineering] Created:", report["created"] or "(none)")
    for name, reason in report["skipped"]:
        print(f"[feature_engineering] Skipped '{name}': {reason}")

    return df, report
