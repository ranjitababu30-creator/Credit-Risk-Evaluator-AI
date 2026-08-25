"""
config.py
Central configuration for the AI Credit Risk & Responsible Lending Platform.

Everything a demo-runner might want to tweak (thresholds, paths, policy
cut-offs) lives here so it is not buried inside application logic.

IMPORTANT: All risk bands, decision thresholds and stability thresholds
below are PROTOTYPE / DEMO values chosen for illustration. They are not
validated credit policy and must not be treated as such.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "credit.db")

GERMAN_CREDIT_CSV = os.path.join(DATA_DIR, "german_credit.csv")
SYNTHETIC_CSV = os.path.join(DATA_DIR, "synthetic_credit_profiles.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "credit_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
ANOMALY_MODEL_PATH = os.path.join(MODELS_DIR, "anomaly_model.pkl")
CALIBRATOR_PATH = os.path.join(MODELS_DIR, "calibrator.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
REFERENCE_STATS_PATH = os.path.join(MODELS_DIR, "reference_stats.json")

# ---------------------------------------------------------------------------
# Target column detection
# ---------------------------------------------------------------------------
# Candidate names the loader will search for automatically, in priority
# order. If none are found, the loader reports the problem instead of
# guessing.
TARGET_CANDIDATES = [
    "target", "class", "credit_risk", "risk", "default",
    "credit", "loan_default", "creditability", "y",
]

# Manual override: set this to a column name to force it as the target,
# bypassing auto-detection entirely. Leave as None to auto-detect.
MANUAL_TARGET_COLUMN = None

# Mapping rules applied when the target is categorical/ordinal instead of
# already being a clean {0,1} "1 = default / bad credit" column.
# Value on the LEFT is treated as "bad credit / default" -> 1
TARGET_VALUE_MAPS = [
    {"bad": 1, "good": 0},
    {"1": 1, "2": 0},          # classic German Credit encoding (1=good,2=bad) - handled specially in loader
    {"yes": 1, "no": 0},
    {"default": 1, "non-default": 0},
]

# ---------------------------------------------------------------------------
# Sensitive / protected attributes
# ---------------------------------------------------------------------------
# These columns, if present in the dataset, are NEVER used as model
# features. They may optionally be retained purely for fairness auditing.
PROTECTED_ATTRIBUTE_CANDIDATES = [
    "sex", "gender", "age_group", "foreign_worker", "marital_status",
    "personal_status", "ethnicity", "race",
]

# ---------------------------------------------------------------------------
# Risk score configuration
# ---------------------------------------------------------------------------
RISK_SCORE_MIN = 0
RISK_SCORE_MAX = 1000

# (lower_bound_inclusive, upper_bound_inclusive, label)
RISK_BANDS = [
    (800, 1000, "LOW"),
    (650, 799, "MODERATE-LOW"),
    (500, 649, "MEDIUM"),
    (350, 499, "HIGH"),
    (0, 349, "VERY HIGH"),
]

# ---------------------------------------------------------------------------
# Decision engine thresholds (transparent, configurable policy)
# ---------------------------------------------------------------------------
APPROVE_SCORE_THRESHOLD = 700   # score >= this -> APPROVE (subject to overrides)
REJECT_SCORE_THRESHOLD = 500    # score < this -> REJECT (subject to overrides)
# Anything in between -> REVIEW

# Overrides: force REVIEW regardless of score when triggered
ANOMALY_FORCES_REVIEW = True
LOW_MODEL_CONFIDENCE_FORCES_REVIEW = True
# "Confidence" here = how far predicted probability sits from the model's
# decision boundary (0.5). Below this margin we treat the model as unsure.
MIN_CONFIDENCE_MARGIN = 0.05

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
ANOMALY_CONTAMINATION = 0.05  # expected proportion of unusual applications

# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------
FAIRNESS_DEMOGRAPHIC_PARITY_WARN = 0.10
FAIRNESS_DEMOGRAPHIC_PARITY_REVIEW = 0.20
FAIRNESS_EQUAL_OPPORTUNITY_WARN = 0.10
FAIRNESS_EQUAL_OPPORTUNITY_REVIEW = 0.20
FAIRNESS_DISCLAIMER = (
    "Fairness metrics are monitoring indicators and do not constitute "
    "legal or regulatory compliance."
)

# ---------------------------------------------------------------------------
# Feature stability / drift (Population Stability Index)
# ---------------------------------------------------------------------------
PSI_STABLE_MAX = 0.10          # PSI < 0.10  -> Stable
PSI_WARNING_MAX = 0.25         # 0.10-0.25   -> Warning; >0.25 -> Unstable
PSI_BUCKETS = 10

# ---------------------------------------------------------------------------
# Misc / disclaimers
# ---------------------------------------------------------------------------
SCENARIO_DISCLAIMER = (
    "Scenario results are model simulations and are not guarantees of "
    "future credit decisions."
)
GENERAL_DISCLAIMER = (
    "This system is a decision-support prototype for demonstration "
    "purposes only. It does not guarantee whether any individual will "
    "repay a loan, and it is not a real banking approval system."
)

MODEL_VERSION = "1.0.0-prototype"
RANDOM_STATE = 42
TEST_SIZE = 0.2
