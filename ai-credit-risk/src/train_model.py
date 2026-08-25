"""
src/train_model.py

Run with:  python -m src.train_model

End-to-end training pipeline:
  1. Load dataset (real if provided, else clearly-labeled synthetic)
  2. Validate / inspect
  3. Engineer safe derived features
  4. Stratified train/test split (fit preprocessing on TRAIN ONLY)
  5. Train Logistic Regression, Random Forest, XGBoost (or GradientBoosting
     fallback if xgboost is not installed)
  6. Evaluate all three on the held-out test set
  7. Calibrate probabilities for the best model
  8. Select the best model considering discrimination + calibration
  9. Train the Isolation Forest anomaly detector on training features
 10. Persist model, preprocessor, calibrator, anomaly detector, metadata,
     reference distributions (for drift), and a holdout snapshot (for the
     fairness/stability admin pages)
 12. Print a clear training summary
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

import config
from src import data_loader, feature_engineering, preprocessing, calibration, anomaly_detection, stability

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def build_candidate_models():
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=config.RANDOM_STATE, n_jobs=-1
        ),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            eval_metric="logloss", random_state=config.RANDOM_STATE,
            use_label_encoder=False,
        )
    else:
        print("[train_model] NOTE: xgboost is not installed in this "
              "environment. Falling back to sklearn GradientBoostingClassifier "
              "as the 'boosted trees' candidate. Install xgboost "
              "(see requirements.txt) for the real XGBoost model.")
        models["gradient_boosting_fallback"] = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=config.RANDOM_STATE
        )
    return models


def evaluate_model(y_true, y_pred, y_prob):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
    }


def select_best_model(results: dict) -> str:
    """
    Selects the best model NOT purely on accuracy. Combines ROC-AUC
    (discrimination) and Brier score (calibration quality) into one
    ranking score. Lower Brier is better, higher ROC-AUC is better.
    """
    best_name, best_rank = None, -np.inf
    for name, r in results.items():
        rank = r["metrics"]["roc_auc"] - r["calibration"]["brier_score"]
        if rank > best_rank:
            best_rank, best_name = rank, name
    return best_name


def main():
    start = time.time()
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    print("\nSTEP 1/9: Loading dataset...")
    loaded = data_loader.load_dataset()
    df, target_col, protected_attrs, is_synthetic = (
        loaded["df"], loaded["target_col"], loaded["protected_attrs"], loaded["is_synthetic"]
    )

    print("\nSTEP 2/9: Engineering derived features...")
    df, fe_report = feature_engineering.engineer_features(df)

    # Feature columns = everything except target and protected attributes
    # (protected attributes are retained only for the fairness audit, never
    # as model inputs) and obvious identifier columns.
    id_like = [c for c in df.columns if c.lower() in ("id", "applicant_id", "customer_id")]
    feature_columns = [c for c in df.columns
                        if c != target_col and c not in protected_attrs and c not in id_like]

    if len(feature_columns) == 0:
        print("ERROR: No usable feature columns remain after excluding the "
              "target and protected attributes. Aborting.")
        sys.exit(1)

    X = df[feature_columns]
    y = df[target_col]

    print(f"\nSTEP 3/9: Splitting data (stratified, test_size={config.TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    # Keep the raw (unprocessed) test rows + protected attrs for the
    # fairness/stability admin pages - fit nothing on this data.
    holdout_raw = df.loc[X_test.index].copy()

    print("Fitting preprocessing pipeline on TRAINING data only...")
    preprocessor, numeric_cols, categorical_cols = preprocessing.build_preprocessor(X, feature_columns)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()
    feature_names_out = preprocessing.get_output_feature_names(preprocessor, numeric_cols, categorical_cols)

    print("\nSTEP 4/9: Training candidate models (Logistic Regression, Random "
          "Forest, XGBoost)...")
    candidates = build_candidate_models()
    results = {}
    for name, model in candidates.items():
        print(f"  training {name}...")
        model.fit(X_train_processed, y_train)
        raw_prob = model.predict_proba(X_test_processed)[:, 1]

        print(f"  calibrating {name}...")
        calibrated = calibration.calibrate_model(model, X_train_processed, y_train)
        cal_prob = calibrated.predict_proba(X_test_processed)[:, 1]
        cal_pred = (cal_prob >= 0.5).astype(int)

        metrics = evaluate_model(y_test, cal_pred, cal_prob)
        cal_metrics = calibration.evaluate_calibration(y_test, cal_prob)

        results[name] = {
            "raw_model": model,
            "calibrated_model": calibrated,
            "metrics": metrics,
            "calibration": cal_metrics,
        }
        print(f"  {name}: {metrics} | calibration: {cal_metrics}")

    print("\nSTEP 5/9: Selecting best model (discrimination + calibration, "
          "not accuracy alone)...")
    best_name = select_best_model(results)
    best = results[best_name]
    print(f"  Selected model: {best_name}")

    print("\nSTEP 6/9: Training Isolation Forest anomaly detector on training features...")
    anomaly_model = anomaly_detection.train_anomaly_model(X_train_processed)

    print("\nSTEP 7/9: Building reference distributions for drift monitoring "
          "(numeric features, training data)...")
    reference_stats = {}
    for col in numeric_cols:
        ref = stability.compute_reference_distribution(X_train[col])
        if ref is not None:
            reference_stats[col] = ref

    print("\nSTEP 8/9: Persisting artifacts...")
    joblib.dump(best["calibrated_model"], config.MODEL_PATH)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
    joblib.dump(anomaly_model, config.ANOMALY_MODEL_PATH)
    joblib.dump(best["raw_model"], os.path.join(config.MODELS_DIR, "raw_model.pkl"))

    with open(config.REFERENCE_STATS_PATH, "w") as f:
        json.dump(reference_stats, f)

    holdout_path = os.path.join(config.MODELS_DIR, "holdout_snapshot.csv")
    holdout_raw.to_csv(holdout_path, index=False)

    metadata = {
        "model_version": config.MODEL_VERSION,
        "selected_model": best_name,
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_path": loaded["path"],
        "is_synthetic_dataset": is_synthetic,
        "n_training_records": int(len(X_train)),
        "n_test_records": int(len(X_test)),
        "target_column": target_col,
        "target_mapping_note": "1 = bad credit / predicted default, 0 = good credit",
        "feature_columns": feature_columns,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "protected_attributes": protected_attrs,
        "feature_engineering_report": fe_report,
        "xgboost_available": XGBOOST_AVAILABLE,
        "model_comparison": {
            name: {"metrics": r["metrics"], "calibration": r["calibration"]}
            for name, r in results.items()
        },
        "training_seconds": round(time.time() - start, 2),
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\nSTEP 9/9: Done.\n")
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Dataset: {loaded['path']} (synthetic={is_synthetic})")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Selected model: {best_name}")
    print(f"Test metrics: {best['metrics']}")
    print(f"Calibration:  {best['calibration']}")
    print(f"Protected attributes retained for fairness audit: {protected_attrs or 'none'}")
    print(f"Artifacts saved to: {config.MODELS_DIR}")
    print(f"Total training time: {round(time.time() - start, 2)}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
