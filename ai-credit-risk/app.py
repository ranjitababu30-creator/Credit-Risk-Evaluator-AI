"""
app.py

AI-Driven Credit Risk & Responsible Lending Platform - Flask app.

Run:
    python app.py
Then open:
    http://127.0.0.1:5000

If you see "Model has not been trained yet", run:
    python -m src.train_model
first.
"""

import json
import os
import sqlite3
import traceback
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
from flask import Flask, render_template, request, jsonify, g

import config
from src.prediction import predict_application, ArtifactBundle, ModelNotTrainedError
from src.scenario import run_scenario
from src import fairness, stability

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        os.makedirs(config.DATABASE_DIR, exist_ok=True)
        g.db = sqlite3.connect(config.DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    os.makedirs(config.DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_features TEXT NOT NULL,
            probability_of_default REAL,
            risk_score INTEGER,
            risk_category TEXT,
            recommendation TEXT,
            is_anomalous INTEGER,
            model_version TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_assessment(input_dict, result):
    db = get_db()
    db.execute(
        """INSERT INTO assessments
           (timestamp, input_features, probability_of_default, risk_score,
            risk_category, recommendation, is_anomalous, model_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            json.dumps(input_dict),
            result["probability_of_default"],
            result["risk_score"],
            result["risk_category"],
            result["recommendation"],
            int(result["anomaly"]["is_anomalous"]),
            result.get("model_version"),
        ),
    )
    db.commit()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def error_response(message, status=400):
    return jsonify({"error": message}), status


def model_ready():
    try:
        ArtifactBundle.get()
        return True, None
    except ModelNotTrainedError as e:
        return False, str(e)


def get_feature_form_spec():
    """Builds dynamic form field specs from model metadata (numeric vs
    categorical + observed categories from the training holdout)."""
    ok, msg = model_ready()
    if not ok:
        return [], msg
    bundle = ArtifactBundle.get()
    fields = []
    holdout_path = os.path.join(config.MODELS_DIR, "holdout_snapshot.csv")
    holdout_df = pd.read_csv(holdout_path) if os.path.exists(holdout_path) else None

    for col in bundle.numeric_columns:
        fields.append({"name": col, "type": "number", "label": col.replace("_", " ").title()})
    for col in bundle.categorical_columns:
        options = []
        if holdout_df is not None and col in holdout_df.columns:
            options = sorted([str(v) for v in holdout_df[col].dropna().unique().tolist()])
        fields.append({
            "name": col, "type": "select",
            "label": col.replace("_", " ").title(),
            "options": options,
        })
    return fields, None


def read_uploaded_dataset(upload):
    """Read a supported tabular upload into a DataFrame without storing it."""
    filename = (upload.filename or "").lower()
    if not filename:
        raise ValueError("The uploaded file must have a filename.")
    file_bytes = upload.read()
    if not file_bytes:
        raise ValueError("The uploaded dataset is empty.")
    try:
        if filename.endswith((".csv", ".txt")):
            return pd.read_csv(BytesIO(file_bytes))
        if filename.endswith(".json"):
            return pd.read_json(BytesIO(file_bytes))
        if filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(BytesIO(file_bytes))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Could not read '{upload.filename}': {e}") from e
    raise ValueError("Unsupported file format. Upload CSV, JSON, XLSX, or XLS.")


def score_uploaded_dataset(upload):
    """Score each upload row and partition it for the batch results panels."""
    dataframe = read_uploaded_dataset(upload)
    if dataframe.empty:
        raise ValueError("The uploaded dataset has no rows.")
    if len(dataframe) > 5000:
        raise ValueError("Upload at most 5,000 rows at a time.")

    groups = {"APPROVE": [], "REVIEW": [], "REJECT": [], "ANOMALOUS": []}
    errors = []
    for source_row, (_, row) in enumerate(dataframe.iterrows(), start=2):
        input_dict = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in row.dropna().to_dict().items()
        }
        try:
            result = predict_application(input_dict, explain=False)
            item = {
                "row_number": source_row,
                "risk_score": result["risk_score"],
                "risk_category": result["risk_category"],
                "recommendation": result["recommendation"],
                "probability_of_default": result["probability_of_default"],
                "anomaly_score": result["anomaly"]["anomaly_score"],
                "is_anomalous": result["anomaly"]["is_anomalous"],
            }
            groups[result["recommendation"]].append(item)
            if result["anomaly"]["is_anomalous"]:
                groups["ANOMALOUS"].append(item)
            save_assessment(input_dict, result)
        except (ValueError, TypeError) as e:
            errors.append({"row_number": source_row, "error": str(e)})
    return {
        "filename": upload.filename,
        "total_rows": len(dataframe),
        "processed_rows": len(dataframe) - len(errors),
        "groups": groups,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    ok, msg = model_ready()
    stats = None
    if ok:
        db = get_db()
        rows = db.execute("SELECT * FROM assessments").fetchall()
        total = len(rows)
        approvals = sum(1 for r in rows if r["recommendation"] == "APPROVE")
        reviews = sum(1 for r in rows if r["recommendation"] == "REVIEW")
        rejections = sum(1 for r in rows if r["recommendation"] == "REJECT")
        anomalies = sum(1 for r in rows if r["is_anomalous"])
        avg_score = round(sum(r["risk_score"] for r in rows) / total, 1) if total else None
        high_risk = sum(1 for r in rows if r["risk_category"] in ("HIGH", "VERY HIGH"))

        bundle = ArtifactBundle.get()
        model_metrics = bundle.metadata["model_comparison"][bundle.metadata["selected_model"]]

        stats = {
            "total": total, "approvals": approvals, "reviews": reviews,
            "rejections": rejections, "anomalies": anomalies,
            "avg_score": avg_score, "high_risk": high_risk,
            "approval_rate": round(100 * approvals / total, 1) if total else 0,
            "review_rate": round(100 * reviews / total, 1) if total else 0,
            "rejection_rate": round(100 * rejections / total, 1) if total else 0,
            "model_metrics": model_metrics,
            "selected_model": bundle.metadata["selected_model"],
        }
    return render_template("index.html", model_ready=ok, error=msg, stats=stats,
                            disclaimer=config.GENERAL_DISCLAIMER)


@app.route("/assessment")
def assessment():
    fields, msg = get_feature_form_spec()
    return render_template("assessment.html", fields=fields, error=msg,
                            disclaimer=config.GENERAL_DISCLAIMER)


@app.route("/dataset")
def dataset_page():
    ok, msg = model_ready()
    return render_template("dataset.html", model_ready=ok, error=msg,
                           disclaimer=config.GENERAL_DISCLAIMER)


@app.route("/fairness")
def fairness_page():
    return render_template("fairness.html", disclaimer=config.FAIRNESS_DISCLAIMER)


@app.route("/anomaly")
def anomaly_page():
    return render_template("anomaly.html")


@app.route("/stability")
def stability_page():
    return render_template("stability.html")


@app.route("/scenario")
def scenario_page():
    fields, msg = get_feature_form_spec()
    return render_template("scenario.html", fields=fields, error=msg,
                            disclaimer=config.SCENARIO_DISCLAIMER)


@app.route("/monitoring")
def monitoring_page():
    ok, msg = model_ready()
    metadata = ArtifactBundle.get().metadata if ok else None
    return render_template("explanation.html", model_ready=ok, error=msg,
                            metadata=metadata, monitoring=True)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    ok, msg = model_ready()
    if not ok:
        return error_response(msg, 503)
    payload = request.get_json(silent=True) or {}
    if not payload:
        return error_response("Request body must be a JSON object of feature values.")
    try:
        result = predict_application(payload, explain=False)
        save_assessment(payload, result)
        return jsonify(result)
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception:  # noqa: BLE001
        app.logger.error(traceback.format_exc())
        return error_response("An unexpected error occurred while scoring this application.", 500)


@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    ok, msg = model_ready()
    if not ok:
        return error_response(msg, 503)
    upload = request.files.get("dataset")
    if upload is None:
        return error_response("Attach a dataset using the 'dataset' form field.")
    try:
        return jsonify(score_uploaded_dataset(upload))
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception:  # noqa: BLE001
        app.logger.error(traceback.format_exc())
        return error_response("An unexpected error occurred while processing the dataset.", 500)


@app.route("/api/explain", methods=["POST"])
def api_explain():
    ok, msg = model_ready()
    if not ok:
        return error_response(msg, 503)
    payload = request.get_json(silent=True) or {}
    if not payload:
        return error_response("Request body must be a JSON object of feature values.")
    try:
        result = predict_application(payload, explain=True)
        return jsonify(result)
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception:  # noqa: BLE001
        app.logger.error(traceback.format_exc())
        return error_response("An unexpected error occurred while explaining this application.", 500)


@app.route("/api/scenario", methods=["POST"])
def api_scenario():
    ok, msg = model_ready()
    if not ok:
        return error_response(msg, 503)
    payload = request.get_json(silent=True) or {}
    original = payload.get("original")
    modified = payload.get("modified")
    if not original or not modified:
        return error_response("Request body must contain 'original' and 'modified' objects.")
    try:
        result = run_scenario(original, modified)
        return jsonify(result)
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception:  # noqa: BLE001
        app.logger.error(traceback.format_exc())
        return error_response("An unexpected error occurred while running the scenario.", 500)


@app.route("/api/fairness")
def api_fairness():
    ok, msg = model_ready()
    if not ok:
        return error_response(msg, 503)
    bundle = ArtifactBundle.get()
    protected_attrs = bundle.metadata.get("protected_attributes", [])
    holdout_path = os.path.join(config.MODELS_DIR, "holdout_snapshot.csv")
    if not protected_attrs or not os.path.exists(holdout_path):
        return jsonify({
            "available": False,
            "reason": "Fairness audit unavailable for this dataset because "
                      "appropriate audit attributes are not provided.",
        })
    try:
        holdout_df = pd.read_csv(holdout_path)
        feature_cols = bundle.feature_columns
        processed = bundle.preprocessor.transform(holdout_df[feature_cols])
        if hasattr(processed, "toarray"):
            processed = processed.toarray()
        probs = bundle.model.predict_proba(processed)[:, 1]
        holdout_df = holdout_df.copy()
        holdout_df["_pred"] = (probs >= 0.5).astype(int)
        holdout_df["_true"] = holdout_df[bundle.metadata["target_column"]]

        result = fairness.audit_all_protected_attributes(
            holdout_df, "_true", "_pred", protected_attrs
        )
        return jsonify(result)
    except Exception:  # noqa: BLE001
        app.logger.error(traceback.format_exc())
        return error_response("An unexpected error occurred while computing fairness metrics.", 500)


@app.route("/api/anomalies")
def api_anomalies():
    ok, msg = model_ready()
    if not ok:
        return error_response(msg, 503)
    db = get_db()
    rows = db.execute(
        "SELECT id, timestamp, risk_score, risk_category, recommendation, is_anomalous "
        "FROM assessments WHERE is_anomalous = 1 ORDER BY id DESC LIMIT 100"
    ).fetchall()
    return jsonify({"count": len(rows), "anomalous_applications": [dict(r) for r in rows]})


@app.route("/api/stability")
def api_stability():
    ok, msg = model_ready()
    if not ok:
        return error_response(msg, 503)
    if not os.path.exists(config.REFERENCE_STATS_PATH):
        return jsonify({"available": False, "reason": "No reference distributions available for comparison."})
    with open(config.REFERENCE_STATS_PATH) as f:
        reference_stats = json.load(f)

    holdout_path = os.path.join(config.MODELS_DIR, "holdout_snapshot.csv")
    if not os.path.exists(holdout_path):
        return jsonify({"available": False, "reason": "No current data snapshot available for comparison."})
    current_df = pd.read_csv(holdout_path)
    result = stability.audit_stability(reference_stats, current_df)
    return jsonify(result)


@app.route("/api/model-metrics")
def api_model_metrics():
    ok, msg = model_ready()
    if not ok:
        return error_response(msg, 503)
    bundle = ArtifactBundle.get()
    return jsonify(bundle.metadata)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    app.logger.error(traceback.format_exc())
    return jsonify({"error": "An unexpected server error occurred."}), 500


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="127.0.0.1", port=5000)
else:
    # Ensure DB exists when imported (e.g. by tests) too.
    init_db()
