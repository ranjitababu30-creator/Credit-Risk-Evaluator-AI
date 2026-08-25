from io import BytesIO

import pytest

import app as flask_app_module
from src.prediction import ArtifactBundle


@pytest.fixture
def client():
    flask_app_module.app.config["TESTING"] = True
    flask_app_module.init_db()
    with flask_app_module.app.test_client() as client:
        yield client


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_assessment_page_loads(client):
    resp = client.get("/assessment")
    assert resp.status_code == 200


def test_dataset_page_loads(client):
    resp = client.get("/dataset")
    assert resp.status_code == 200


def test_scenario_page_loads(client):
    resp = client.get("/scenario")
    assert resp.status_code == 200


def test_fairness_page_loads(client):
    resp = client.get("/fairness")
    assert resp.status_code == 200


def test_anomaly_page_loads(client):
    resp = client.get("/anomaly")
    assert resp.status_code == 200


def test_stability_page_loads(client):
    resp = client.get("/stability")
    assert resp.status_code == 200


def test_monitoring_page_loads(client):
    resp = client.get("/monitoring")
    assert resp.status_code == 200


def _sample_payload():
    bundle = ArtifactBundle.get()
    values = {}
    for col in bundle.numeric_columns:
        values[col] = 1000
    for col in bundle.categorical_columns:
        values[col] = "other"
    return values


def test_api_predict(client):
    resp = client.post("/api/predict", json=_sample_payload())
    assert resp.status_code == 200
    data = resp.get_json()
    assert "risk_score" in data
    assert "recommendation" in data


def test_api_predict_empty_body_returns_400(client):
    resp = client.post("/api/predict", json={})
    # Empty dict is falsy -> treated as missing body
    assert resp.status_code == 400


def test_api_batch_predict_csv(client):
    bundle = ArtifactBundle.get()
    values = {col: 1000 for col in bundle.numeric_columns}
    values.update({col: "other" for col in bundle.categorical_columns})
    csv_data = ",".join(values) + "\n" + ",".join(str(value) for value in values.values()) + "\n"
    resp = client.post(
        "/api/batch-predict",
        data={"dataset": (BytesIO(csv_data.encode()), "applications.csv")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total_rows"] == 1
    assert data["processed_rows"] == 1
    assert sum(len(data["groups"][key]) for key in ("APPROVE", "REVIEW", "REJECT")) == 1


def test_api_batch_predict_rejects_unsupported_format(client):
    resp = client.post(
        "/api/batch-predict",
        data={"dataset": (BytesIO(b"data"), "applications.pdf")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_api_explain(client):
    resp = client.post("/api/explain", json=_sample_payload())
    assert resp.status_code == 200
    data = resp.get_json()
    assert "explanation" in data


def test_api_scenario(client):
    payload = {"original": _sample_payload(), "modified": _sample_payload()}
    resp = client.post("/api/scenario", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "before" in data and "after" in data


def test_api_scenario_missing_fields_returns_400(client):
    resp = client.post("/api/scenario", json={"original": {}})
    assert resp.status_code == 400


def test_api_fairness(client):
    resp = client.get("/api/fairness")
    assert resp.status_code == 200


def test_api_anomalies(client):
    resp = client.get("/api/anomalies")
    assert resp.status_code == 200


def test_api_stability(client):
    resp = client.get("/api/stability")
    assert resp.status_code == 200


def test_api_model_metrics(client):
    resp = client.get("/api/model-metrics")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "selected_model" in data


def test_404_returns_json(client):
    resp = client.get("/this-route-does-not-exist")
    assert resp.status_code == 404
    data = resp.get_json()
    assert "error" in data
