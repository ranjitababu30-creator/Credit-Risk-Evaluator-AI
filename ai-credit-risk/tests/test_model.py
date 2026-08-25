import json
import os

import config
from src import data_loader, feature_engineering, preprocessing


def test_dataset_loads_and_has_binary_target():
    loaded = data_loader.load_dataset()
    df = loaded["df"]
    assert len(df) > 0
    assert set(df[loaded["target_col"]].unique().tolist()) <= {0, 1}


def test_feature_engineering_reports_created_and_skipped():
    loaded = data_loader.load_dataset()
    df, report = feature_engineering.engineer_features(loaded["df"])
    assert "created" in report and "skipped" in report
    assert isinstance(report["created"], list)


def test_preprocessor_builds_without_error():
    loaded = data_loader.load_dataset()
    df, _ = feature_engineering.engineer_features(loaded["df"])
    feature_cols = [c for c in df.columns if c != loaded["target_col"]
                     and c not in loaded["protected_attrs"]]
    preprocessor, num_cols, cat_cols = preprocessing.build_preprocessor(df, feature_cols)
    transformed = preprocessor.fit_transform(df[feature_cols])
    assert transformed.shape[0] == len(df)


def test_training_artifacts_exist():
    for path in [config.MODEL_PATH, config.PREPROCESSOR_PATH,
                 config.ANOMALY_MODEL_PATH, config.METADATA_PATH]:
        assert os.path.exists(path), f"Missing artifact: {path}"


def test_metadata_has_model_comparison():
    with open(config.METADATA_PATH) as f:
        metadata = json.load(f)
    assert "model_comparison" in metadata
    assert metadata["selected_model"] in metadata["model_comparison"]
    for name, r in metadata["model_comparison"].items():
        assert "roc_auc" in r["metrics"]
        assert "brier_score" in r["calibration"]
