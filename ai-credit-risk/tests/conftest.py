import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import config


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Ensures a trained model exists before any test runs. Trains once
    per test session if artifacts are missing."""
    if not os.path.exists(config.MODEL_PATH):
        from src import train_model
        train_model.main()
    yield
