"""
Test suite for model training module.
"""

import mlflow
import numpy as np
import pytest

from flaschen_depot.models import ModelTrainer


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 3, 100)
    X_test = np.random.rand(20, 5)
    y_test = np.random.randint(0, 3, 20)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def trainer():
    """Create ModelTrainer instance for testing."""
    return ModelTrainer(model_path="/tmp/test_models")


@pytest.fixture(autouse=True)
def cleanup_mlflow_runs():
    """Cleanup MLflow runs after each test."""
    yield
    # End any active runs
    if mlflow.active_run():
        mlflow.end_run()


def test_train_classifier(trainer, sample_training_data):
    """Test classifier training."""
    X_train, y_train, X_test, y_test = sample_training_data

    model = trainer.train_classifier(X_train, y_train)

    assert model is not None
    assert hasattr(model, "predict")


def test_evaluate_classifier(trainer, sample_training_data):
    """Test classifier evaluation."""
    X_train, y_train, X_test, y_test = sample_training_data

    trainer.train_classifier(X_train, y_train)
    metrics = trainer.evaluate_classifier(X_test, y_test)

    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_save_and_load_model(trainer, sample_training_data):
    """Test model saving and loading."""
    X_train, y_train, X_test, y_test = sample_training_data

    trainer.train_classifier(X_train, y_train)
    trainer.save_model("test_model.pkl")

    # Load model
    loaded_model = trainer.load_model("test_model.pkl")

    assert loaded_model is not None
    assert hasattr(loaded_model, "predict")
