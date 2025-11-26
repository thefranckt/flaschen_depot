"""
Model training module for Flaschen Depot project.
Handles model training, evaluation, and persistence.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Class for training and evaluating machine learning models.
    """

    def __init__(self, model_path: str = "models", experiment_name: str = "flaschen_depot"):
        """
        Initialize ModelTrainer.

        Args:
            model_path: Path to save trained models
            experiment_name: Name for MLflow experiment
        """
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.model = None

        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        logger.info(f"ModelTrainer initialized with experiment: {experiment_name}")

    def train_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> RandomForestClassifier:
        """
        Train a Random Forest classifier.

        Args:
            X_train: Training features
            y_train: Training labels
            params: Model hyperparameters

        Returns:
            Trained classifier model
        """
        logger.info("Training Random Forest classifier")

        if params is None:
            params = {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
            }

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)

            # Train model
            self.model = RandomForestClassifier(**params)
            self.model.fit(X_train, y_train)

            # Log model
            mlflow.sklearn.log_model(self.model, "model")

            logger.info("Classifier training completed")

        return self.model

    def train_regressor(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> RandomForestRegressor:
        """
        Train a Random Forest regressor.

        Args:
            X_train: Training features
            y_train: Training labels
            params: Model hyperparameters

        Returns:
            Trained regressor model
        """
        logger.info("Training Random Forest regressor")

        if params is None:
            params = {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
            }

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)

            # Train model
            self.model = RandomForestRegressor(**params)
            self.model.fit(X_train, y_train)

            # Log model
            mlflow.sklearn.log_model(self.model, "model")

            logger.info("Regressor training completed")

        return self.model

    def evaluate_classifier(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classifier performance.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating classifier")

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Classifier accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))

        # Log metrics to MLflow
        try:
            mlflow.log_metric("accuracy", accuracy)
        except Exception as e:
            logger.warning(f"Could not log metrics to MLflow: {e}")

        return {"accuracy": accuracy}

    def evaluate_regressor(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regressor performance.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating regressor")

        y_pred = self.model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
        }

        logger.info(f"Regressor metrics: {metrics}")

        # Log metrics to MLflow
        try:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
        except Exception as e:
            logger.warning(f"Could not log metrics to MLflow: {e}")

        return metrics

    def save_model(self, filename: str = "model.pkl") -> None:
        """
        Save the trained model to disk.

        Args:
            filename: Name of the file to save the model
        """
        if self.model is None:
            logger.error("No model to save")
            raise ValueError("No trained model available")

        filepath = self.model_path / filename
        logger.info(f"Saving model to {filepath}")

        joblib.dump(self.model, filepath)
        logger.info("Model saved successfully")

    def load_model(self, filename: str = "model.pkl") -> Any:
        """
        Load a trained model from disk.

        Args:
            filename: Name of the file to load the model from

        Returns:
            Loaded model
        """
        filepath = self.model_path / filename
        logger.info(f"Loading model from {filepath}")

        if not filepath.exists():
            logger.error(f"Model file not found: {filepath}")
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.model = joblib.load(filepath)
        logger.info("Model loaded successfully")

        return self.model
