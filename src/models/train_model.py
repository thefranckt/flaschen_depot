"""
Model Training Modul für Service-Zeit-Vorhersage
"""
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import json

import pandas as pd
import numpy as np
import yaml
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Klasse für Model Training und Evaluation"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialisiert den ModelTrainer
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.best_model = None
        self.model_path = Path(self.config['training']['model_path'])
        
        # MLflow Setup
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
    
    def get_model(self, model_name: str = None) -> Any:
        """
        Erstellt ein ML-Modell basierend auf Konfiguration
        
        Args:
            model_name: Name des Modells (optional)
            
        Returns:
            Instanziiertes ML-Modell
        """
        if model_name is None:
            model_name = self.config['model']['algorithm']
        
        models = {
            'random_forest': RandomForestRegressor(
                **self.config['model']['hyperparameters']
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(
                alpha=1.0,
                random_state=42
            ),
            'lasso': Lasso(
                alpha=1.0,
                random_state=42
            ),
            'svr': SVR(
                kernel='rbf',
                C=1.0
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Unbekanntes Modell: {model_name}")
        
        logger.info(f"Erstelle Modell: {model_name}")
        return models[model_name]
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        model_name: str = None
    ) -> Any:
        """
        Trainiert das Modell
        
        Args:
            X_train: Training Features
            y_train: Training Labels
            model_name: Name des Modells (optional)
            
        Returns:
            Trainiertes Modell
        """
        with mlflow.start_run():
            self.model = self.get_model(model_name)
            
            logger.info("Starte Training")
            self.model.fit(X_train, y_train)
            logger.info("Training abgeschlossen")
            
            # Log Parameters zu MLflow
            mlflow.log_params(self.model.get_params())
            
            # Cross-Validation
            cv_scores = cross_val_score(
                self.model, 
                X_train, 
                y_train, 
                cv=self.config['training']['cross_validation'],
                scoring='neg_mean_absolute_error'
            )
            
            logger.info(f"Cross-Validation MAE Scores: {-cv_scores}")
            logger.info(f"Mean CV MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Log Metrics zu MLflow
            mlflow.log_metric("cv_mae_mean", -cv_scores.mean())
            mlflow.log_metric("cv_mae_std", cv_scores.std())
            
            return self.model
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluiert das Modell
        
        Args:
            X_test: Test Features
            y_test: Test Labels
            
        Returns:
            Dictionary mit Metriken
        """
        if self.model is None:
            raise ValueError("Modell muss zuerst trainiert werden!")
        
        logger.info("Starte Evaluation")
        
        # Vorhersagen
        y_pred = self.model.predict(X_test)
        
        # Metriken berechnen
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2_score': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred)
        }
        
        # Log Metriken zu MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Ausgabe
        logger.info("Evaluation Metriken:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Zusätzliche Statistiken
        residuals = y_test - y_pred
        logger.info(f"\nResidual Statistiken:")
        logger.info(f"  Mean: {residuals.mean():.4f}")
        logger.info(f"  Std: {residuals.std():.4f}")
        logger.info(f"  Min: {residuals.min():.4f}")
        logger.info(f"  Max: {residuals.max():.4f}")
        
        return metrics
    
    def hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict[str, list]
    ) -> Any:
        """
        Führt Hyperparameter-Tuning durch
        
        Args:
            X_train: Training Features
            y_train: Training Labels
            param_grid: Parameter-Grid für GridSearch
            
        Returns:
            Best model nach GridSearch
        """
        logger.info("Starte Hyperparameter Tuning")
        
        base_model = self.get_model()
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.config['training']['cross_validation'],
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best Parameters: {grid_search.best_params_}")
        logger.info(f"Best CV MAE Score: {-grid_search.best_score_:.4f}")
        
        # Log zu MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_mae", -grid_search.best_score_)
        
        self.best_model = grid_search.best_estimator_
        return self.best_model
    
    def save_model(self, filename: str = "model.pkl"):
        """
        Speichert das trainierte Modell
        
        Args:
            filename: Name der Modelldatei
        """
        if self.model is None:
            raise ValueError("Kein Modell zum Speichern vorhanden!")
        
        self.model_path.mkdir(parents=True, exist_ok=True)
        filepath = self.model_path / filename
        
        joblib.dump(self.model, filepath)
        logger.info(f"Modell gespeichert: {filepath}")
        
        # Log Modell zu MLflow
        mlflow.sklearn.log_model(self.model, "model")
    
    def load_model(self, filename: str = "model.pkl"):
        """
        Lädt ein gespeichertes Modell
        
        Args:
            filename: Name der Modelldatei
        """
        filepath = self.model_path / filename
        self.model = joblib.load(filepath)
        logger.info(f"Modell geladen: {filepath}")
    
    def compare_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_names: list = None
    ) -> pd.DataFrame:
        """
        Vergleicht verschiedene Modelle
        
        Args:
            X_train, y_train: Training Data
            X_test, y_test: Test Data
            model_names: Liste von Modellnamen
            
        Returns:
            DataFrame mit Vergleichsmetriken
        """
        if model_names is None:
            model_names = ['random_forest', 'gradient_boosting', 'linear_regression']
        
        results = []
        
        for model_name in model_names:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'='*50}")
            
            # Trainiere Modell
            self.train(X_train, y_train, model_name)
            
            # Evaluiere Modell
            metrics = self.evaluate(X_test, y_test)
            metrics['model'] = model_name
            results.append(metrics)
        
        # Erstelle Vergleichs-DataFrame
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('mae', ascending=True)
        
        logger.info("\n" + "="*50)
        logger.info("Model Comparison Results")
        logger.info("="*50)
        print(df_results.to_string(index=False))
        
        return df_results


if __name__ == "__main__":
    # Lade Daten
    data_path = Path("data/processed")
    X_train = pd.read_csv(data_path / "X_train.csv").values
    X_test = pd.read_csv(data_path / "X_test.csv").values
    y_train = pd.read_csv(data_path / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_path / "y_test.csv").values.ravel()
    
    # Feature Engineering anwenden
    from src.features.build_features import FeatureEngineer
    
    fe = FeatureEngineer()
    X_train_df = pd.read_csv(data_path / "X_train.csv")
    X_test_df = pd.read_csv(data_path / "X_test.csv")
    
    X_train = fe.fit_transform(X_train_df)
    X_test = fe.transform(X_test_df)
    fe.save_preprocessor()
    
    # Model Training
    trainer = ModelTrainer()
    
    # Vergleiche mehrere Modelle
    results = trainer.compare_models(X_train, y_train, X_test, y_test)
    
    # Trainiere bestes Modell
    trainer.train(X_train, y_train, model_name='random_forest')
    metrics = trainer.evaluate(X_test, y_test)
    
    # Speichere Modell
    trainer.save_model()
    
    logger.info("\nModel Training abgeschlossen!")
