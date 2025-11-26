"""
Modell-Training-Skript
Trainiert Service-Zeit-Vorhersagemodell mit MLflow-Tracking und Versionierung.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer

# Logging einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Behandelt Modell-Training, -Evaluation und -Speicherung mit MLflow."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        ModelTrainer mit Konfiguration initialisieren.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.random_state = self.config['model']['random_state']
        self.model_type = self.config['model']['type']
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        logger.info(f"ModelTrainer initialized with config: {config_path}")
    
    def load_and_prepare_data(self):
        """
        Daten für Training laden und vorbereiten.
        
        Returns:
            Tuple von (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
        """
        logger.info("Lade und bereite Daten vor...")
        
        # Daten laden
        loader = DataLoader(self.config['data']['raw_dir'])
        orders, articles, service_times, driver_mapping = loader.load_all()
        
        # Feature Engineering
        engineer = FeatureEngineer(random_state=self.random_state)
        X, y, feature_names, df = engineer.process_pipeline(
            orders, articles, service_times, driver_mapping
        )
        
        # Verarbeitete Daten speichern
        processed_dir = Path(self.config['data']['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        X.to_parquet(processed_dir / 'features.parquet')
        y.to_frame('service_time_in_minutes').to_parquet(processed_dir / 'target.parquet')
        df.to_parquet(processed_dir / 'full_dataset.parquet')
        
        logger.info(f"Verarbeitete Daten gespeichert in {processed_dir}")
        
        # Daten aufteilen
        test_size = self.config['model']['test_size']
        val_size = self.config['model']['validation_size']
        
        # Erste Aufteilung: train+val und test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Zweite Aufteilung: train und val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
        )
        
        logger.info(f"Daten aufgeteilt - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    
    def create_model(self):
        """
        Modell basierend auf Konfiguration erstellen.
        
        Returns:
            Initialisiertes Modell
        """
        model_params = self.config['model'][self.model_type]
        
        if self.model_type == 'lightgbm':
            model = LGBMRegressor(**model_params)
        elif self.model_type == 'xgboost':
            model = XGBRegressor(**model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Use 'lightgbm' or 'xgboost'")
        
        logger.info(f"Created {self.model_type} model with params: {model_params}")
        return model
    
    def evaluate_model(self, model, X, y, dataset_name=""):
        """
        Modell evaluieren und Metriken zurückgeben.
        
        Args:
            model: Trainiertes Modell
            X: Features
            y: Target
            dataset_name: Name für Logging
            
        Returns:
            Dictionary von Metriken
        """
        predictions = model.predict(X)
        
        metrics = {
            f'{dataset_name}_rmse': np.sqrt(mean_squared_error(y, predictions)),
            f'{dataset_name}_mae': mean_absolute_error(y, predictions),
            f'{dataset_name}_r2': r2_score(y, predictions)
        }
        
        logger.info(f"{dataset_name} Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def train(self):
        """
        Vollständige Training-Pipeline mit MLflow-Tracking ausführen.
        
        Returns:
            Tuple von (model, metrics, feature_names)
        """
        logger.info("Starting training pipeline...")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Load and prepare data
            X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.load_and_prepare_data()
            
            # Daten-Info loggen
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("feature_names", feature_names)
            
            # Modell erstellen und trainieren
            model = self.create_model()
            
            # Modell-Parameter loggen
            mlflow.log_params(self.config['model'][self.model_type])
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("random_state", self.random_state)
            
            logger.info("Trainiere Modell...")
            model.fit(X_train, y_train)
            logger.info("Training abgeschlossen")
            
            # Auf allen Datensätzen evaluieren
            train_metrics = self.evaluate_model(model, X_train, y_train, "train")
            val_metrics = self.evaluate_model(model, X_val, y_val, "val")
            test_metrics = self.evaluate_model(model, X_test, y_test, "test")
            
            # Alle Metriken kombinieren
            all_metrics = {**train_metrics, **val_metrics, **test_metrics}
            
            # Metriken zu MLflow loggen
            mlflow.log_metrics(all_metrics)
            
            # Feature-Wichtigkeit loggen
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                logger.info("\nFeature-Wichtigkeit:")
                logger.info(feature_importance.to_string())
                
                # Feature-Wichtigkeit speichern
                importance_path = Path("models") / "feature_importance.csv"
                importance_path.parent.mkdir(parents=True, exist_ok=True)
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path))
            
            # Modell speichern
            model_dir = Path(self.config['mlflow']['artifact_location'])
            model_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = model_dir / f"model_{self.model_type}_{timestamp}.joblib"
            
            joblib.dump(model, model_path)
            logger.info(f"Modell gespeichert in {model_path}")
            
            # Auch als 'latest' speichern
            latest_path = model_dir / "model_latest.joblib"
            joblib.dump(model, latest_path)
            logger.info(f"Modell als latest gespeichert in {latest_path}")
            
            # Modell zu MLflow loggen
            mlflow.sklearn.log_model(model, "model")
            
            # Save metadata
            metadata = {
                'model_type': self.model_type,
                'timestamp': timestamp,
                'metrics': all_metrics,
                'feature_names': feature_names,
                'model_path': str(model_path)
            }
            
            metadata_path = model_dir / f"metadata_{timestamp}.yaml"
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f)
            mlflow.log_artifact(str(metadata_path))
            
            logger.info(f"Training pipeline complete. Run ID: {mlflow.active_run().info.run_id}")
            
            return model, all_metrics, feature_names


def main():
    """Haupt-Training-Funktion."""
    trainer = ModelTrainer()
    model, metrics, feature_names = trainer.train()
    
    print("\n" + "=" * 80)
    print("TRAINING ABGESCHLOSSEN")
    print("=" * 80)
    print(f"\nModell-Typ: {trainer.model_type}")
    print(f"Features: {len(feature_names)}")
    print("\nTest-Set-Metriken:")
    print(f"  RMSE: {metrics['test_rmse']:.4f} Minuten")
    print(f"  MAE: {metrics['test_mae']:.4f} Minuten")
    print(f"  R²: {metrics['test_r2']:.4f}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
