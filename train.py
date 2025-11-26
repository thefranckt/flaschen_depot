"""
Model Training Script
Trains service time prediction model with MLflow tracking and versioning.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, evaluation, and saving with MLflow."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize ModelTrainer with configuration.
        
        Args:
            config_path: Path to configuration file
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
        Load and prepare data for training.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
        """
        logger.info("Loading and preparing data...")
        
        # Load data
        loader = DataLoader(self.config['data']['raw_dir'])
        orders, articles, service_times, driver_mapping = loader.load_all()
        
        # Feature engineering
        engineer = FeatureEngineer(random_state=self.random_state)
        X, y, feature_names, df = engineer.process_pipeline(
            orders, articles, service_times, driver_mapping
        )
        
        # Save processed data
        processed_dir = Path(self.config['data']['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        X.to_parquet(processed_dir / 'features.parquet')
        y.to_frame('service_time_in_minutes').to_parquet(processed_dir / 'target.parquet')
        df.to_parquet(processed_dir / 'full_dataset.parquet')
        
        logger.info(f"Processed data saved to {processed_dir}")
        
        # Split data
        test_size = self.config['model']['test_size']
        val_size = self.config['model']['validation_size']
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    
    def create_model(self):
        """
        Create model based on configuration.
        
        Returns:
            Initialized model
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
        Evaluate model and return metrics.
        
        Args:
            model: Trained model
            X: Features
            y: Target
            dataset_name: Name for logging
            
        Returns:
            Dictionary of metrics
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
        Execute full training pipeline with MLflow tracking.
        
        Returns:
            Tuple of (model, metrics, feature_names)
        """
        logger.info("Starting training pipeline...")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Load and prepare data
            X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.load_and_prepare_data()
            
            # Log data info
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("feature_names", feature_names)
            
            # Create and train model
            model = self.create_model()
            
            # Log model parameters
            mlflow.log_params(self.config['model'][self.model_type])
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("random_state", self.random_state)
            
            logger.info("Training model...")
            model.fit(X_train, y_train)
            logger.info("Training complete")
            
            # Evaluate on all datasets
            train_metrics = self.evaluate_model(model, X_train, y_train, "train")
            val_metrics = self.evaluate_model(model, X_val, y_val, "val")
            test_metrics = self.evaluate_model(model, X_test, y_test, "test")
            
            # Combine all metrics
            all_metrics = {**train_metrics, **val_metrics, **test_metrics}
            
            # Log metrics to MLflow
            mlflow.log_metrics(all_metrics)
            
            # Log feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                logger.info("\nFeature Importance:")
                logger.info(feature_importance.to_string())
                
                # Save feature importance
                importance_path = Path("models") / "feature_importance.csv"
                importance_path.parent.mkdir(parents=True, exist_ok=True)
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path))
            
            # Save model
            model_dir = Path(self.config['mlflow']['artifact_location'])
            model_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = model_dir / f"model_{self.model_type}_{timestamp}.joblib"
            
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Also save as 'latest'
            latest_path = model_dir / "model_latest.joblib"
            joblib.dump(model, latest_path)
            logger.info(f"Model saved as latest to {latest_path}")
            
            # Log model to MLflow
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
    """Main training function."""
    trainer = ModelTrainer()
    model, metrics, feature_names = trainer.train()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModel Type: {trainer.model_type}")
    print(f"Features: {len(feature_names)}")
    print("\nTest Set Metrics:")
    print(f"  RMSE: {metrics['test_rmse']:.4f} minutes")
    print(f"  MAE: {metrics['test_mae']:.4f} minutes")
    print(f"  R²: {metrics['test_r2']:.4f}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
