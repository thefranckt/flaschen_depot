#!/usr/bin/env python
"""
Vollständige Pipeline zum Trainieren des Modells
"""
import logging
import sys
from pathlib import Path

# Füge src zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.make_dataset import DataLoader, generate_sample_data
from src.features.build_features import FeatureEngineer
from src.models.train_model import ModelTrainer
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Führt die komplette ML-Pipeline aus"""
    
    logger.info("="*60)
    logger.info("FLASCHENDEPOT ML PIPELINE")
    logger.info("="*60)
    
    # Schritt 1: Daten generieren (falls nicht vorhanden)
    logger.info("\n[1/5] Generiere Beispieldaten...")
    raw_data_path = "data/raw/bottles.csv"
    if not Path(raw_data_path).exists():
        generate_sample_data(raw_data_path, n_samples=1000)
    else:
        logger.info(f"Daten existieren bereits: {raw_data_path}")
    
    # Schritt 2: Daten laden und verarbeiten
    logger.info("\n[2/5] Lade und verarbeite Daten...")
    loader = DataLoader()
    df = loader.load_raw_data('bottles.csv')
    df_clean = loader.clean_data(df)
    
    target = loader.config['features']['target_variable']
    X_train, X_test, y_train, y_test = loader.split_data(df_clean, target)
    
    loader.save_processed_data(X_train, X_test, y_train, y_test)
    
    # Schritt 3: Feature Engineering
    logger.info("\n[3/5] Feature Engineering...")
    fe = FeatureEngineer()
    
    X_train_processed = fe.fit_transform(X_train)
    X_test_processed = fe.transform(X_test)
    
    fe.save_preprocessor()
    
    logger.info(f"Training Features Shape: {X_train_processed.shape}")
    logger.info(f"Test Features Shape: {X_test_processed.shape}")
    
    # Schritt 4: Model Training
    logger.info("\n[4/5] Trainiere Modelle...")
    trainer = ModelTrainer()
    
    # Vergleiche verschiedene Modelle
    results = trainer.compare_models(
        X_train_processed, 
        y_train.values if isinstance(y_train, pd.Series) else y_train,
        X_test_processed, 
        y_test.values if isinstance(y_test, pd.Series) else y_test,
        model_names=['random_forest', 'gradient_boosting', 'logistic_regression']
    )
    
    # Trainiere bestes Modell
    best_model_name = results.iloc[0]['model']
    logger.info(f"\nTrainiere bestes Modell: {best_model_name}")
    
    trainer.train(
        X_train_processed, 
        y_train.values if isinstance(y_train, pd.Series) else y_train,
        model_name=best_model_name
    )
    
    # Schritt 5: Evaluation und Speichern
    logger.info("\n[5/5] Finale Evaluation...")
    final_metrics = trainer.evaluate(
        X_test_processed, 
        y_test.values if isinstance(y_test, pd.Series) else y_test
    )
    
    trainer.save_model()
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE ERFOLGREICH ABGESCHLOSSEN!")
    logger.info("="*60)
    logger.info(f"\nFinale Metriken:")
    for metric, value in final_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nGespeicherte Artefakte:")
    logger.info("  - Modell: models/model.pkl")
    logger.info("  - Preprocessor: models/preprocessor.pkl")
    logger.info("  - Daten: data/processed/")
    logger.info("  - MLflow: mlruns/")


if __name__ == "__main__":
    main()
