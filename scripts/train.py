"""
Training pipeline for Flaschen Depot project.
"""

import logging
from pathlib import Path

from flaschen_depot.data import DataIngestion
from flaschen_depot.data.preprocessing import DataPreprocessor
from flaschen_depot.models import ModelTrainer
from flaschen_depot.utils import setup_logging
from flaschen_depot.utils.config import ConfigLoader

logger = logging.getLogger(__name__)


def run_training_pipeline():
    """
    Execute the complete training pipeline.
    """
    # Setup logging
    setup_logging()
    logger.info("Starting training pipeline")

    # Load configuration
    config = ConfigLoader()

    # Initialize components
    data_ingestion = DataIngestion(config.get("data.raw_path", "data/raw"))
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer(
        model_path="models",
        experiment_name=config.get("mlflow.experiment_name", "flaschen_depot"),
    )

    # Step 1: Load or create data
    logger.info("Step 1: Data ingestion")
    df = data_ingestion.create_sample_data(n_samples=1000)
    data_ingestion.save_data(df, "bottles.csv")

    # Step 2: Preprocess data
    logger.info("Step 2: Data preprocessing")
    df_clean = preprocessor.clean_data(df)

    categorical_cols = config.get(
        "features.categorical", ["bottle_type", "condition"]
    )
    df_encoded = preprocessor.encode_categorical(df_clean, categorical_cols)

    # Step 3: Prepare features and target
    logger.info("Step 3: Feature preparation")
    target_column = config.get("features.target", "condition")

    if target_column in df_encoded.columns:
        X, y = preprocessor.prepare_features(df_encoded, target_column)
    else:
        logger.warning(f"Target column '{target_column}' not found, using first column")
        X = df_encoded.iloc[:, 1:]
        y = df_encoded.iloc[:, 0]

    # Step 4: Split data
    logger.info("Step 4: Train-test split")
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X, y, test_size=config.get("data.train_test_split", 0.2)
    )

    # Step 5: Train model
    logger.info("Step 5: Model training")
    hyperparameters = config.get_hyperparameters()
    model_type = config.get("model.type", "classifier")

    if model_type == "classifier":
        trainer.train_classifier(X_train, y_train, hyperparameters)
        metrics = trainer.evaluate_classifier(X_test, y_test)
    else:
        trainer.train_regressor(X_train, y_train, hyperparameters)
        metrics = trainer.evaluate_regressor(X_test, y_test)

    logger.info(f"Model evaluation metrics: {metrics}")

    # Step 6: Save model
    logger.info("Step 6: Saving model")
    trainer.save_model("model.pkl")

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    run_training_pipeline()
