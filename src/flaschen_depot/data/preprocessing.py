"""
Data preprocessing module for Flaschen Depot project.
Handles data cleaning, transformation, and feature engineering.
"""

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Class for preprocessing bottle depot data.
    """

    def __init__(self):
        """Initialize preprocessor with necessary transformers."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        logger.info("DataPreprocessor initialized")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input data by handling missing values and outliers.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning")
        df_clean = df.copy()

        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")

        # Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        df_clean = df_clean.dropna()
        logger.info(f"Removed {missing_before} missing values")

        return df_clean

    def encode_categorical(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Encode categorical columns using LabelEncoder.

        Args:
            df: Input DataFrame
            columns: List of categorical columns to encode

        Returns:
            DataFrame with encoded categorical columns
        """
        logger.info(f"Encoding categorical columns: {columns}")
        df_encoded = df.copy()

        for col in columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                logger.info(f"Encoded column '{col}'")

        return df_encoded

    def scale_features(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.

        Args:
            df: Input DataFrame
            columns: List of columns to scale

        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling features: {columns}")
        df_scaled = df.copy()

        df_scaled[columns] = self.scaler.fit_transform(df[columns])
        logger.info("Feature scaling completed")

        return df_scaled

    def prepare_features(self, df: pd.DataFrame, target_column: str = None) -> Tuple:
        """
        Prepare features and target for model training.

        Args:
            df: Input DataFrame
            target_column: Name of the target column (optional)

        Returns:
            Tuple of (features, target) or just features if no target
        """
        logger.info("Preparing features for modeling")

        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            logger.info(f"Prepared {X.shape[1]} features and target '{target_column}'")
            return X, y
        else:
            logger.info(f"Prepared {df.shape[1]} features (no target)")
            return df

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple:
        """
        Split data into training and testing sets.

        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data with test_size={test_size}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info(
            f"Split completed - Train: {len(X_train)}, Test: {len(X_test)}"
        )

        return X_train, X_test, y_train, y_test
