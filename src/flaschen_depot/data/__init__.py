"""
Data ingestion module for Flaschen Depot project.
Handles loading and initial processing of bottle depot data.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Class for ingesting data from various sources.
    """

    def __init__(self, data_path: str = "data/raw"):
        """
        Initialize DataIngestion with data path.

        Args:
            data_path: Path to raw data directory
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataIngestion initialized with path: {self.data_path}")

    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filename: Name of the CSV file

        Returns:
            DataFrame with loaded data
        """
        filepath = self.data_path / filename
        logger.info(f"Loading data from {filepath}")
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded {len(df)} rows from {filename}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def create_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create sample bottle depot data for testing.

        Args:
            n_samples: Number of sample records to generate

        Returns:
            DataFrame with sample bottle depot data
        """
        import numpy as np

        logger.info(f"Generating {n_samples} sample records")

        # Generate sample data for bottle depot
        data = {
            "bottle_id": range(1, n_samples + 1),
            "bottle_type": np.random.choice(["PET", "Glass", "Aluminum"], n_samples),
            "volume_ml": np.random.choice([330, 500, 750, 1000], n_samples),
            "deposit_amount": np.random.uniform(0.08, 0.25, n_samples).round(2),
            "condition": np.random.choice(["excellent", "good", "fair", "poor"], n_samples),
            "return_count": np.random.randint(0, 50, n_samples),
            "last_return_days": np.random.randint(1, 365, n_samples),
        }

        df = pd.DataFrame(data)
        logger.info("Sample data generated successfully")
        return df

    def save_data(self, df: pd.DataFrame, filename: str, processed: bool = False) -> None:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame to save
            filename: Name of the output file
            processed: Whether to save in processed directory
        """
        save_path = Path("data/processed" if processed else self.data_path)
        save_path.mkdir(parents=True, exist_ok=True)
        filepath = save_path / filename

        logger.info(f"Saving data to {filepath}")
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved successfully: {len(df)} rows")
