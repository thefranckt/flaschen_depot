"""
Test suite for data ingestion module.
"""

import pandas as pd
import pytest

from flaschen_depot.data import DataIngestion


@pytest.fixture
def data_ingestion():
    """Create DataIngestion instance for testing."""
    return DataIngestion(data_path="/tmp/test_data")


def test_data_ingestion_init(data_ingestion):
    """Test DataIngestion initialization."""
    assert data_ingestion.data_path.exists()


def test_create_sample_data(data_ingestion):
    """Test sample data creation."""
    df = data_ingestion.create_sample_data(n_samples=100)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "bottle_id" in df.columns
    assert "bottle_type" in df.columns
    assert "volume_ml" in df.columns


def test_save_and_load_data(data_ingestion):
    """Test saving and loading data."""
    df = data_ingestion.create_sample_data(n_samples=50)
    filename = "test_bottles.csv"

    # Save data
    data_ingestion.save_data(df, filename)

    # Load data
    df_loaded = data_ingestion.load_csv(filename)

    assert len(df_loaded) == 50
    assert list(df.columns) == list(df_loaded.columns)
