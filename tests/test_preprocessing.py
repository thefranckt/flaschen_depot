"""
Test suite for data preprocessing module.
"""

import pandas as pd
import pytest

from flaschen_depot.data.preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(
        {
            "bottle_type": ["PET", "Glass", "Aluminum", "PET"],
            "volume_ml": [500, 750, 330, 1000],
            "deposit_amount": [0.25, 0.15, 0.08, 0.25],
            "condition": ["good", "excellent", "fair", "good"],
            "return_count": [5, 10, 3, 7],
            "last_return_days": [30, 15, 60, 45],
        }
    )


@pytest.fixture
def preprocessor():
    """Create DataPreprocessor instance for testing."""
    return DataPreprocessor()


def test_clean_data(preprocessor, sample_data):
    """Test data cleaning."""
    df_clean = preprocessor.clean_data(sample_data)

    assert isinstance(df_clean, pd.DataFrame)
    assert len(df_clean) <= len(sample_data)
    assert df_clean.isnull().sum().sum() == 0


def test_encode_categorical(preprocessor, sample_data):
    """Test categorical encoding."""
    df_encoded = preprocessor.encode_categorical(
        sample_data, ["bottle_type", "condition"]
    )

    assert isinstance(df_encoded, pd.DataFrame)
    assert df_encoded["bottle_type"].dtype != object
    assert df_encoded["condition"].dtype != object


def test_split_data(preprocessor, sample_data):
    """Test train-test split."""
    df_encoded = preprocessor.encode_categorical(
        sample_data, ["bottle_type", "condition"]
    )
    X = df_encoded.drop(columns=["condition"])
    y = df_encoded["condition"]

    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X, y, test_size=0.25, random_state=42
    )

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert len(X_test) == 1  # 25% of 4 samples
