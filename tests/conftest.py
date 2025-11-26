"""
Pytest configuration and fixtures.
"""

import pytest


@pytest.fixture(scope="session")
def test_data_path(tmp_path_factory):
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def test_model_path(tmp_path_factory):
    """Create temporary directory for test models."""
    return tmp_path_factory.mktemp("test_models")
