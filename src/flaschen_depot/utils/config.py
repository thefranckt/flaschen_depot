"""
Configuration loader for Flaschen Depot project.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Class for loading and managing configuration.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize ConfigLoader.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return {}

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Configuration loaded from {self.config_path}")
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'model.name')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get("model", {})

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get("data", {})

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return self.config.get("hyperparameters", {})
