"""
Utility Functions für das Flaschendepot-Projekt
"""
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Any
import joblib


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
):
    """
    Konfiguriert Logging für das Projekt
    
    Args:
        level: Logging Level (DEBUG, INFO, WARNING, ERROR)
        log_file: Pfad zur Log-Datei (optional)
        format_string: Format für Log-Nachrichten
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(file_handler)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Lädt Konfigurationsdatei
    
    Args:
        config_path: Pfad zur Konfigurationsdatei
        
    Returns:
        Dictionary mit Konfiguration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str = "configs/config.yaml"):
    """
    Speichert Konfigurationsdatei
    
    Args:
        config: Konfiguration als Dictionary
        config_path: Pfad zum Speichern
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_metrics(metrics: Dict[str, float], filepath: str):
    """
    Speichert Metriken als JSON
    
    Args:
        metrics: Dictionary mit Metriken
        filepath: Pfad zum Speichern
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath: str) -> Dict[str, float]:
    """
    Lädt Metriken von JSON-Datei
    
    Args:
        filepath: Pfad zur Metrik-Datei
        
    Returns:
        Dictionary mit Metriken
    """
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


def save_artifact(obj: Any, filepath: str):
    """
    Speichert beliebiges Objekt mit joblib
    
    Args:
        obj: Zu speicherndes Objekt
        filepath: Pfad zum Speichern
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)


def load_artifact(filepath: str) -> Any:
    """
    Lädt Objekt mit joblib
    
    Args:
        filepath: Pfad zur Datei
        
    Returns:
        Geladenes Objekt
    """
    return joblib.load(filepath)


def ensure_dir(directory: str):
    """
    Stellt sicher, dass ein Verzeichnis existiert
    
    Args:
        directory: Pfad zum Verzeichnis
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    Gibt das Projekt-Root-Verzeichnis zurück
    
    Returns:
        Path zum Projekt-Root
    """
    return Path(__file__).parent.parent.parent


if __name__ == "__main__":
    # Test utilities
    setup_logging(level="INFO")
    
    logger = logging.getLogger(__name__)
    logger.info("Utility functions loaded successfully!")
    
    # Test config
    config = load_config()
    logger.info(f"Config loaded: {config['project']['name']}")
