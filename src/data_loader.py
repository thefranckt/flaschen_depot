"""
Daten-Lade-Modul
Lädt und validiert Parquet-Dateien für das Service-Zeit-Vorhersage-Projekt.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class DataLoader:
    """Behandelt das Laden und die initiale Validierung von Rohdaten-Dateien."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        DataLoader mit Datenverzeichnis initialisieren.
        
        Args:
            data_dir: Pfad zum Verzeichnis mit rohen Parquet-Dateien
        """
        self.data_dir = Path(data_dir)
        
    def load_orders(self) -> pd.DataFrame:
        """
        Bestellungen laden.
        
        Returns:
            DataFrame mit Spalten: warehouse_id, order_time, has_elevator, 
            floor, is_business, web_order_id, customer_id
        """
        filepath = self.data_dir / "orders.parquet"
        logger.info(f"Loading orders from {filepath}")
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} orders")
        return df
    
    def load_articles(self) -> pd.DataFrame:
        """
        Artikel laden.
        
        Returns:
            DataFrame mit Spalten: warehouse_id, box_id, article_id, 
            article_weight_in_g, web_order_id
        """
        filepath = self.data_dir / "articles.parquet"
        logger.info(f"Loading articles from {filepath}")
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} articles")
        return df
    
    def load_service_times(self) -> pd.DataFrame:
        """
        Service-Zeiten laden.
        
        Returns:
            DataFrame mit Spalten: service_time_start, service_time_end, 
            service_time_in_minutes, order_datetime, web_order_id, driver_id, 
            trip_id, customer_id
        """
        filepath = self.data_dir / "service_times.parquet"
        logger.info(f"Loading service times from {filepath}")
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} service time records")
        return df
    
    def load_driver_mapping(self) -> pd.DataFrame:
        """
        Fahrer-Bestellungs-Zuordnung laden.
        
        Returns:
            DataFrame mit Spalten: driver_id, web_order_id
        """
        filepath = self.data_dir / "driver_order_mapping.parquet"
        logger.info(f"Loading driver mapping from {filepath}")
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} driver-order mappings")
        return df
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Alle Datendateien laden.
        
        Returns:
            Tuple von (orders, articles, service_times, driver_mapping) DataFrames
        """
        logger.info("Loading all data files...")
        orders = self.load_orders()
        articles = self.load_articles()
        service_times = self.load_service_times()
        driver_mapping = self.load_driver_mapping()
        logger.info("All data files loaded successfully")
        return orders, articles, service_times, driver_mapping
    
    def get_data_summary(self) -> Dict[str, Dict]:
        """
        Zusammenfassungs-Statistiken für alle Datensätze abrufen.
        
        Returns:
            Dictionary mit Zusammenfassungs-Info für jeden Datensatz
        """
        orders, articles, service_times, driver_mapping = self.load_all()
        
        summary = {
            "orders": {
                "rows": len(orders),
                "columns": list(orders.columns),
                "missing_values": orders.isnull().sum().to_dict()
            },
            "articles": {
                "rows": len(articles),
                "columns": list(articles.columns),
                "missing_values": articles.isnull().sum().to_dict()
            },
            "service_times": {
                "rows": len(service_times),
                "columns": list(service_times.columns),
                "missing_values": service_times.isnull().sum().to_dict()
            },
            "driver_mapping": {
                "rows": len(driver_mapping),
                "columns": list(driver_mapping.columns),
                "missing_values": driver_mapping.isnull().sum().to_dict()
            }
        }
        
        return summary


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test data loading
    loader = DataLoader()
    summary = loader.get_data_summary()
    
    print("\nData Summary:")
    for dataset_name, info in summary.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Rows: {info['rows']}")
        print(f"  Columns: {info['columns']}")
        print(f"  Missing values: {info['missing_values']}")
