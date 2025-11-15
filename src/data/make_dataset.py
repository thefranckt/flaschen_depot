"""
Datenverarbeitungs-Modul für Getränkelieferservice-Daten
Lädt und verarbeitet 4 Parquet-Dateien:
- articles.parquet: Artikel-Informationen mit Gewichten
- orders.parquet: Bestellinformationen mit Kontext
- driver_order_mapping.parquet: Zuordnung Fahrer zu Bestellungen
- service_times.parquet: Service-Zeiten (Zielvariable)
"""
import logging
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

# Logger konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Klasse zum Laden und Vorbereiten von Getränkelieferservice-Daten"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialisiert den DataLoader
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_path = Path(self.config['data']['raw_data_path'])
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        self.test_size = self.config['data']['test_size']
        self.random_state = self.config['data']['random_state']
        
        # Parquet-Dateinamen
        self.parquet_files = {
            'articles': 'articles.parquet',
            'orders': 'orders.parquet',
            'driver_mapping': 'driver_order_mapping.parquet',
            'service_times': 'service_times.parquet'
        }
    
    def load_parquet_files(self) -> Dict[str, pd.DataFrame]:
        """
        Lädt alle Parquet-Dateien
        
        Returns:
            Dictionary mit DataFrames
        """
        logger.info("Lade Parquet-Dateien...")
        
        dataframes = {}
        
        for key, filename in self.parquet_files.items():
            filepath = self.raw_data_path / filename
            logger.info(f"  Lade {filename}...")
            
            try:
                df = pd.read_parquet(filepath)
                dataframes[key] = df
                logger.info(f"    ✓ {len(df):,} Zeilen geladen")
            except FileNotFoundError:
                logger.error(f"  ✗ Datei nicht gefunden: {filepath}")
                raise
            except Exception as e:
                logger.error(f"  ✗ Fehler beim Laden: {e}")
                raise
        
        return dataframes
    
    def merge_datasets(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merged alle Datasets zusammen
        
        Args:
            dataframes: Dictionary mit den einzelnen DataFrames
            
        Returns:
            Zusammengeführter DataFrame
        """
        logger.info("Merge Datasets...")
        
        # Start mit service_times (hat die Zielvariable)
        df = dataframes['service_times'].copy()
        logger.info(f"  Start: service_times ({len(df):,} Zeilen)")
        
        # Merge mit orders
        df = df.merge(
            dataframes['orders'],
            on='web_order_id',
            how='left',
            suffixes=('', '_order')
        )
        logger.info(f"  Nach orders merge: {len(df):,} Zeilen")
        
        # Aggregiere articles pro Bestellung
        articles = dataframes['articles'].copy()
        
        # Berechne Artikel-Statistiken pro Bestellung
        article_stats = articles.groupby('web_order_id').agg({
            'article_id': 'count',  # Anzahl Artikel
            'article_weight_in_g': ['sum', 'mean', 'max']  # Gewichtsstatistiken
        }).reset_index()
        
        # Flatten column names
        article_stats.columns = [
            'web_order_id',
            'num_articles',
            'total_weight_g',
            'avg_article_weight_g',
            'max_article_weight_g'
        ]
        
        # Merge mit article stats
        df = df.merge(article_stats, on='web_order_id', how='left')
        logger.info(f"  Nach articles merge: {len(df):,} Zeilen")
        
        logger.info(f"✓ Finaler DataFrame: {len(df):,} Zeilen, {len(df.columns)} Spalten")
        
        return df
    
    def load_raw_data(self, filename: str = None) -> pd.DataFrame:
        """
        Lädt und merged alle Rohdaten
        
        Args:
            filename: Wird ignoriert (für Kompatibilität)
            
        Returns:
            DataFrame mit allen zusammengeführten Daten
        """
        # Lade alle Parquet-Dateien
        dataframes = self.load_parquet_files()
        
        # Merge zusammen
        df = self.merge_datasets(dataframes)
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bereinigt die Daten
        
        Args:
            df: Eingabe DataFrame
            
        Returns:
            Bereinigter DataFrame
        """
        logger.info("Starte Datenbereinigung")
        initial_rows = len(df)
        
        # Entferne Zeilen ohne service_time (Zielvariable)
        df = df[df['service_time_in_minutes'].notna()].copy()
        logger.info(f"  {initial_rows - len(df):,} Zeilen ohne service_time entfernt")
        
        # Entferne unrealistische Service-Zeiten (< 0 oder > 6 Stunden)
        df = df[
            (df['service_time_in_minutes'] > 0) & 
            (df['service_time_in_minutes'] <= 360)
        ].copy()
        logger.info(f"  Unrealistische Service-Zeiten gefiltert")
        
        # Duplikate entfernen
        dup_count = df.duplicated(subset=['web_order_id']).sum()
        df = df.drop_duplicates(subset=['web_order_id'])
        logger.info(f"  {dup_count:,} Duplikate entfernt")
        
        # Fill NaN für floor mit 0 (Erdgeschoss)
        df['floor'] = df['floor'].fillna(0)
        
        # Fill NaN für Artikel-Features mit 0
        article_cols = ['num_articles', 'total_weight_g', 'avg_article_weight_g', 'max_article_weight_g']
        for col in article_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        logger.info(f"✓ Bereinigung abgeschlossen: {len(df):,} Zeilen")
        
        return df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_column: str = 'service_time_in_minutes'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Teilt Daten in Train- und Test-Sets
        
        Args:
            df: Eingabe DataFrame
            target_column: Name der Zielvariable
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Teile Daten in Train und Test")
        
        # Features und Target trennen
        feature_cols = [col for col in df.columns if col != target_column]
        X = df[feature_cols]
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        logger.info(f"  Training Set: {len(X_train):,} Samples")
        logger.info(f"  Test Set: {len(X_test):,} Samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ):
        """
        Speichert verarbeitete Daten
        
        Args:
            X_train, X_test, y_train, y_test: Daten-Splits
        """
        logger.info("Speichere verarbeitete Daten")
        
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        X_train.to_csv(self.processed_data_path / 'X_train.csv', index=False)
        X_test.to_csv(self.processed_data_path / 'X_test.csv', index=False)
        y_train.to_csv(self.processed_data_path / 'y_train.csv', index=False)
        y_test.to_csv(self.processed_data_path / 'y_test.csv', index=False)
        
        logger.info("Daten erfolgreich gespeichert")


def generate_sample_data(output_path: str = "data/raw/bottles.csv", n_samples: int = 1000):
    """
    DEPRECATED: Wird nicht mehr verwendet, da echte Parquet-Dateien vorhanden sind
    """
    logger.warning("generate_sample_data() ist deprecated - nutze die echten Parquet-Dateien!")
    pass


if __name__ == "__main__":
    # DataLoader testen mit echten Parquet-Dateien
    loader = DataLoader()
    
    # Lade alle Daten
    df = loader.load_raw_data()
    logger.info(f"\n{'='*60}")
    logger.info(f"Geladene Daten: {len(df):,} Zeilen, {len(df.columns)} Spalten")
    logger.info(f"{'='*60}")
    
    # Bereinige Daten
    df_clean = loader.clean_data(df)
    
    # Split
    X_train, X_test, y_train, y_test = loader.split_data(df_clean)
    
    # Speichere
    loader.save_processed_data(X_train, X_test, y_train, y_test)
    
    logger.info("\n✓ Datenverarbeitung abgeschlossen!")
    logger.info(f"  Zielvariable: service_time_in_minutes")
    logger.info(f"  Min: {y_train.min():.2f} min")
    logger.info(f"  Max: {y_train.max():.2f} min")
    logger.info(f"  Mean: {y_train.mean():.2f} min")
    logger.info(f"  Median: {y_train.median():.2f} min")
