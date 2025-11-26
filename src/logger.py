"""
Logging-Modul
Behandelt Feature- und Vorhersage-Logging in SQLite-Datenbanken.
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureLogger:
    """Loggt Features, die für Vorhersagen verwendet werden."""
    
    def __init__(self, db_path: str = "logs/feature_store.db"):
        """
        FeatureLogger initialisieren.
        
        Args:
            db_path: Pfad zur SQLite-Datenbankdatei
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
    
    def _initialize_db(self):
        """Create feature logging table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                web_order_id TEXT NOT NULL,
                driver_id TEXT NOT NULL,
                features TEXT NOT NULL,
                model_version TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Feature logger initialized at {self.db_path}")
    
    def log_features(
        self,
        web_order_id: str,
        driver_id: str,
        features: Dict,
        model_version: Optional[str] = None
    ):
        """
        Log features for a single prediction.
        
        Args:
            web_order_id: Order identifier
            driver_id: Driver identifier
            features: Dictionary of feature values
            model_version: Version of model used
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        features_json = json.dumps(features)
        
        cursor.execute("""
            INSERT INTO feature_logs (timestamp, web_order_id, driver_id, features, model_version)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, web_order_id, driver_id, features_json, model_version))
        
        conn.commit()
        conn.close()
    
    def log_batch_features(
        self,
        web_order_ids: List[str],
        driver_ids: List[str],
        features_list: List[Dict],
        model_version: Optional[str] = None
    ):
        """
        Log features for batch predictions.
        
        Args:
            web_order_ids: List of order identifiers
            driver_ids: List of driver identifiers
            features_list: List of feature dictionaries
            model_version: Version of model used
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        data = [
            (timestamp, woid, did, json.dumps(feats), model_version)
            for woid, did, feats in zip(web_order_ids, driver_ids, features_list)
        ]
        
        cursor.executemany("""
            INSERT INTO feature_logs (timestamp, web_order_id, driver_id, features, model_version)
            VALUES (?, ?, ?, ?, ?)
        """, data)
        
        conn.commit()
        conn.close()
        logger.info(f"Logged {len(data)} feature records")
    
    def get_features(
        self,
        web_order_id: Optional[str] = None,
        driver_id: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Retrieve logged features.
        
        Args:
            web_order_id: Filter by order ID
            driver_id: Filter by driver ID
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with logged features
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM feature_logs WHERE 1=1"
        params = []
        
        if web_order_id:
            query += " AND web_order_id = ?"
            params.append(web_order_id)
        
        if driver_id:
            query += " AND driver_id = ?"
            params.append(driver_id)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df


class PredictionLogger:
    """Logs predictions and their metadata."""
    
    def __init__(self, db_path: str = "logs/prediction_store.db"):
        """
        Initialize PredictionLogger.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
    
    def _initialize_db(self):
        """Create prediction logging table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                web_order_id TEXT NOT NULL,
                driver_id TEXT NOT NULL,
                predicted_service_time REAL NOT NULL,
                model_version TEXT,
                request_id TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Prediction logger initialized at {self.db_path}")
    
    def log_prediction(
        self,
        web_order_id: str,
        driver_id: str,
        predicted_service_time: float,
        model_version: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """
        Log a single prediction.
        
        Args:
            web_order_id: Order identifier
            driver_id: Driver identifier
            predicted_service_time: Predicted service time in minutes
            model_version: Version of model used
            request_id: Unique request identifier
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO prediction_logs 
            (timestamp, web_order_id, driver_id, predicted_service_time, model_version, request_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, web_order_id, driver_id, predicted_service_time, model_version, request_id))
        
        conn.commit()
        conn.close()
    
    def log_batch_predictions(
        self,
        web_order_ids: List[str],
        driver_ids: List[str],
        predictions: List[float],
        model_version: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """
        Log batch predictions.
        
        Args:
            web_order_ids: List of order identifiers
            driver_ids: List of driver identifiers
            predictions: List of predicted service times
            model_version: Version of model used
            request_id: Unique request identifier
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        data = [
            (timestamp, woid, did, pred, model_version, request_id)
            for woid, did, pred in zip(web_order_ids, driver_ids, predictions)
        ]
        
        cursor.executemany("""
            INSERT INTO prediction_logs 
            (timestamp, web_order_id, driver_id, predicted_service_time, model_version, request_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, data)
        
        conn.commit()
        conn.close()
        logger.info(f"Logged {len(data)} prediction records")
    
    def get_predictions(
        self,
        web_order_id: Optional[str] = None,
        driver_id: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Retrieve logged predictions.
        
        Args:
            web_order_id: Filter by order ID
            driver_id: Filter by driver ID
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with logged predictions
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM prediction_logs WHERE 1=1"
        params = []
        
        if web_order_id:
            query += " AND web_order_id = ?"
            params.append(web_order_id)
        
        if driver_id:
            query += " AND driver_id = ?"
            params.append(driver_id)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_statistics(self) -> Dict:
        """
        Get summary statistics of predictions.
        
        Returns:
            Dictionary with prediction statistics
        """
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        
        # Total predictions
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM prediction_logs")
        stats['total_predictions'] = cursor.fetchone()[0]
        
        # Average prediction
        cursor.execute("SELECT AVG(predicted_service_time) FROM prediction_logs")
        stats['avg_prediction'] = cursor.fetchone()[0]
        
        # By model version
        cursor.execute("""
            SELECT model_version, COUNT(*), AVG(predicted_service_time)
            FROM prediction_logs
            GROUP BY model_version
        """)
        stats['by_model_version'] = cursor.fetchall()
        
        conn.close()
        
        return stats


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test feature logger
    feature_logger = FeatureLogger()
    feature_logger.log_features(
        web_order_id="12345",
        driver_id="D001",
        features={"floor": 3, "has_elevator": 1, "total_weight_g": 5000},
        model_version="v1"
    )
    
    # Test prediction logger
    prediction_logger = PredictionLogger()
    prediction_logger.log_prediction(
        web_order_id="12345",
        driver_id="D001",
        predicted_service_time=12.5,
        model_version="v1",
        request_id="req_001"
    )
    
    print("Logging test complete!")
