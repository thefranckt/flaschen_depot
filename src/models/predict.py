"""
Prediction Modul für Service-Zeit-Vorhersage
"""
import logging
from pathlib import Path
from typing import Union, List, Dict
import json

import pandas as pd
import numpy as np
import joblib
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceTimePredictor:
    """Klasse für Vorhersagen mit trainiertem Modell"""
    
    def __init__(
        self,
        model_path: str = "models/model.pkl",
        preprocessor_path: str = "models/preprocessor.pkl",
        config_path: str = "configs/config.yaml"
    ):
        """
        Initialisiert den Predictor
        
        Args:
            model_path: Pfad zum trainierten Modell
            preprocessor_path: Pfad zum Preprocessor
            config_path: Pfad zur Konfiguration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.preprocessor = None
        
        self.load_model(model_path)
        self.load_preprocessor(preprocessor_path)
    
    def load_model(self, filepath: str):
        """
        Lädt das trainierte Modell
        
        Args:
            filepath: Pfad zum Modell
        """
        try:
            self.model = joblib.load(filepath)
            logger.info(f"Modell geladen: {filepath}")
        except FileNotFoundError:
            logger.warning(f"Modell nicht gefunden: {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """
        Lädt den Preprocessor
        
        Args:
            filepath: Pfad zum Preprocessor
        """
        try:
            self.preprocessor = joblib.load(filepath)
            logger.info(f"Preprocessor geladen: {filepath}")
        except FileNotFoundError:
            logger.warning(f"Preprocessor nicht gefunden: {filepath}")
    
    def preprocess_input(self, data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        """
        Preprocessiert Eingabedaten
        
        Args:
            data: Eingabedaten als DataFrame oder Dictionary
            
        Returns:
            Preprocessierte Features
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor nicht geladen!")
        
        X_processed = self.preprocessor.transform(data)
        return X_processed
    
    def predict(self, data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        """
        Macht Vorhersagen für Service-Zeit in Minuten
        
        Args:
            data: Eingabedaten
            
        Returns:
            Vorhersagte Service-Zeit in Minuten
        """
        if self.model is None:
            raise ValueError("Modell nicht geladen!")
        
        X = self.preprocess_input(data)
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        """
        Gibt Wahrscheinlichkeiten für Vorhersagen zurück
        
        Args:
            data: Eingabedaten
            
        Returns:
            Wahrscheinlichkeiten für jede Klasse
        """
        if self.model is None:
            raise ValueError("Modell nicht geladen!")
        
        X = self.preprocess_input(data)
        probabilities = self.model.predict_proba(X)
        
        return probabilities
    
    def predict_single(self, order_data: Dict) -> Dict:
        """
        Macht Vorhersage für eine einzelne Bestellung
        
        Args:
            order_data: Dictionary mit Bestellungsdaten
            
        Returns:
            Dictionary mit vorhergesagter Service-Zeit
        """
        prediction = self.predict(order_data)[0]
        
        result = {
            'predicted_service_time_minutes': float(prediction),
            'predicted_service_time_hours': float(prediction / 60),
            'order_data': order_data
        }
        
        logger.info(f"Vorhersage: {prediction:.2f} Minuten")
        return result
    
    def batch_predict(self, data_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Macht Batch-Vorhersagen für CSV-Datei
        
        Args:
            data_path: Pfad zur Eingabedatei
            output_path: Pfad zur Ausgabedatei (optional)
            
        Returns:
            DataFrame mit Vorhersagen
        """
        logger.info(f"Lade Daten von {data_path}")
        df = pd.read_csv(data_path)
        
        # Vorhersagen
        predictions = self.predict(df)
        
        # Ergebnisse hinzufügen
        df['predicted_service_time_minutes'] = predictions
        df['predicted_service_time_hours'] = predictions / 60
        
        # Speichern
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Vorhersagen gespeichert: {output_path}")
        
        return df
    
    def explain_prediction(self, order_data: Dict) -> Dict:
        """
        Erklärt eine Vorhersage (Feature Importance)
        
        Args:
            order_data: Dictionary mit Bestellungsdaten
            
        Returns:
            Dictionary mit Erklärung
        """
        result = self.predict_single(order_data)
        
        # Feature Importance (wenn verfügbar)
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.config['features']['categorical_features'] + \
                          self.config['features']['numerical_features']
            
            importance_dict = dict(zip(
                feature_names,
                self.model.feature_importances_
            ))
            
            # Sortiere nach Wichtigkeit
            sorted_importance = dict(sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            result['feature_importance'] = sorted_importance
        
        return result


def create_sample_prediction():
    """Beispiel für eine Einzelvorhersage"""
    
    sample_order = {
        'warehouse_id': 1,
        'has_elevator': True,
        'floor': 3,
        'is_pre_order': False,
        'is_business': False,
        'num_articles': 15,
        'total_weight_g': 8500,
        'avg_article_weight_g': 567,
        'max_article_weight_g': 1200,
        'hour_of_day': 14,
        'day_of_week': 2,
        'is_weekend': False,
        'month': 5
    }
    
    predictor = ServiceTimePredictor()
    result = predictor.predict_single(sample_order)
    
    print("\n" + "="*50)
    print("Service-Zeit Vorhersage für Beispielbestellung")
    print("="*50)
    print(json.dumps(sample_order, indent=2))
    print("\nErgebnis:")
    print(json.dumps(result, indent=2))
    print("="*50)
    
    return result


if __name__ == "__main__":
    # Beispielvorhersage
    create_sample_prediction()
    
    logger.info("Prediction abgeschlossen!")
