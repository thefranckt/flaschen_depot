"""
Feature Engineering Module für Getränkelieferservice Service-Time-Prediction
"""
import logging
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Klasse für Feature Engineering - Service-Time-Prediction"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialisiert den FeatureEngineer
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.categorical_features = self.config['features']['categorical_features']
        self.numerical_features = self.config['features']['numerical_features']
        self.temporal_features = self.config['features']['temporal_features']
        self.preprocessor = None
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt zeitliche Features aus Timestamps
        
        Args:
            df: Eingabe DataFrame
            
        Returns:
            DataFrame mit zeitlichen Features
        """
        logger.info("Erstelle zeitliche Features")
        
        df_copy = df.copy()
        
        # Konvertiere zu datetime falls nötig
        if 'order_datetime' in df_copy.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_copy['order_datetime']):
                df_copy['order_datetime'] = pd.to_datetime(df_copy['order_datetime'])
            
            # Extrahiere zeitliche Features
            df_copy['hour_of_day'] = df_copy['order_datetime'].dt.hour
            df_copy['day_of_week'] = df_copy['order_datetime'].dt.dayofweek  # 0=Monday
            df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)
            df_copy['month'] = df_copy['order_datetime'].dt.month
            
            logger.info("  ✓ Zeitliche Features erstellt: hour_of_day, day_of_week, is_weekend, month")
        
        return df_copy
    
    def create_delivery_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt lieferungs-spezifische Features
        
        Args:
            df: Eingabe DataFrame
            
        Returns:
            DataFrame mit zusätzlichen Features
        """
        logger.info("Erstelle Lieferungs-Features")
        
        df_copy = df.copy()
        
        # Gewicht pro Artikel (Durchschnitt)
        if 'total_weight_g' in df_copy.columns and 'num_articles' in df_copy.columns:
            df_copy['weight_per_article'] = df_copy['total_weight_g'] / (df_copy['num_articles'] + 1)
        
        # Gewicht in kg (besser interpretierbar)
        if 'total_weight_g' in df_copy.columns:
            df_copy['total_weight_kg'] = df_copy['total_weight_g'] / 1000
        
        # Schwierigkeits-Score (kombiniert Stockwerk und Gewicht)
        if 'floor' in df_copy.columns and 'total_weight_kg' in df_copy.columns:
            df_copy['difficulty_score'] = (
                df_copy['floor'] * 0.5 +  # Stockwerk hat moderaten Einfluss
                df_copy['total_weight_kg'] * 0.1  # Gewicht hat kleineren Einfluss
            )
            
            # Bonus für Aufzug
            if 'has_elevator' in df_copy.columns:
                df_copy['difficulty_score'] = df_copy['difficulty_score'] * (
                    1 - df_copy['has_elevator'].astype(int) * 0.3
                )
        
        # Artikel-Komplexität
        if 'num_articles' in df_copy.columns:
            # Kategorisiere Anzahl Artikel
            df_copy['order_size_category'] = pd.cut(
                df_copy['num_articles'],
                bins=[0, 5, 15, 30, 1000],
                labels=['small', 'medium', 'large', 'very_large']
            )
        
        logger.info(f"  ✓ {len([c for c in df_copy.columns if c not in df.columns])} neue Features erstellt")
        
        return df_copy
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt alle zusätzlichen Features
        
        Args:
            df: Eingabe DataFrame
            
        Returns:
            DataFrame mit zusätzlichen Features
        """
        logger.info("Starte Feature-Erstellung")
        
        df_copy = df.copy()
        
        # Zeitliche Features
        df_copy = self.create_temporal_features(df_copy)
        
        # Lieferungs-Features
        df_copy = self.create_delivery_features(df_copy)
        
        logger.info(f"✓ Feature-Erstellung abgeschlossen: {len(df_copy.columns)} Spalten")
        
        return df_copy
    
    def select_features_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wählt nur die Features aus, die für Training verwendet werden
        
        Args:
            df: DataFrame mit allen Features
            
        Returns:
            DataFrame nur mit Training-Features
        """
        # Alle Features die wir verwenden wollen
        feature_cols = (
            self.categorical_features +
            self.numerical_features +
            self.temporal_features +
            ['total_weight_kg', 'difficulty_score', 'order_size_category']
        )
        
        # Nur vorhandene Features auswählen
        available_cols = [col for col in feature_cols if col in df.columns]
        
        logger.info(f"Wähle {len(available_cols)} Features für Training")
        
        return df[available_cols]
    
    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Erstellt einen Preprocessing-Pipeline
        
        Args:
            X: Feature DataFrame
            
        Returns:
            ColumnTransformer für Preprocessing
        """
        logger.info("Erstelle Preprocessing Pipeline")
        
        # Identifiziere tatsächlich vorhandene Spalten
        cat_features = [f for f in self.categorical_features if f in X.columns]
        num_features = [f for f in self.numerical_features if f in X.columns]
        temporal_features = [f for f in self.temporal_features if f in X.columns]
        
        # Zusätzliche numerische Features
        additional_num = ['total_weight_kg', 'difficulty_score']
        num_features.extend([f for f in additional_num if f in X.columns])
        
        # Zusätzliche kategorische Features
        if 'order_size_category' in X.columns:
            cat_features.append('order_size_category')
        
        logger.info(f"  Kategorische Features: {len(cat_features)}")
        logger.info(f"  Numerische Features: {len(num_features)}")
        logger.info(f"  Zeitliche Features: {len(temporal_features)}")
        
        # Numerische Features: Standardisierung
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Kategorische Features: One-Hot Encoding
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Kombiniere alle
        transformers = []
        
        if num_features:
            transformers.append(('num', numeric_transformer, num_features))
        if temporal_features:
            transformers.append(('temporal', numeric_transformer, temporal_features))
        if cat_features:
            transformers.append(('cat', categorical_transformer, cat_features))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fitted und transformiert Features
        
        Args:
            X: Eingabe DataFrame
            
        Returns:
            Transformierte Features als numpy array
        """
        if self.preprocessor is None:
            self.build_preprocessor(X)
        
        logger.info("Fitte und transformiere Features")
        X_transformed = self.preprocessor.fit_transform(X)
        
        logger.info(f"  ✓ Output Shape: {X_transformed.shape}")
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transformiert Features mit gefittetem Preprocessor
        
        Args:
            X: Eingabe DataFrame
            
        Returns:
            Transformierte Features als numpy array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor muss zuerst gefittet werden!")
        
        logger.info("Transformiere Features")
        X_transformed = self.preprocessor.transform(X)
        
        return X_transformed
    
    def save_preprocessor(self, filepath: str = "models/preprocessor.pkl"):
        """
        Speichert den gefitteten Preprocessor
        
        Args:
            filepath: Pfad zum Speichern
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor muss zuerst gefittet werden!")
        
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.preprocessor, filepath)
        logger.info(f"✓ Preprocessor gespeichert: {filepath}")
    
    def load_preprocessor(self, filepath: str = "models/preprocessor.pkl"):
        """
        Lädt einen gespeicherten Preprocessor
        
        Args:
            filepath: Pfad zum Laden
        """
        self.preprocessor = joblib.load(filepath)
        logger.info(f"✓ Preprocessor geladen: {filepath}")


if __name__ == "__main__":
    # Teste Feature Engineering
    from pathlib import Path
    
    logger.info("Teste Feature Engineering mit echten Daten")
    
    # Lade Daten (falls vorhanden)
    data_path = Path("data/processed")
    
    if (data_path / "X_train.csv").exists():
        X_train = pd.read_csv(data_path / "X_train.csv")
        X_test = pd.read_csv(data_path / "X_test.csv")
        
        # Feature Engineering
        fe = FeatureEngineer()
        
        # Erstelle Features
        X_train_enhanced = fe.create_features(X_train)
        X_test_enhanced = fe.create_features(X_test)
        
        # Wähle Features
        X_train_selected = fe.select_features_for_training(X_train_enhanced)
        X_test_selected = fe.select_features_for_training(X_test_enhanced)
        
        # Preprocessing
        X_train_processed = fe.fit_transform(X_train_selected)
        X_test_processed = fe.transform(X_test_selected)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Train Features Shape: {X_train_processed.shape}")
        logger.info(f"Test Features Shape: {X_test_processed.shape}")
        logger.info(f"{'='*60}")
        
        # Speichere Preprocessor
        fe.save_preprocessor()
        
        logger.info("\n✓ Feature Engineering abgeschlossen!")
    else:
        logger.warning("Keine verarbeiteten Daten gefunden. Führe zuerst make_dataset.py aus!")
