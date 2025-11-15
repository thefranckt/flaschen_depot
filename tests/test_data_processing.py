"""
Tests für Data Processing Module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Füge src zum Python Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.make_dataset import DataLoader, generate_sample_data
from src.features.build_features import FeatureEngineer


class TestDataLoader:
    """Tests für DataLoader Klasse"""
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Erstellt Beispieldaten für Tests"""
        df = pd.DataFrame({
            'bottle_type': ['Bier', 'Wasser', 'Saft'],
            'material': ['Glas', 'Plastik', 'Glas'],
            'size_category': ['Mittel', 'Klein', 'Groß'],
            'volume_ml': [500, 330, 1000],
            'deposit_amount': [0.08, 0.25, 0.15],
            'weight_grams': [450, 30, 600],
            'return_status': [1, 0, 1]
        })
        
        # Speichere temporäre Datei
        filepath = tmp_path / "test_data.csv"
        df.to_csv(filepath, index=False)
        
        return df, filepath
    
    def test_load_raw_data(self, sample_data):
        """Testet das Laden von Rohdaten"""
        df_expected, filepath = sample_data
        
        # Simuliere DataLoader mit angepasstem Pfad
        df_loaded = pd.read_csv(filepath)
        
        assert len(df_loaded) == len(df_expected)
        assert list(df_loaded.columns) == list(df_expected.columns)
    
    def test_clean_data(self):
        """Testet Datenbereinigung"""
        # Daten mit Duplikaten und NaN
        df = pd.DataFrame({
            'A': [1, 2, 2, np.nan, 5],
            'B': [10, 20, 20, 40, 50],
            'C': [100, 200, 200, 400, np.nan]
        })
        
        loader = DataLoader()
        df_clean = loader.clean_data(df)
        
        # Prüfe: keine Duplikate, keine NaN
        assert df_clean.duplicated().sum() == 0
        assert df_clean.isnull().sum().sum() == 0
        assert len(df_clean) == 2  # Nur 2 saubere Zeilen übrig
    
    def test_split_data(self, sample_data):
        """Testet Train-Test Split"""
        df, _ = sample_data
        
        loader = DataLoader()
        X_train, X_test, y_train, y_test = loader.split_data(df, 'return_status')
        
        # Prüfe Shapes
        assert len(X_train) + len(X_test) == len(df)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        assert 'return_status' not in X_train.columns


class TestFeatureEngineer:
    """Tests für FeatureEngineer Klasse"""
    
    @pytest.fixture
    def sample_features(self):
        """Erstellt Beispiel-Features"""
        return pd.DataFrame({
            'bottle_type': ['Bier', 'Wasser', 'Saft'],
            'material': ['Glas', 'Plastik', 'Glas'],
            'size_category': ['Mittel', 'Klein', 'Groß'],
            'volume_ml': [500, 330, 1000],
            'deposit_amount': [0.08, 0.25, 0.15],
            'weight_grams': [450, 30, 600]
        })
    
    def test_create_features(self, sample_features):
        """Testet Feature-Erstellung"""
        fe = FeatureEngineer()
        df_enhanced = fe.create_features(sample_features)
        
        # Prüfe neue Features
        assert 'deposit_per_ml' in df_enhanced.columns
        assert 'weight_per_ml' in df_enhanced.columns
        assert len(df_enhanced.columns) > len(sample_features.columns)
    
    def test_preprocessor(self, sample_features):
        """Testet Preprocessor Pipeline"""
        fe = FeatureEngineer()
        
        X_transformed = fe.fit_transform(sample_features)
        
        # Prüfe Transformation
        assert isinstance(X_transformed, np.ndarray)
        assert len(X_transformed) == len(sample_features)
    
    def test_preprocessor_transform(self, sample_features):
        """Testet Transform nach Fit"""
        fe = FeatureEngineer()
        
        # Fit auf den Daten
        X_train = fe.fit_transform(sample_features)
        
        # Transform auf neuen Daten
        X_test = fe.transform(sample_features.iloc[:1])
        
        assert X_test.shape[1] == X_train.shape[1]


class TestGenerateSampleData:
    """Tests für Datengenerierung"""
    
    def test_generate_sample_data(self, tmp_path):
        """Testet Sample-Daten Generierung"""
        output_path = tmp_path / "test_bottles.csv"
        
        df = generate_sample_data(str(output_path), n_samples=100)
        
        # Prüfe Generierung
        assert len(df) == 100
        assert output_path.exists()
        
        # Prüfe Spalten
        expected_columns = [
            'bottle_type', 'material', 'size_category',
            'volume_ml', 'deposit_amount', 'weight_grams', 'return_status'
        ]
        assert list(df.columns) == expected_columns
    
    def test_data_ranges(self, tmp_path):
        """Testet Datenbereiche"""
        output_path = tmp_path / "test_bottles.csv"
        df = generate_sample_data(str(output_path), n_samples=50)
        
        # Prüfe Wertebereiche
        assert df['volume_ml'].min() >= 250
        assert df['volume_ml'].max() <= 1500
        assert df['return_status'].isin([0, 1]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
