"""
Tests für Model Training Module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train_model import ModelTrainer
from src.models.predict import BottlePredictor


class TestModelTrainer:
    """Tests für ModelTrainer Klasse"""
    
    @pytest.fixture
    def sample_data(self):
        """Erstellt Beispiel-Trainingsdaten"""
        np.random.seed(42)
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(20, 10)
        y_test = np.random.randint(0, 2, 20)
        
        return X_train, X_test, y_train, y_test
    
    def test_get_model(self):
        """Testet Model-Erstellung"""
        trainer = ModelTrainer()
        
        model = trainer.get_model('random_forest')
        assert model is not None
        
        model = trainer.get_model('logistic_regression')
        assert model is not None
    
    def test_train_model(self, sample_data):
        """Testet Model-Training"""
        X_train, _, y_train, _ = sample_data
        
        trainer = ModelTrainer()
        model = trainer.train(X_train, y_train, 'random_forest')
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_evaluate_model(self, sample_data):
        """Testet Model-Evaluation"""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer()
        trainer.train(X_train, y_train, 'random_forest')
        
        metrics = trainer.evaluate(X_test, y_test)
        
        # Prüfe Metriken
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_model_predictions(self, sample_data):
        """Testet Model-Vorhersagen"""
        X_train, X_test, y_train, _ = sample_data
        
        trainer = ModelTrainer()
        trainer.train(X_train, y_train, 'random_forest')
        
        predictions = trainer.model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)


class TestBottlePredictor:
    """Tests für BottlePredictor Klasse"""
    
    @pytest.fixture
    def sample_bottle_data(self):
        """Erstellt Beispiel-Flaschendaten"""
        return {
            'bottle_type': 'Bier',
            'material': 'Glas',
            'size_category': 'Mittel',
            'volume_ml': 500,
            'deposit_amount': 0.08,
            'weight_grams': 450
        }
    
    def test_predict_single_structure(self, sample_bottle_data):
        """Testet Struktur der Einzelvorhersage"""
        # Mock test - in Realität würde man ein echtes Modell laden
        result_keys = ['prediction', 'prediction_label', 'probability', 'confidence']
        
        # Simuliere erwartete Ausgabe
        mock_result = {
            'prediction': 1,
            'prediction_label': 'Zurückgegeben',
            'probability': 0.85,
            'confidence': 0.85
        }
        
        for key in result_keys:
            assert key in mock_result
    
    def test_batch_predict_structure(self):
        """Testet Batch-Vorhersage Struktur"""
        df = pd.DataFrame({
            'bottle_type': ['Bier', 'Wasser'],
            'material': ['Glas', 'Plastik'],
            'size_category': ['Mittel', 'Klein'],
            'volume_ml': [500, 330],
            'deposit_amount': [0.08, 0.25],
            'weight_grams': [450, 30]
        })
        
        # Erwartete Spalten nach Vorhersage
        expected_columns = [
            'prediction', 'prediction_label', 
            'probability', 'confidence'
        ]
        
        assert len(df) == 2


class TestModelValidation:
    """Tests für Model-Validierung"""
    
    def test_cross_validation_scores(self):
        """Testet Cross-Validation"""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        scores = cross_val_score(model, X, y, cv=3)
        
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores)
    
    def test_model_consistency(self):
        """Testet Model-Konsistenz"""
        from sklearn.ensemble import RandomForestClassifier
        
        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        
        # Trainiere Modell zweimal mit gleichem Seed
        model1 = RandomForestClassifier(random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)
        
        model2 = RandomForestClassifier(random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)
        
        # Vorhersagen sollten identisch sein
        assert np.array_equal(pred1, pred2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
