"""
FastAPI Application für Flaschenpfand-Vorhersage
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import logging

from src.models.predict import BottlePredictor

# Logger konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(
    title="Flaschendepot ML API",
    description="API für Flaschenpfand-Rückgabe-Vorhersagen",
    version="0.1.0"
)

# Predictor laden
try:
    predictor = BottlePredictor()
    logger.info("Predictor erfolgreich geladen")
except Exception as e:
    logger.error(f"Fehler beim Laden des Predictors: {e}")
    predictor = None


class BottleData(BaseModel):
    """Schema für Flaschendaten"""
    bottle_type: str
    material: str
    size_category: str
    volume_ml: int
    deposit_amount: float
    weight_grams: int

    class Config:
        schema_extra = {
            "example": {
                "bottle_type": "Bier",
                "material": "Glas",
                "size_category": "Mittel",
                "volume_ml": 500,
                "deposit_amount": 0.08,
                "weight_grams": 450
            }
        }


class PredictionResponse(BaseModel):
    """Schema für Vorhersage-Antwort"""
    prediction: int
    prediction_label: str
    probability: float
    confidence: float


@app.get("/")
async def root():
    """Root Endpoint"""
    return {
        "message": "Flaschendepot ML API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health Check Endpoint"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(bottle_data: BottleData):
    """
    Macht eine Vorhersage für eine Flasche
    
    Args:
        bottle_data: Flaschendaten
        
    Returns:
        Vorhersage mit Wahrscheinlichkeit
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Konvertiere zu Dictionary
        data_dict = bottle_data.dict()
        
        # Mache Vorhersage
        result = predictor.predict_single(data_dict)
        
        return result
    
    except Exception as e:
        logger.error(f"Fehler bei Vorhersage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(bottles: List[BottleData]):
    """
    Macht Vorhersagen für mehrere Flaschen
    
    Args:
        bottles: Liste von Flaschendaten
        
    Returns:
        Liste von Vorhersagen
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        
        for bottle_data in bottles:
            data_dict = bottle_data.dict()
            result = predictor.predict_single(data_dict)
            results.append(result)
        
        return {
            "count": len(results),
            "predictions": results
        }
    
    except Exception as e:
        logger.error(f"Fehler bei Batch-Vorhersage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """
    Gibt Informationen über das geladene Modell zurück
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_class = type(predictor.model).__name__
    
    info = {
        "model_type": model_class,
        "features": {
            "categorical": predictor.config['features']['categorical_features'],
            "numerical": predictor.config['features']['numerical_features']
        }
    }
    
    # Feature Importance (wenn verfügbar)
    if hasattr(predictor.model, 'feature_importances_'):
        feature_names = (
            predictor.config['features']['categorical_features'] +
            predictor.config['features']['numerical_features']
        )
        
        importance = dict(zip(
            feature_names,
            predictor.model.feature_importances_.tolist()
        ))
        
        info['feature_importance'] = importance
    
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
