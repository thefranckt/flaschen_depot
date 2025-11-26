"""
FastAPI Anwendung für Service-Zeit-Vorhersage
Bietet REST API Endpunkte für Batch-Vorhersagen und Modellverwaltung.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import joblib
import yaml
import logging
from pathlib import Path
from datetime import datetime
import uuid

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.logger import FeatureLogger, PredictionLogger

# Logging einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Konfiguration laden
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# App initialisieren
app = FastAPI(
    title="Service-Zeit-Vorhersage API",
    description="API zur Vorhersage von Lieferzeiten",
    version="1.0.0"
)

# CORS Middleware hinzufügen
if config['api'].get('cors_enabled', True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Logger initialisieren
feature_logger = FeatureLogger(config['logging']['feature_log_db'])
prediction_logger = PredictionLogger(config['logging']['prediction_log_db'])

# Globale Variablen für Modelle und Daten
models = {}
processed_data = None
feature_engineer = None


class PredictionRequest(BaseModel):
    """Anfrage-Modell für einzelne oder Batch-Vorhersagen."""
    driver_id: str = Field(..., description="Fahrer-ID")
    web_order_id: str = Field(..., description="Bestell-ID aus dem orders-Datensatz")


class BatchPredictionRequest(BaseModel):
    """Anfrage-Modell für Batch-Vorhersagen."""
    requests: List[PredictionRequest] = Field(..., description="Liste der Vorhersage-Anfragen")
    model_version: Optional[str] = Field("latest", description="Zu verwendende Modellversion")


class PredictionResponse(BaseModel):
    """Antwort-Modell für Vorhersagen."""
    driver_id: str
    web_order_id: str
    predicted_service_time: float
    model_version: str
    timestamp: str
    request_id: str


class BatchPredictionResponse(BaseModel):
    """Antwort-Modell für Batch-Vorhersagen."""
    predictions: List[PredictionResponse]
    total_count: int
    request_id: str
    timestamp: str


class ModelInfo(BaseModel):
    """Modellinformations-Antwort."""
    version: str
    path: str
    loaded: bool


class HealthResponse(BaseModel):
    """Gesundheitscheck-Antwort."""
    status: str
    timestamp: str
    models_loaded: int
    available_versions: List[str]


class ModelMetrics(BaseModel):
    """Modell-Performance-Metriken-Antwort."""
    model_version: str
    model_type: str
    timestamp: str
    training_metrics: Dict
    test_metrics: Dict
    feature_importance: Optional[List[Dict]]


def load_model(version: str = "latest"):
    """
    Modell von der Festplatte laden.
    
    Args:
        version: Modellversion zum Laden
        
    Returns:
        Geladenes Modell
    """
    if version in models:
        return models[version]
    
    model_dir = Path(config['mlflow']['artifact_location'])
    
    if version == "latest":
        model_path = model_dir / "model_latest.joblib"
    else:
        model_path = model_dir / f"model_{version}.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    models[version] = model
    
    return model


def load_processed_data():
    """Verarbeitete Daten für Feature-Lookup laden."""
    global processed_data, feature_engineer
    
    if processed_data is not None:
        return processed_data
    
    processed_dir = Path(config['data']['processed_dir'])
    data_path = processed_dir / 'full_dataset.parquet'
    
    if not data_path.exists():
        logger.warning(f"Processed data not found at {data_path}. Loading raw data...")
        # Load and process raw data
        loader = DataLoader(config['data']['raw_dir'])
        orders, articles, service_times, driver_mapping = loader.load_all()
        
        feature_engineer = FeatureEngineer(random_state=config['model']['random_state'])
        _, _, _, processed_data = feature_engineer.process_pipeline(
            orders, articles, service_times, driver_mapping
        )
    else:
        logger.info(f"Loading processed data from {data_path}")
        processed_data = pd.read_parquet(data_path)
    
    return processed_data


def extract_features(web_order_id: str, driver_id: str) -> Dict:
    """
    Features für Vorhersage extrahieren.
    
    Args:
        web_order_id: Bestell-Identifikator
        driver_id: Fahrer-Identifikator (kann beliebig sein)
        
    Returns:
        Dictionary mit Features
    """
    df = load_processed_data()
    
    # Bestellung in verarbeiteten Daten finden
    order_data = df[df['web_order_id'] == web_order_id]
    
    if len(order_data) == 0:
        raise ValueError(f"Order {web_order_id} not found in dataset")
    
    # Ersten Treffer nehmen (falls Duplikate)
    order_data = order_data.iloc[0]
    
    # Features extrahieren
    feature_cols = [
        'floor',
        'has_elevator',
        'is_business',
        'order_hour',
        'order_day_of_week',
        'order_month',
        'is_weekend',
        'total_boxes',
        'total_articles',
        'total_weight_g',
        'avg_article_weight_g',
        'max_article_weight_g',
        'min_article_weight_g',
        'weight_per_box',
        'floor_elevator_interaction',
        'business_floor_interaction',
    ]
    
    features = {}
    for col in feature_cols:
        if col in order_data.index:
            features[col] = float(order_data[col])
        else:
            features[col] = 0.0
    
    return features


@app.on_event("startup")
async def startup_event():
    """Bei Start initialisieren."""
    logger.info("Service-Zeit-Vorhersage API wird gestartet...")
    
    # Logger initialisieren (sicherstellen, dass DB-Tabellen erstellt werden)
    try:
        feature_logger._initialize_db()
        prediction_logger._initialize_db()
        logger.info("✓ Datenbank-Logger initialisiert")
    except Exception as e:
        logger.warning(f"Logger konnten nicht initialisiert werden: {e}")
    
    # Standard-Modell laden
    try:
        load_model("latest")
        logger.info("✓ Aktuelles Modell geladen")
    except Exception as e:
        logger.warning(f"Aktuelles Modell konnte nicht geladen werden: {e}")
    
    # Verarbeitete Daten laden
    try:
        load_processed_data()
        logger.info("✓ Verarbeitete Daten geladen")
    except Exception as e:
        logger.warning(f"Verarbeitete Daten konnten nicht geladen werden: {e}")
    
    logger.info("✓ API bereit")


@app.get("/", response_model=HealthResponse)
async def root():
    """Gesundheitscheck-Endpunkt."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(models),
        available_versions=list(models.keys())
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detaillierter Gesundheitscheck."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(models),
        available_versions=list(models.keys())
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """Verfügbare Modelle auflisten."""
    model_dir = Path(config['mlflow']['artifact_location'])
    model_files = list(model_dir.glob("model_*.joblib"))
    
    models_info = []
    for model_path in model_files:
        version = model_path.stem.replace("model_", "")
        models_info.append(
            ModelInfo(
                version=version,
                path=str(model_path),
                loaded=version in models
            )
        )
    
    return models_info


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, model_version: str = "latest"):
    """
    Einzel-Vorhersage-Endpunkt.
    
    Args:
        request: Vorhersage-Anfrage mit driver_id und web_order_id
        model_version: Zu verwendende Modellversion
        
    Returns:
        Vorhersage-Antwort
    """
    try:
        # Load model
        model = load_model(model_version)
        
        # Extract features
        features = extract_features(request.web_order_id, request.driver_id)
        
        # Create feature array
        feature_array = np.array([list(features.values())])
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        
        # Generate IDs
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Log features and prediction
        feature_logger.log_features(
            web_order_id=request.web_order_id,
            driver_id=request.driver_id,
            features=features,
            model_version=model_version
        )
        
        prediction_logger.log_prediction(
            web_order_id=request.web_order_id,
            driver_id=request.driver_id,
            predicted_service_time=float(prediction),
            model_version=model_version,
            request_id=request_id
        )
        
        return PredictionResponse(
            driver_id=request.driver_id,
            web_order_id=request.web_order_id,
            predicted_service_time=float(prediction),
            model_version=model_version,
            timestamp=timestamp,
            request_id=request_id
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_request: BatchPredictionRequest):
    """
    Batch-Vorhersage-Endpunkt.
    
    Args:
        batch_request: Batch von Vorhersage-Anfragen
        
    Returns:
        Batch-Vorhersage-Antwort
    """
    try:
        # Validate batch size
        max_batch_size = config['api'].get('max_batch_size', 100)
        if len(batch_request.requests) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum of {max_batch_size}"
            )
        
        # Load model
        model = load_model(batch_request.model_version)
        
        predictions = []
        features_list = []
        web_order_ids = []
        driver_ids = []
        
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Process each request
        for req in batch_request.requests:
            try:
                # Extract features
                features = extract_features(req.web_order_id, req.driver_id)
                features_list.append(features)
                web_order_ids.append(req.web_order_id)
                driver_ids.append(req.driver_id)
                
            except Exception as e:
                logger.warning(f"Error processing {req.web_order_id}: {e}")
                continue
        
        if not features_list:
            raise HTTPException(status_code=400, detail="No valid orders found")
        
        # Create feature matrix
        feature_matrix = np.array([list(f.values()) for f in features_list])
        
        # Make predictions
        predictions_array = model.predict(feature_matrix)
        
        # Log batch features and predictions
        feature_logger.log_batch_features(
            web_order_ids=web_order_ids,
            driver_ids=driver_ids,
            features_list=features_list,
            model_version=batch_request.model_version
        )
        
        prediction_logger.log_batch_predictions(
            web_order_ids=web_order_ids,
            driver_ids=driver_ids,
            predictions=[float(p) for p in predictions_array],
            model_version=batch_request.model_version,
            request_id=request_id
        )
        
        # Create response
        for woid, did, pred in zip(web_order_ids, driver_ids, predictions_array):
            predictions.append(
                PredictionResponse(
                    driver_id=did,
                    web_order_id=woid,
                    predicted_service_time=float(pred),
                    model_version=batch_request.model_version,
                    timestamp=timestamp,
                    request_id=request_id
                )
            )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            request_id=request_id,
            timestamp=timestamp
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/features")
async def get_feature_logs(
    web_order_id: Optional[str] = None,
    driver_id: Optional[str] = None,
    limit: int = 100
):
    """Geloggte Features abrufen."""
    try:
        logs = feature_logger.get_features(web_order_id, driver_id, limit)
        return logs.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/predictions")
async def get_prediction_logs(
    web_order_id: Optional[str] = None,
    driver_id: Optional[str] = None,
    limit: int = 100
):
    """Geloggte Vorhersagen abrufen."""
    try:
        logs = prediction_logger.get_predictions(web_order_id, driver_id, limit)
        return logs.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/statistics")
async def get_prediction_statistics():
    """Vorhersage-Statistiken abrufen."""
    try:
        stats = prediction_logger.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=ModelMetrics)
async def get_model_metrics(version: str = "latest"):
    """
    Modell-Performance-Metriken abrufen.
    
    Args:
        version: Modellversion (Standard: latest)
    
    Returns:
        Modell-Metriken inkl. Training- und Test-Performance
    """
    try:
        model_dir = Path(config['mlflow']['artifact_location'])
        
        # Find the latest metadata file
        if version == "latest":
            metadata_files = sorted(model_dir.glob("metadata_*.yaml"))
            if not metadata_files:
                raise HTTPException(status_code=404, detail="No metadata files found")
            metadata_path = metadata_files[-1]
        else:
            metadata_path = model_dir / f"metadata_{version}.yaml"
            if not metadata_path.exists():
                raise HTTPException(status_code=404, detail=f"Metadata for version {version} not found")
        
        # Load metadata with UnsafeLoader to handle numpy objects
        with open(metadata_path, 'r') as f:
            metadata = yaml.load(f, Loader=yaml.UnsafeLoader)
        
        # Convert numpy values to Python floats
        def convert_to_float(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return float(obj.item())
            elif isinstance(obj, (int, float)):
                return float(obj)
            return obj
        
        # Process metrics
        training_metrics = {}
        test_metrics = {}
        if 'metrics' in metadata:
            for key, value in metadata['metrics'].items():
                converted_value = convert_to_float(value)
                if key.startswith('train_'):
                    training_metrics[key.replace('train_', '')] = converted_value
                elif key.startswith('test_'):
                    test_metrics[key.replace('test_', '')] = converted_value
        
        # Load feature importance if available
        feature_importance = None
        feature_importance_path = model_dir / "feature_importance.csv"
        if feature_importance_path.exists():
            importance_df = pd.read_csv(feature_importance_path)
            feature_importance = importance_df.to_dict('records')
        
        return ModelMetrics(
            model_version=metadata.get('model_version', version),
            model_type=metadata.get('model_type', 'unknown'),
            timestamp=metadata.get('timestamp', 'unknown'),
            training_metrics=training_metrics,
            test_metrics=test_metrics,
            feature_importance=feature_importance
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['api']['reload'],
        log_level=config['api']['log_level']
    )
