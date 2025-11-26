"""
FastAPI Application for Service Time Prediction
Provides REST API endpoints for batch predictions and model management.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize app
app = FastAPI(
    title="Service Time Prediction API",
    description="API for predicting delivery service times",
    version="1.0.0"
)

# Add CORS middleware
if config['api'].get('cors_enabled', True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize loggers
feature_logger = FeatureLogger(config['logging']['feature_log_db'])
prediction_logger = PredictionLogger(config['logging']['prediction_log_db'])

# Global variables for models and data
models = {}
processed_data = None
feature_engineer = None


class PredictionRequest(BaseModel):
    """Request model for single or batch predictions."""
    driver_id: str = Field(..., description="Driver ID")
    web_order_id: str = Field(..., description="Order ID from orders dataset")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    requests: List[PredictionRequest] = Field(..., description="List of prediction requests")
    model_version: Optional[str] = Field("latest", description="Model version to use")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    driver_id: str
    web_order_id: str
    predicted_service_time: float
    model_version: str
    timestamp: str
    request_id: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_count: int
    request_id: str
    timestamp: str


class ModelInfo(BaseModel):
    """Model information response."""
    version: str
    path: str
    loaded: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_loaded: int
    available_versions: List[str]


def load_model(version: str = "latest"):
    """
    Load model from disk.
    
    Args:
        version: Model version to load
        
    Returns:
        Loaded model
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
    """Load processed data for feature lookup."""
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
    Extract features for prediction.
    
    Args:
        web_order_id: Order identifier
        driver_id: Driver identifier (can be arbitrary)
        
    Returns:
        Dictionary of features
    """
    df = load_processed_data()
    
    # Find order in processed data
    order_data = df[df['web_order_id'] == web_order_id]
    
    if len(order_data) == 0:
        raise ValueError(f"Order {web_order_id} not found in dataset")
    
    # Take first match (in case of duplicates)
    order_data = order_data.iloc[0]
    
    # Extract features
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
    """Initialize on startup."""
    logger.info("Starting Service Time Prediction API...")
    
    # Load default model
    try:
        load_model("latest")
        logger.info("✓ Latest model loaded")
    except Exception as e:
        logger.warning(f"Could not load latest model: {e}")
    
    # Load processed data
    try:
        load_processed_data()
        logger.info("✓ Processed data loaded")
    except Exception as e:
        logger.warning(f"Could not load processed data: {e}")
    
    logger.info("✓ API ready")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(models),
        available_versions=list(models.keys())
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(models),
        available_versions=list(models.keys())
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models."""
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
    Single prediction endpoint.
    
    Args:
        request: Prediction request with driver_id and web_order_id
        model_version: Model version to use
        
    Returns:
        Prediction response
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
    Batch prediction endpoint.
    
    Args:
        batch_request: Batch of prediction requests
        
    Returns:
        Batch prediction response
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
    """Get logged features."""
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
    """Get logged predictions."""
    try:
        logs = prediction_logger.get_predictions(web_order_id, driver_id, limit)
        return logs.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/statistics")
async def get_prediction_statistics():
    """Get prediction statistics."""
    try:
        stats = prediction_logger.get_statistics()
        return stats
    except Exception as e:
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
