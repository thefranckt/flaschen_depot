"""
FastAPI application for serving Flaschen Depot ML models.
"""

import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Flaschen Depot API",
    description="API for bottle depot prediction and management",
    version="0.1.0",
)

# Model placeholder
model = None


class BottleData(BaseModel):
    """Input data model for bottle prediction."""

    bottle_type: int = Field(..., description="Encoded bottle type (0: Aluminum, 1: Glass, 2: PET)")
    volume_ml: float = Field(..., description="Bottle volume in milliliters", gt=0)
    deposit_amount: float = Field(..., description="Deposit amount in currency", ge=0)
    condition: int = Field(..., description="Encoded condition (0-3)")
    return_count: int = Field(..., description="Number of times returned", ge=0)
    last_return_days: int = Field(..., description="Days since last return", ge=0)


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    prediction: float
    model_version: str = "0.1.0"


@app.on_event("startup")
async def load_model():
    """Load the ML model on startup."""
    global model
    model_path = Path("models/model.pkl")

    if model_path.exists():
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            model = None
    else:
        logger.warning(f"Model file not found at {model_path}")


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Flaschen Depot API",
        "version": "0.1.0",
        "status": "active",
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(bottle: BottleData) -> PredictionResponse:
    """
    Make a prediction for a bottle.

    Args:
        bottle: Bottle data for prediction

    Returns:
        Prediction response with result
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model is trained and saved.",
        )

    try:
        # Convert input to numpy array
        features = np.array(
            [
                [
                    bottle.bottle_type,
                    bottle.volume_ml,
                    bottle.deposit_amount,
                    bottle.condition,
                    bottle.return_count,
                    bottle.last_return_days,
                ]
            ]
        )

        # Make prediction
        prediction = model.predict(features)[0]

        logger.info(f"Prediction made: {prediction}")

        return PredictionResponse(prediction=float(prediction))

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(bottles: List[BottleData]) -> List[PredictionResponse]:
    """
    Make predictions for multiple bottles.

    Args:
        bottles: List of bottle data for predictions

    Returns:
        List of prediction responses
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model is trained and saved.",
        )

    try:
        # Convert inputs to numpy array
        features = np.array(
            [
                [
                    bottle.bottle_type,
                    bottle.volume_ml,
                    bottle.deposit_amount,
                    bottle.condition,
                    bottle.return_count,
                    bottle.last_return_days,
                ]
                for bottle in bottles
            ]
        )

        # Make predictions
        predictions = model.predict(features)

        logger.info(f"Batch prediction made for {len(bottles)} bottles")

        return [PredictionResponse(prediction=float(pred)) for pred in predictions]

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
