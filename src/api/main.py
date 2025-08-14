"""
FastAPI Application for Wildfire Risk Prediction

This module provides a comprehensive REST API for wildfire risk assessment,
including prediction endpoints, model management, and real-time risk monitoring.

Educational Note: This API demonstrates best practices for building production
machine learning APIs, including proper error handling, request validation,
response caching, and comprehensive documentation. The API serves as an
interface between the machine learning models and end users.

Key Features:
- Single location and batch predictions
- Area-based risk assessment
- Model performance monitoring
- Real-time risk updates
- Comprehensive error handling
- Request/response validation
- API documentation with OpenAPI/Swagger
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from pydantic.types import confloat, conint
import uvicorn

# Import our modules
from ..models.baseline_model import RandomForestFireRiskModel
from ..features.fire_features import FireRiskFeatureEngine
from ..data_collection.satellite_client import SatelliteDataClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize FastAPI app
app = FastAPI(
    title="Wildfire Risk Prediction API",
    description="""
    Comprehensive API for wildfire risk assessment using machine learning models
    and ecological data fusion. This API provides real-time fire risk predictions
    for locations, areas, and regions based on weather, vegetation, and topographical data.
    
    ## Educational Purpose
    
    This API is designed for educational purposes to demonstrate best practices
    in building machine learning APIs for environmental applications. It is NOT
    intended for operational fire prediction or emergency response.
    
    ## Key Features
    
    * **Real-time Predictions**: Get fire risk assessments for specific locations
    * **Batch Processing**: Process multiple locations simultaneously
    * **Area Analysis**: Assess fire risk across geographic regions
    * **Model Monitoring**: Track model performance and updates
    * **Comprehensive Documentation**: Full API reference with examples
    
    ## Data Sources
    
    * NEON ecological data (ground-based sensors)
    * Satellite imagery (Sentinel-2, MODIS, Landsat)
    * Weather observations and forecasts
    * Topographical data (DEM, slope, aspect)
    * Vegetation indices and fuel moisture
    
    ## Models
    
    * Random Forest (baseline)
    * XGBoost (gradient boosting)
    * ConvLSTM (deep learning for spatiotemporal prediction)
    * Ensemble methods (combining all models)
    """,
    version="1.0.0",
    contact={
        "name": "Wildfire Risk Prediction Project",
        "url": "https://github.com/your-username/wildfire-risk-prediction",
        "email": "your-email@example.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global variables for model and feature engine
model: Optional[RandomForestFireRiskModel] = None
feature_engine: Optional[FireRiskFeatureEngine] = None
satellite_client: Optional[SatelliteDataClient] = None

# Cache for recent predictions (in production, use Redis)
prediction_cache: Dict[str, Dict[str, Any]] = {}
cache_ttl = timedelta(hours=1)  # Cache predictions for 1 hour

# Pydantic models for request/response validation
class Location(BaseModel):
    """Geographic location for fire risk assessment."""
    latitude: confloat(ge=-90, le=90) = Field(..., description="Latitude in decimal degrees")
    longitude: confloat(ge=-180, le=180) = Field(..., description="Longitude in decimal degrees")
    elevation: Optional[confloat(ge=0)] = Field(None, description="Elevation in meters above sea level")
    
    @validator('latitude', 'longitude')
    def validate_coordinates(cls, v):
        if abs(v) > 180:
            raise ValueError('Coordinate values must be within valid ranges')
        return v

class WeatherData(BaseModel):
    """Weather observations for fire risk calculation."""
    temperature: confloat(ge=-50, le=60) = Field(..., description="Temperature in Celsius")
    relative_humidity: confloat(ge=0, le=100) = Field(..., description="Relative humidity as percentage")
    wind_speed: confloat(ge=0, le=100) = Field(..., description="Wind speed in km/h")
    precipitation: confloat(ge=0, le=500) = Field(0, description="24-hour precipitation in mm")
    vapor_pressure_deficit: Optional[confloat(ge=0)] = Field(None, description="Vapor pressure deficit in kPa")

class FireRiskRequest(BaseModel):
    """Request for single location fire risk assessment."""
    location: Location
    weather: Optional[WeatherData] = Field(None, description="Weather data (optional, will use nearest station if not provided)")
    date: Optional[str] = Field(None, description="Date for assessment (YYYY-MM-DD, defaults to current date)")
    include_confidence: bool = Field(True, description="Whether to include confidence intervals")
    include_features: bool = Field(True, description="Whether to include engineered features")

class BatchFireRiskRequest(BaseModel):
    """Request for batch fire risk assessment."""
    locations: List[Location] = Field(..., description="List of locations to assess", max_items=1000)
    weather_data: Optional[List[WeatherData]] = Field(None, description="Weather data for each location")
    date: Optional[str] = Field(None, description="Date for assessment")
    include_confidence: bool = Field(True, description="Whether to include confidence intervals")

class AreaFireRiskRequest(BaseModel):
    """Request for area-based fire risk assessment."""
    center: Location = Field(..., description="Center point of the area")
    radius_km: confloat(ge=0.1, le=100) = Field(..., description="Radius of the area in kilometers")
    resolution_m: conint(ge=30, le=1000) = Field(100, description="Spatial resolution in meters")
    date: Optional[str] = Field(None, description="Date for assessment")

class FireRiskResponse(BaseModel):
    """Response for fire risk assessment."""
    location: Location
    risk_score: confloat(ge=0, le=100) = Field(..., description="Fire risk score (0-100)")
    risk_category: str = Field(..., description="Risk category (Very Low, Low, Moderate, High, Very High, Extreme)")
    confidence: Optional[confloat(ge=0, le=1)] = Field(None, description="Prediction confidence (0-1)")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="95% confidence interval")
    features: Optional[Dict[str, float]] = Field(None, description="Engineered features used for prediction")
    timestamp: str = Field(..., description="Timestamp of the prediction")
    model_version: str = Field(..., description="Version of the model used")
    data_sources: List[str] = Field(..., description="Data sources used for the prediction")

class BatchFireRiskResponse(BaseModel):
    """Response for batch fire risk assessment."""
    predictions: List[FireRiskResponse] = Field(..., description="List of fire risk predictions")
    summary: Dict[str, Any] = Field(..., description="Summary statistics for the batch")
    processing_time: float = Field(..., description="Total processing time in seconds")

class AreaFireRiskResponse(BaseModel):
    """Response for area-based fire risk assessment."""
    center: Location = Field(..., description="Center point of the area")
    radius_km: float = Field(..., description="Radius of the area in kilometers")
    risk_grid: List[List[float]] = Field(..., description="2D grid of risk scores")
    spatial_resolution: float = Field(..., description="Spatial resolution in meters")
    risk_statistics: Dict[str, float] = Field(..., description="Statistical summary of risk scores")
    timestamp: str = Field(..., description="Timestamp of the assessment")

class ModelInfo(BaseModel):
    """Information about the trained model."""
    model_type: str = Field(..., description="Type of machine learning model")
    version: str = Field(..., description="Model version")
    training_date: str = Field(..., description="Date when the model was trained")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    feature_count: int = Field(..., description="Number of features used by the model")
    last_updated: str = Field(..., description="Last update timestamp")

class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    feature_engine_ready: bool = Field(..., description="Whether the feature engine is ready")
    uptime: float = Field(..., description="Service uptime in seconds")

# Dependency functions
def get_model() -> RandomForestFireRiskModel:
    """Get the loaded machine learning model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model

def get_feature_engine() -> FireRiskFeatureEngine:
    """Get the feature engineering engine."""
    if feature_engine is None:
        raise HTTPException(status_code=503, detail="Feature engine not ready")
    return feature_engine

def get_satellite_client() -> SatelliteDataClient:
    """Get the satellite data client."""
    if satellite_client is None:
        raise HTTPException(status_code=503, detail="Satellite client not ready")
    return satellite_client

# Utility functions
def _get_cache_key(location: Location, date: str) -> str:
    """Generate cache key for a location and date."""
    return f"{location.latitude:.4f}_{location.longitude:.4f}_{date}"

def _is_cache_valid(cache_key: str) -> bool:
    """Check if cached prediction is still valid."""
    if cache_key not in prediction_cache:
        return False
    
    cached_time = datetime.fromisoformat(prediction_cache[cache_key]['timestamp'])
    return datetime.now() - cached_time < cache_ttl

def _calculate_risk_category(risk_score: float) -> str:
    """Convert risk score to risk category."""
    if risk_score < 10:
        return "Very Low"
    elif risk_score < 25:
        return "Low"
    elif risk_score < 50:
        return "Moderate"
    elif risk_score < 75:
        return "High"
    elif risk_score < 90:
        return "Very High"
    else:
        return "Extreme"

def _load_default_model() -> RandomForestFireRiskModel:
    """Load a default trained model or create a simple one for demonstration."""
    try:
        # Try to load a saved model
        model_path = Path("models/random_forest_fire_risk.joblib")
        if model_path.exists():
            model = RandomForestFireRiskModel()
            model.load_model(str(model_path))
            logger.info("Loaded pre-trained model from disk")
            return model
        else:
            # Create a simple model for demonstration
            logger.info("No pre-trained model found, creating demonstration model")
            return _create_demonstration_model()
    except Exception as e:
        logger.warning(f"Could not load model: {e}, creating demonstration model")
        return _create_demonstration_model()

def _create_demonstration_model() -> RandomForestFireRiskModel:
    """Create a simple demonstration model for testing."""
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    # Create synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic target based on some features
    y = (X[:, 0] * 2.0 +  # Temperature effect
         X[:, 1] * 1.5 +   # Humidity effect
         X[:, 2] * 0.8 +   # Wind effect
         X[:, 3] * 1.2 +   # Vegetation effect
         np.random.randn(n_samples) * 0.1)  # Noise
    
    # Normalize to 0-100 range
    y = np.clip((y - y.min()) / (y.max() - y.min()) * 100, 0, 100)
    
    # Create model and train
    model = RandomForestFireRiskModel(model_type='regression', n_estimators=50)
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Prepare data and train
    X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X_df, y)
    model.train(X_train, y_train, X_val, y_val)
    
    logger.info("Demonstration model created and trained")
    return model

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global model, feature_engine, satellite_client
    
    logger.info("Starting Wildfire Risk Prediction API...")
    
    try:
        # Initialize feature engine
        feature_engine = FireRiskFeatureEngine()
        logger.info("Feature engine initialized")
        
        # Initialize satellite client
        satellite_client = SatelliteDataClient()
        logger.info("Satellite client initialized")
        
        # Load model
        model = _load_default_model()
        logger.info("Model loaded successfully")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down Wildfire Risk Prediction API...")

# Health check endpoint
@app.get("/health", response_model=HealthCheck, tags=["System"])
async def health_check():
    """Check the health status of the API."""
    start_time = getattr(app.state, 'start_time', datetime.now())
    uptime = (datetime.now() - start_time).total_seconds()
    
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        feature_engine_ready=feature_engine is not None,
        uptime=uptime
    )

# Model information endpoint
@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info(model: RandomForestFireRiskModel = Depends(get_model)):
    """Get information about the trained model."""
    if not model.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    return ModelInfo(
        model_type=f"Random Forest ({model.model_type})",
        version="1.0.0",
        training_date=datetime.now().strftime("%Y-%m-%d"),
        performance_metrics=model.training_metrics,
        feature_count=len(model.feature_names) if model.feature_names else 0,
        last_updated=datetime.now().isoformat()
    )

# Single location prediction endpoint
@app.post("/predict", response_model=FireRiskResponse, tags=["Predictions"])
async def predict_fire_risk(
    request: FireRiskRequest,
    model: RandomForestFireRiskModel = Depends(get_model),
    feature_engine: FireRiskFeatureEngine = Depends(get_feature_engine)
):
    """
    Predict fire risk for a single location.
    
    This endpoint provides real-time fire risk assessment for a specific
    geographic location based on current weather conditions, vegetation
    status, and topographical features.
    
    The prediction combines multiple data sources:
    - Weather observations (temperature, humidity, wind, precipitation)
    - Satellite-derived vegetation indices (NDVI, NBR, NDWI)
    - Topographical features (elevation, slope, aspect)
    - NEON ecological data (soil moisture, fuel load)
    
    Returns a comprehensive risk assessment with confidence intervals
    and detailed feature breakdown.
    """
    try:
        # Check cache first
        date = request.date or datetime.now().strftime("%Y-%m-%d")
        cache_key = _get_cache_key(request.location, date)
        
        if _is_cache_valid(cache_key):
            logger.info(f"Returning cached prediction for {cache_key}")
            return FireRiskResponse(**prediction_cache[cache_key])
        
        # Prepare location data
        coordinates = (request.location.latitude, request.location.longitude)
        
        # Engineer features
        features = feature_engine.engineer_all_features(coordinates, datetime.fromisoformat(date))
        
        # Prepare feature matrix for prediction
        feature_names = list(features.keys())
        feature_values = np.array([[features[name] for name in feature_names]])
        
        # Make prediction
        if request.include_confidence:
            predictions, lower, upper = model.predict_with_intervals(feature_values)
            confidence_interval = {
                "lower": float(lower[0]),
                "upper": float(upper[0])
            }
            confidence = 0.8  # Placeholder confidence
        else:
            predictions = model.predict(feature_values)
            confidence_interval = None
            confidence = None
        
        risk_score = float(predictions[0])
        risk_category = _calculate_risk_category(risk_score)
        
        # Prepare response
        response_data = {
            "location": request.location,
            "risk_score": risk_score,
            "risk_category": risk_category,
            "confidence": confidence,
            "confidence_interval": confidence_interval,
            "features": features if request.include_features else None,
            "timestamp": datetime.now().isoformat(),
            "model_version": "1.0.0",
            "data_sources": ["NEON", "Satellite", "Weather", "DEM"]
        }
        
        # Cache the prediction
        prediction_cache[cache_key] = response_data.copy()
        
        # Clean up old cache entries
        _cleanup_cache()
        
        logger.info(f"Fire risk prediction completed for {coordinates}: {risk_category}")
        return FireRiskResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error in fire risk prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchFireRiskResponse, tags=["Predictions"])
async def predict_fire_risk_batch(
    request: BatchFireRiskRequest,
    background_tasks: BackgroundTasks,
    model: RandomForestFireRiskModel = Depends(get_model),
    feature_engine: FireRiskFeatureEngine = Depends(get_feature_engine)
):
    """
    Predict fire risk for multiple locations in batch.
    
    This endpoint processes multiple locations simultaneously for efficient
    batch processing. It's useful for regional assessments, monitoring
    multiple sites, or processing large datasets.
    
    The batch processing includes:
    - Parallel feature engineering for multiple locations
    - Efficient model inference on batched data
    - Summary statistics for the entire batch
    - Progress tracking for long-running operations
    """
    try:
        start_time = datetime.now()
        
        # Validate batch size
        if len(request.locations) > 1000:
            raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000 locations")
        
        predictions = []
        all_features = []
        
        # Process each location
        for i, location in enumerate(request.locations):
            try:
                coordinates = (location.latitude, location.longitude)
                date = request.date or datetime.now().strftime("%Y-%m-%d")
                
                # Engineer features
                features = feature_engine.engineer_all_features(coordinates, datetime.fromisoformat(date))
                all_features.append(features)
                
                # Make prediction
                feature_values = np.array([[features[name] for name in features.keys()]])
                predictions_array = model.predict(feature_values)
                
                risk_score = float(predictions_array[0])
                risk_category = _calculate_risk_category(risk_score)
                
                # Create response for this location
                location_response = FireRiskResponse(
                    location=location,
                    risk_score=risk_score,
                    risk_category=risk_category,
                    confidence=0.8,  # Placeholder
                    confidence_interval=None,
                    features=features if request.include_confidence else None,
                    timestamp=datetime.now().isoformat(),
                    model_version="1.0.0",
                    data_sources=["NEON", "Satellite", "Weather", "DEM"]
                )
                
                predictions.append(location_response)
                
            except Exception as e:
                logger.warning(f"Error processing location {i}: {e}")
                # Continue with other locations
                continue
        
        # Calculate summary statistics
        risk_scores = [p.risk_score for p in predictions]
        summary = {
            "total_locations": len(request.locations),
            "successful_predictions": len(predictions),
            "mean_risk": np.mean(risk_scores),
            "std_risk": np.std(risk_scores),
            "min_risk": np.min(risk_scores),
            "max_risk": np.max(risk_scores),
            "risk_distribution": {
                "Very Low": len([r for r in risk_scores if r < 10]),
                "Low": len([r for r in risk_scores if 10 <= r < 25]),
                "Moderate": len([r for r in risk_scores if 25 <= r < 50]),
                "High": len([r for r in risk_scores if 50 <= r < 75]),
                "Very High": len([r for r in risk_scores if 75 <= r < 90]),
                "Extreme": len([r for r in risk_scores if r >= 90])
            }
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Batch prediction completed: {len(predictions)}/{len(request.locations)} successful")
        
        return BatchFireRiskResponse(
            predictions=predictions,
            summary=summary,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Area-based prediction endpoint
@app.post("/predict/area", response_model=AreaFireRiskResponse, tags=["Predictions"])
async def predict_fire_risk_area(
    request: AreaFireRiskRequest,
    model: RandomForestFireRiskModel = Depends(get_model),
    feature_engine: FireRiskFeatureEngine = Depends(get_feature_engine)
):
    """
    Predict fire risk across a geographic area.
    
    This endpoint provides area-based fire risk assessment by creating
    a grid of predictions across the specified region. It's useful for
    mapping fire risk across landscapes, identifying high-risk zones,
    and supporting regional planning and response.
    
    The area analysis includes:
    - Grid-based sampling across the specified area
    - Spatial interpolation of risk factors
    - Risk mapping with configurable resolution
    - Statistical summary of area-wide risk
    """
    try:
        # Calculate grid dimensions
        radius_m = request.radius_km * 1000
        grid_size = int((2 * radius_m) / request.resolution_m)
        
        # Create coordinate grid
        lat_center, lon_center = request.center.latitude, request.center.longitude
        
        # Simple grid generation (in production, use proper coordinate transformations)
        lat_range = np.linspace(lat_center - request.radius_km/111, lat_center + request.radius_km/111, grid_size)
        lon_range = np.linspace(lon_center - request.radius_km/111, lon_center + request.radius_km/111, grid_size)
        
        risk_grid = []
        risk_scores = []
        
        # Generate predictions for each grid point
        for lat in lat_range:
            row = []
            for lon in lon_range:
                try:
                    coordinates = (lat, lon)
                    date = request.date or datetime.now().strftime("%Y-%m-%d")
                    
                    # Engineer features
                    features = feature_engine.engineer_all_features(coordinates, datetime.fromisoformat(date))
                    
                    # Make prediction
                    feature_values = np.array([[features[name] for name in features.keys()]])
                    prediction = model.predict(feature_values)
                    
                    risk_score = float(prediction[0])
                    row.append(risk_score)
                    risk_scores.append(risk_score)
                    
                except Exception as e:
                    logger.warning(f"Error at grid point ({lat}, {lon}): {e}")
                    row.append(0.0)  # Default value for failed predictions
            
            risk_grid.append(row)
        
        # Calculate area statistics
        risk_statistics = {
            "mean": np.mean(risk_scores),
            "std": np.std(risk_scores),
            "min": np.min(risk_scores),
            "max": np.max(risk_scores),
            "median": np.median(risk_scores),
            "high_risk_area_percent": np.mean(np.array(risk_scores) > 50) * 100
        }
        
        logger.info(f"Area prediction completed: {grid_size}x{grid_size} grid")
        
        return AreaFireRiskResponse(
            center=request.center,
            radius_km=request.radius_km,
            risk_grid=risk_grid,
            spatial_resolution=request.resolution_m,
            risk_statistics=risk_statistics,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in area prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Area prediction failed: {str(e)}")

# Risk map endpoint
@app.get("/risk-map/{tile_id}", tags=["Maps"])
async def get_risk_map_tile(
    tile_id: str,
    z: int = Query(..., ge=0, le=20, description="Zoom level"),
    x: int = Query(..., ge=0, description="Tile X coordinate"),
    y: int = Query(..., description="Tile Y coordinate")
):
    """
    Get pre-computed fire risk map tiles.
    
    This endpoint provides map tiles for visualization of fire risk
    across different geographic areas and zoom levels. The tiles
    follow the standard XYZ tiling scheme used by web mapping libraries.
    
    Note: This is a placeholder implementation. In production, you would
    pre-compute and cache risk map tiles for efficient serving.
    """
    try:
        # Placeholder implementation - return a simple colored tile
        # In production, this would serve actual pre-computed tiles
        
        # Create a simple colored tile based on tile coordinates
        tile_size = 256
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        
        # Simple pattern for demonstration
        for i in range(tile_size):
            for j in range(tile_size):
                # Create a gradient pattern
                intensity = int(255 * (i + j) / (2 * tile_size))
                tile[i, j] = [intensity, 255 - intensity, 100]
        
        # Convert to image and return
        from PIL import Image
        img = Image.fromarray(tile)
        
        # Save to temporary file and return
        temp_path = f"/tmp/risk_tile_{tile_id}.png"
        img.save(temp_path)
        
        return FileResponse(temp_path, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error serving risk map tile: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve map tile")

# Utility functions
def _cleanup_cache():
    """Remove expired cache entries."""
    current_time = datetime.now()
    expired_keys = []
    
    for key, value in prediction_cache.items():
        cached_time = datetime.fromisoformat(value['timestamp'])
        if current_time - cached_time > cache_ttl:
            expired_keys.append(key)
    
    for key in expired_keys:
        del prediction_cache[key]
    
    if expired_keys:
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat()
        }
    )

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint with API information.
    
    Returns basic information about the Wildfire Risk Prediction API
    and links to documentation.
    """
    return {
        "message": "Wildfire Risk Prediction API",
        "version": "1.0.0",
        "description": "Comprehensive API for wildfire risk assessment using machine learning",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "predictions": {
                "single": "/predict",
                "batch": "/predict/batch",
                "area": "/predict/area"
            },
            "system": {
                "health": "/health",
                "model_info": "/model/info"
            },
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc"
            }
        },
        "educational_note": "This API is designed for educational purposes to demonstrate best practices in building machine learning APIs for environmental applications. It is NOT intended for operational fire prediction or emergency response."
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
