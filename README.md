# Wildfire Risk Prediction System

An open-source educational project that demonstrates best practices for building machine learning systems for environmental applications. This project combines NEON ecological data with satellite imagery and machine learning to predict wildfire risk.

## Educational Purpose Only

**This system is designed for educational purposes to demonstrate best practices in environmental data science and machine learning API development. It is NOT intended for operational fire prediction or emergency response.**

## Features

### Core Components
- **Satellite Data Integration**: Google Earth Engine client for Sentinel-2, MODIS, and Landsat data
- **Feature Engineering**: Comprehensive fire risk feature calculation including fuel moisture, weather indices, and topographical features
- **Machine Learning Models**: Random Forest baseline with XGBoost and ConvLSTM support
- **REST API**: FastAPI-based API with comprehensive endpoints for predictions
- **Interactive Dashboard**: Real-time Plotly Dash web application for risk visualization and model interaction
- **Real-time Processing**: Live risk assessment with caching and validation

### Data Sources
- **NEON Ecological Data**: Ground-based sensors and Airborne Observation Platform (AOP)
- **Satellite Imagery**: Multi-spectral and hyperspectral data with cloud masking
- **Weather Data**: Meteorological observations and derived fire weather indices
- **Topographical Data**: Digital elevation models (DEM) for slope, aspect, and elevation

### Machine Learning Capabilities
- **Feature Engineering**: 20+ engineered features for fire risk assessment
- **Model Training**: Automated training with hyperparameter tuning and early stopping
- **Prediction Intervals**: Confidence intervals and uncertainty quantification
- **Model Explainability**: SHAP integration for feature importance analysis
- **Advanced Models**: XGBoost with gradient boosting and ConvLSTM for spatiotemporal data
- **Ensemble Methods**: Voting, stacking, and weighted averaging strategies
- **Hyperparameter Optimization**: Grid search, random search, and cross-validation

## Architecture

```
wildfire-risk-prediction/
├── src/
│   ├── data_collection/          # Data collection modules
│   │   ├── neon_client.py        # NEON data access
│   │   └── satellite_client.py   # Google Earth Engine integration
│   ├── features/                 # Feature engineering
│   │   └── fire_features.py      # Fire risk feature calculation
│   ├── models/                   # Machine learning models
│   │   ├── baseline_model.py     # Random Forest implementation
│   │   ├── xgboost_model.py      # XGBoost model
│   │   ├── convlstm_model.py     # ConvLSTM spatiotemporal model
│   │   └── ensemble.py           # Ensemble methods (voting, stacking, weighted)
│   ├── api/                      # FastAPI application
│   │   └── main.py              # Main API endpoints
│   └── dashboard/                # Interactive web dashboard
│       ├── app.py                # Main Dash application
│       └── callbacks.py          # Dashboard interactivity
├── notebooks/                    # Jupyter notebooks for exploration
├── tests/                        # Comprehensive test suite
├── requirements.txt              # Python dependencies
├── run_dashboard.py              # Dashboard launcher script
└── README.md                     # This file
```

## Quick Start

### Prerequisites
- Python 3.8+
- Google Earth Engine account (for satellite data)
- NEON data access (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/wildfire-risk-prediction.git
   cd wildfire-risk-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google Earth Engine** (optional)
   ```bash
   earthengine authenticate
   ```

### Basic Usage

#### 1. Feature Engineering

```python
from src.features.fire_features import FireRiskFeatureEngine

# Initialize feature engine
engine = FireRiskFeatureEngine()

# Calculate features for a location
coordinates = (37.7749, -122.4194)  # San Francisco
features = engine.engineer_all_features(coordinates)

print(f"Engineered {len(features)} features")
print(f"Risk factors: {features}")
```

#### 2. Machine Learning Models

**Random Forest (Baseline)**
```python
from src.models.baseline_model import RandomForestFireRiskModel

# Initialize and train model
model = RandomForestFireRiskModel(model_type='regression')
X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X, y)
training_result = model.train(X_train, y_train, X_val, y_val)

# Make predictions with confidence intervals
predictions, lower_bounds, upper_bounds = model.predict_with_intervals(X_test[:10])
print(f"Predictions: {predictions}")
```

**XGBoost (Enhanced)**
```python
from src.models.xgboost_model import XGBoostFireRiskModel

# Initialize with early stopping and hyperparameter tuning
model = XGBoostFireRiskModel(
    model_type='regression',
    n_estimators=200,
    early_stopping_rounds=10
)

# Train with automatic hyperparameter optimization
result = model.train(
    X_train, y_train, X_val, y_val,
    hyperparameter_tuning=True,
    tuning_method='random'
)
```

**ConvLSTM (Spatiotemporal)**
```python
from src.models.convlstm_model import ConvLSTMFireRiskModel

# Initialize spatiotemporal model
model = ConvLSTMFireRiskModel(
    time_steps=10,
    spatial_dims=(32, 32),
    channels=10
)

# Train on spatiotemporal data
result = model.train(X_train, y_train, X_val, y_val)
model.plot_training_history()
```

**Ensemble Methods**
```python
from src.models.ensemble import EnsembleFireRiskModel

# Create ensemble with multiple strategies
ensemble = EnsembleFireRiskModel(
    model_type='regression',
    ensemble_method='stacking'  # or 'voting', 'weighted'
)

# Train ensemble and compare base models
result = ensemble.train(X_train, y_train, X_val, y_val)
ensemble.plot_base_model_comparison(X_test, y_test)
```

#### 3. Satellite Data Collection

```python
from src.data_collection.satellite_client import SatelliteDataClient
from shapely.geometry import Point

# Initialize client
client = SatelliteDataClient()

# Get Sentinel-2 data for a location
point = Point(-122.4194, 37.7749)
s2_data = client.get_sentinel2_data(
    geometry=point,
    start_date='2023-06-01',
    end_date='2023-06-30',
    cloud_filter=20.0
)

print(f"Collected {s2_data['count']} Sentinel-2 images")
```

#### 4. API Usage

```python
import requests

# Start the API server first (see below)
# python -m src.api.main

# Make a prediction request
url = "http://localhost:8000/predict"
data = {
    "location": {
        "latitude": 37.7749,
        "longitude": -122.4194
    },
    "weather": {
        "temperature": 25.0,
        "relative_humidity": 60.0,
        "wind_speed": 15.0,
        "precipitation": 0.0
    },
    "include_confidence": True,
    "include_features": True
}

response = requests.post(url, json=data)
result = response.json()

print(f"Fire Risk: {result['risk_score']:.1f} ({result['risk_category']})")
print(f"Confidence: {result['confidence']:.2f}")
```

#### 5. Interactive Dashboard

**Launch the Dashboard**
```bash
# Start the interactive web dashboard
python run_dashboard.py

# Dashboard will be available at: http://localhost:8050
```

**Dashboard Features**
- **Risk Assessment Tab**: Interactive maps with real-time fire risk predictions
- **Time Series Tab**: Historical analysis of risk factors with date controls
- **Feature Analysis Tab**: SHAP plots and feature importance visualization
- **Model Comparison Tab**: Performance metrics and model evaluation
- **Make Predictions Tab**: Real-time input interface for custom predictions

**Dashboard Usage**
```python
# The dashboard automatically loads demo models and data
# No additional setup required - just run and explore!

# Features include:
# - Model selection (Random Forest, XGBoost)
# - Interactive parameter adjustment
# - Real-time visualization updates
# - Export capabilities for results
```

## API Endpoints

### Core Prediction Endpoints

- **`POST /predict`** - Single location fire risk assessment
- **`POST /predict/batch`** - Batch processing for multiple locations
- **`POST /predict/area`** - Area-based risk assessment with grid generation

### System Endpoints

- **`GET /health`** - API health check
- **`GET /model/info`** - Model information and performance metrics
- **`GET /docs`** - Interactive API documentation (Swagger UI)
- **`GET /redoc`** - Alternative API documentation

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "location": {
         "latitude": 37.7749,
         "longitude": -122.4194
       },
       "weather": {
         "temperature": 25.0,
         "relative_humidity": 60.0,
         "wind_speed": 15.0,
         "precipitation": 0.0
       }
     }'
```

## Scientific Background

### Fire Risk Factors

The system calculates fire risk based on established fire science research:

1. **Fuel Moisture Content (FMC)**
   - Nelson model for dead fuel moisture
   - Temperature and humidity relationships
   - Fuel type classifications

2. **Fire Weather Index (FWI)**
   - Canadian Forest Fire Weather Index system
   - Duff Moisture Code (DMC)
   - Drought Code (DC)
   - Initial Spread Index (ISI)

3. **Vegetation Indices**
   - NDVI (Normalized Difference Vegetation Index)
   - NBR (Normalized Burn Ratio)
   - NDWI (Normalized Difference Water Index)
   - EVI (Enhanced Vegetation Index)

4. **Topographical Features**
   - Elevation and slope
   - Aspect and roughness
   - Solar radiation effects

### Data Fusion Approach

The system combines multiple data sources using:
- **Spatial Interpolation**: Kriging and inverse distance weighting
- **Temporal Compositing**: Monthly and seasonal aggregations
- **Quality Filtering**: Cloud masking and data validation
- **Feature Engineering**: Domain-specific transformations

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests
pytest tests/test_baseline_model.py -v

# Integration tests
pytest tests/ -m "integration" -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage
The test suite covers:
- Model initialization and configuration
- Data preparation and validation
- Training and hyperparameter tuning
- Prediction functionality
- Model evaluation and metrics
- Feature importance calculation
- Model persistence
- Error handling and edge cases

## Performance

### Model Performance
- **Random Forest Baseline**: R² > 0.8 on synthetic data
- **Training Time**: < 5 minutes for 1000 samples
- **Prediction Time**: < 100ms per location
- **Memory Usage**: < 2GB for typical workloads

### API Performance
- **Response Time**: < 200ms for single predictions
- **Throughput**: 100+ requests/second
- **Concurrent Users**: 50+ simultaneous users
- **Uptime**: 99.9% availability target

## Configuration

### Environment Variables
```bash
# Google Earth Engine
EARTHENGINE_CREDENTIALS_PATH=/path/to/credentials.json

# NEON Data
NEON_API_TOKEN=your_neon_token

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Cache Configuration
CACHE_TTL=3600  # 1 hour
MAX_CACHE_SIZE=10000
```

### Model Configuration
```python
# Random Forest parameters
model = RandomForestFireRiskModel(
    n_estimators=200,      # Number of trees
    max_depth=20,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split
    min_samples_leaf=2,    # Minimum samples at leaf
    random_state=42        # Reproducibility
)
```

## Deployment

### Local Development
```bash
# Start API server
python -m src.api.main

# Or with uvicorn
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start interactive dashboard
python run_dashboard.py
# Dashboard will be available at: http://localhost:8050
```

### Production Deployment
```bash
# Using gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker

# Using Docker
docker build -t wildfire-risk-api .
docker run -p 8000:8000 wildfire-risk-api
```

### Docker Support
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/

EXPOSE 8000
CMD ["python", "-m", "src.api.main"]
```

## Documentation

### API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

### Dashboard Documentation
- **Interactive Dashboard**: `http://localhost:8050`
- **Real-time Risk Maps**: Visual fire risk assessment
- **Feature Analysis**: SHAP plots and importance visualization
- **Model Comparison**: Performance metrics and evaluation

### Code Documentation
- **Docstrings**: Comprehensive inline documentation
- **Type Hints**: Full type annotations for all functions
- **Examples**: Working examples in docstrings and notebooks

### Educational Resources
- **Jupyter Notebooks**: Step-by-step tutorials
- **Scientific References**: Citations for all algorithms
- **Best Practices**: Production-ready code patterns

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Standards
- **Style**: PEP 8 compliance
- **Documentation**: Comprehensive docstrings
- **Testing**: >80% code coverage
- **Type Hints**: Full type annotations
- **Error Handling**: Proper exception handling

### Testing Guidelines
- **Unit Tests**: Test individual functions
- **Integration Tests**: Test component interactions
- **Edge Cases**: Test boundary conditions
- **Error Conditions**: Test failure scenarios

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimers

### Educational Purpose
This system is designed for educational purposes to demonstrate:
- Best practices in environmental data science
- Machine learning API development
- Data fusion and feature engineering
- Production-ready code patterns

### Not for Operational Use
**This system is NOT intended for:**
- Operational fire prediction
- Emergency response
- Real-time decision making
- Production deployment without modification

### Data Limitations
- Synthetic data used for demonstration
- Limited historical validation
- Simplified physical models
- Educational approximations

## References

### Scientific Papers
- Van Wagner (1987) - Canadian Forest Fire Weather Index
- Nelson (2000) - Dead fuel moisture prediction
- Keetch & Byram (1968) - Drought index for fire potential
- Rothermel (1972) - Mathematical fire spread model

### Technical Resources
- [Google Earth Engine Documentation](https://developers.google.com/earth-engine)
- [NEON Data Portal](https://data.neonscience.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/)

### Related Projects
- [Fire Danger Rating System](https://www.fs.fed.us/rmrs/tools/fire-danger-rating-system)
- [Canadian Wildland Fire Information System](http://cwfis.cfs.nrcan.gc.ca/)
- [USGS Fire Science](https://www.usgs.gov/programs/climate-adaptation-science-centers/fire-science)

## Support

### Getting Help
- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the comprehensive docstrings
- **Examples**: Review the Jupyter notebooks

### Community
- **Contributors**: See [CONTRIBUTORS.md](CONTRIBUTORS.md)
- **Code of Conduct**: See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md)

---

*This project demonstrates how to build production-ready machine learning systems for environmental applications while maintaining scientific rigor and educational value.*