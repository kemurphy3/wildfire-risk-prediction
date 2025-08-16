# Project Structure

Here's how the wildfire risk prediction system is organized.

## Directory Overview

```
wildfire-risk-prediction/
├── src/                          # Core source code
│   ├── api/                      # FastAPI REST endpoints
│   ├── dashboard/                # Interactive web dashboard
│   ├── data_collection/          # Data clients and integration
│   ├── features/                 # Feature engineering
│   ├── models/                   # Machine learning models
│   └── __init__.py
├── tests/                        # Test suite
├── notebooks/                    # Jupyter notebooks for exploration
├── README.md                     # Main project documentation
├── QUICK_START.md               # 5-minute setup guide
├── PROJECT_STRUCTURE.md         # This file
├── requirements.txt              # Python dependencies
├── config.py                     # Centralized configuration
├── create_env_file.py           # Secure API key setup
└── run_dashboard.py             # Dashboard launcher
```

## Core Components

### src/api/
FastAPI REST API for programmatic access
- `main.py` - API implementation with multiple endpoints
- RESTful design with automatic documentation
- Error handling and validation

### src/dashboard/
Interactive web dashboard for real-time risk visualization
- `app.py` - Main Dash application
- `callbacks.py` - Interactive functionality and data updates
- `data_integration.py` - Real-time data from multiple sources

### src/data_collection/
Data integration from multiple sources
- `neon_client.py` - NEON ecological data access
- `satellite_client.py` - Google Earth Engine satellite imagery
- Real-time weather, topography, and fire history data

### src/features/
Feature engineering for fire risk calculations
- `fire_features.py` - Engineered features for fire risk
- Scientific algorithms (FWI, FMC, KBDI, etc.)
- Topographical and environmental factor calculations

### src/models/
Machine learning models for prediction
- `baseline_model.py` - Random Forest with feature importance
- `xgboost_model.py` - Gradient boosting
- `lightgbm_model.py` - LightGBM framework
- `convlstm_model.py` - Deep learning for spatiotemporal data
- `ensemble.py` - Ensemble methods (voting, stacking)

## Data Flow

```
Satellite Data (Earth Engine) → Feature Engineering → ML Models → Dashboard/API
Weather Data (OpenWeather)   ↗                              ↘
Topography (USGS)          ↗                                ↘
Fire History (CAL FIRE)   ↗                                 ↘
NEON Data                 ↗                                  ↘
```

## Key Features by Component

### API (src/api/)
- REST endpoints for all system functions
- Automatic documentation (Swagger/ReDoc)
- Input validation and error handling
- Rate limiting and security features

### Dashboard (src/dashboard/)
- Multiple functional tabs with real-time updates
- Interactive maps and visualizations
- Live data integration from multiple sources
- Responsive design for all devices

### Models (src/models/)
- Different ML algorithms for diverse use cases
- Hyperparameter tuning and optimization
- Feature importance analysis
- Model persistence and loading
- Performance metrics and validation

### Features (src/features/)
- Engineered features for risk assessment
- Modern fire indices based on recent research
- Satellite-derived features from VIIRS, Sentinel-5P, and ECOSTRESS
- Social & environmental data including WUI proximity
- Real-time calculations from environmental data

## Configuration Management

### config.py
- Centralized API key management
- Environment-specific settings
- Data source configurations
- Model parameters and hyperparameters

### create_env_file.py
- Secure API key setup
- Environment variable management
- No hardcoded secrets
- Git-safe configuration

## Testing

### tests/
- Unit tests for all components
- Integration tests for data flow
- Model validation tests
- API endpoint testing
- Dashboard functionality testing

### Quality Standards
- Type hints throughout codebase
- Docstrings for all functions
- Error handling and logging
- PEP 8 compliance

## Scalability Features

### Performance
- Async API endpoints for concurrency
- Data caching for improved response times
- Batch processing for multiple locations
- Efficient algorithms for real-time use

### Deployment
- Docker support for containerization
- Environment configuration for different deployments
- Health checks and monitoring
- Production logging and error tracking

## Use Cases

### Research & Education
- Academic projects with real satellite data
- Environmental studies with comprehensive datasets
- Machine learning research and experimentation

### Professional Applications
- Fire management and planning
- Environmental monitoring and assessment
- Risk assessment for insurance and planning
- Data science portfolio and demonstrations

### Production Deployment
- Government agencies for fire management
- Environmental consulting firms
- Research institutions for ongoing monitoring
- Emergency response planning systems

---

This structure keeps things modular and maintainable while supporting both research and production use cases.