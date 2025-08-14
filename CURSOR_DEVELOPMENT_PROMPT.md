# Cursor AI Development Prompt for Wildfire Risk Prediction

Use this prompt in Cursor AI to continue developing the Wildfire Risk Prediction project:

## Initial Development Prompt

```
I'm developing an open-source educational project for wildfire risk prediction that combines NEON ecological data with satellite imagery and machine learning. The project structure is already set up with:

- NEON data collection module (src/data_collection/neon_client.py) with support for both ground-based sensors and Airborne Observation Platform (AOP) data
- AOP-satellite crosswalk capabilities for hyperspectral/LiDAR to multispectral mapping
- Basic project structure and README
- Example notebooks for data exploration and AOP crosswalk
- Test framework

Key enhancement: The project now includes NEON AOP integration using DP3 mosaic products:
- DP3.30006.001: Spectrometer orthorectified surface directional reflectance - mosaic (hyperspectral)
- DP3.30010.001: High-resolution orthorectified camera imagery - mosaic (RGB)
- DP3.30015.001: Ecosystem structure (LiDAR-derived canopy metrics)
- DP3.30024.001: Elevation - LiDAR (Digital Terrain and Surface Models)

Please help me implement the following components systematically:

1. **Satellite Data Integration** (src/data_collection/satellite_client.py)
   - Google Earth Engine integration for Sentinel-2, MODIS, and Landsat
   - Cloud masking and temporal compositing
   - Calculation of vegetation indices (NDVI, NBR, NDWI)
   - Proper caching and error handling

2. **Feature Engineering Pipeline** (src/features/fire_features.py)
   - Fuel Moisture Content (FMC) calculation from NEON data
   - Fire Weather Index (FWI) components
   - Topographical features (slope, aspect, elevation)
   - Vegetation type classification
   - Data fusion between NEON ground data and satellite imagery

3. **Machine Learning Models** (src/models/)
   - Baseline Random Forest implementation with scikit-learn
   - XGBoost model with hyperparameter tuning
   - ConvLSTM for spatiotemporal prediction using PyTorch
   - Ensemble methods combining all models
   - SHAP integration for model explainability

4. **API Development** (src/api/)
   - FastAPI application with proper error handling
   - Endpoints for single location and batch predictions
   - Caching layer with Redis
   - API documentation with OpenAPI/Swagger

5. **Visualization Dashboard** (src/visualization/)
   - Interactive risk maps using Folium or Plotly Dash
   - Time series visualization of risk factors
   - Feature importance displays
   - Model performance metrics dashboard

For each component:
- Write production-quality code with proper error handling
- Include comprehensive docstrings
- Add unit tests
- Follow the existing code style
- Keep the educational focus with clear comments explaining the science

This is an educational project to demonstrate best practices for ecological data fusion, not for operational use.
```

## Component-Specific Prompts

### Satellite Data Integration

```
Create a satellite data collection module at src/data_collection/satellite_client.py that:

1. Integrates with Google Earth Engine API
2. Downloads Sentinel-2, MODIS, and Landsat imagery for specified locations
3. Implements cloud masking using QA bands
4. Calculates these vegetation indices:
   - NDVI (Normalized Difference Vegetation Index)
   - NBR (Normalized Burn Ratio)
   - NDWI (Normalized Difference Water Index)
   - LST (Land Surface Temperature)

5. Handles temporal compositing (e.g., monthly medians)
6. Includes proper error handling and retry logic
7. Implements caching to avoid redundant API calls

The module should follow the same pattern as neon_client.py with a main class and clear methods. Include docstrings explaining the remote sensing concepts for educational purposes.
```

### Feature Engineering

```
Implement the feature engineering pipeline at src/features/fire_features.py:

1. Create a FireRiskFeatureEngine class that:
   - Loads NEON ecological data and satellite imagery
   - Calculates Fuel Moisture Content (FMC) from vegetation data
   - Implements Fire Weather Index (FWI) calculations
   - Extracts topographical features from DEM data
   - Performs data fusion between ground and satellite observations

2. Include these specific features:
   - Vapor Pressure Deficit (VPD)
   - Dead fuel moisture from Nelson model
   - Keetch-Byram Drought Index (KBDI)
   - Energy Release Component (ERC)
   - Spread Component (SC)

3. Add methods for:
   - Feature scaling and normalization
   - Handling missing data
   - Creating temporal features (rolling averages, lags)
   - Spatial features (distance to roads, water bodies)

Include detailed comments explaining the fire science behind each feature.
```

### ML Model Development

```
Develop the machine learning models in src/models/ with:

1. **baseline_model.py** - Random Forest implementation:
   - Use scikit-learn's RandomForestRegressor
   - Implement proper cross-validation
   - Add feature importance analysis
   - Include prediction intervals

2. **xgboost_model.py** - Gradient boosting model:
   - XGBoost with custom objective for fire risk
   - Hyperparameter tuning using Optuna
   - Early stopping and regularization
   - GPU support if available

3. **convlstm_model.py** - Deep learning for spatiotemporal prediction:
   - PyTorch implementation of ConvLSTM
   - Handle sequences of satellite imagery
   - Attention mechanism for important features
   - Proper data loaders for training

4. **ensemble.py** - Combine all models:
   - Weighted average ensemble
   - Stacking with meta-learner
   - Uncertainty quantification

Each model should have train(), predict(), and evaluate() methods with consistent interfaces.
```

### API Development

```
Create a FastAPI application in src/api/main.py with:

1. Core endpoints:
   - POST /predict - Single location prediction
   - POST /predict/batch - Multiple locations
   - POST /predict/area - Polygon-based prediction
   - GET /risk-map/{tile_id} - Pre-computed risk tiles
   - GET /model/info - Model metadata and performance

2. Middleware for:
   - Request validation
   - Error handling
   - Response caching
   - Rate limiting

3. Integration with:
   - Model loading and inference
   - Feature preprocessing pipeline
   - Result post-processing

4. API documentation:
   - Pydantic models for requests/responses
   - OpenAPI schema
   - Example requests in docstrings

Include health checks and monitoring endpoints.
```

### Visualization Dashboard

```
Build an interactive dashboard in src/visualization/dashboard.py using Plotly Dash:

1. Layout with multiple tabs:
   - Real-time risk map
   - Historical analysis
   - Feature importance
   - Model performance

2. Components:
   - Interactive map with risk overlay
   - Time series plots of risk factors
   - SHAP waterfall plots
   - Confusion matrix for fire/no-fire predictions

3. Interactivity:
   - Date range selection
   - Location picker
   - Model selection
   - Export functionality

4. Real-time updates:
   - WebSocket connection for live predictions
   - Auto-refresh of risk maps
   - Alert notifications for high-risk areas

Make it educational with explanations of what each visualization shows.
```

## Testing Strategy

```
Add comprehensive tests for each module:

1. Unit tests in tests/unit/:
   - Test data collection with mocked APIs
   - Test feature calculations with known inputs/outputs
   - Test model predictions with fixture data
   - Test API endpoints with TestClient

2. Integration tests in tests/integration/:
   - End-to-end data pipeline
   - Model training and evaluation
   - API with real model inference

3. Use pytest with:
   - Fixtures for test data
   - Parametrized tests for multiple scenarios
   - Coverage reporting
   - Mock external services

Aim for >80% code coverage.
```

## Documentation Updates

```
Update the documentation:

1. Expand README.md with:
   - Detailed installation instructions
   - Example usage for each component
   - Performance benchmarks
   - Limitations and caveats

2. Create docs/ folder with:
   - API reference (auto-generated from docstrings)
   - Model architecture explanations
   - Feature engineering guide
   - Deployment instructions

3. Add Jupyter notebooks:
   - 02_satellite_data_processing.ipynb
   - 03_feature_engineering.ipynb
   - 04_model_training.ipynb
   - 05_model_evaluation.ipynb

Use clear, educational language throughout.
```

## Development Best Practices

When implementing these components:

1. **Code Quality**:
   - Follow PEP 8 style guide
   - Use type hints throughout
   - Write comprehensive docstrings
   - Keep functions focused and testable

2. **Scientific Rigor**:
   - Cite relevant papers in comments
   - Explain assumptions clearly
   - Validate against known benchmarks
   - Document limitations

3. **Educational Value**:
   - Include comments explaining the science
   - Provide references for further reading
   - Create clear visualizations
   - Make code readable for learners

4. **Ethical Considerations**:
   - Add disclaimers about operational use
   - Emphasize this is for education
   - Include uncertainty in predictions
   - Document biases and limitations

Remember: This is an educational project demonstrating best practices for environmental data science, not an operational fire prediction system.