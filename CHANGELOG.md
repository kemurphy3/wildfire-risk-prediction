# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-19

### Added
- **Complete Wildfire Risk Prediction System**
  - Satellite data integration with Google Earth Engine
  - Feature engineering pipeline for fire risk factors
  - Machine learning models (Random Forest, XGBoost, ConvLSTM, Ensemble)
  - FastAPI REST API with comprehensive endpoints
  - Interactive Plotly Dash dashboard
  - Comprehensive test suite with >80% coverage

### Core Features
- **Data Collection**
  - NEON ecological data integration
  - Sentinel-2, MODIS, and Landsat satellite data
  - Cloud masking and temporal compositing
  - Vegetation indices calculation (NDVI, NBR, NDWI, LST)

- **Feature Engineering**
  - Fuel Moisture Content (FMC) calculation
  - Fire Weather Index (FWI) components
  - Topographical features (slope, aspect, elevation)
  - Data fusion between ground and satellite observations

- **Machine Learning**
  - Random Forest baseline with feature importance and SHAP
  - XGBoost with hyperparameter tuning and early stopping
  - ConvLSTM for spatiotemporal prediction
  - Ensemble methods (voting, stacking, weighted averaging)
  - Prediction intervals and uncertainty quantification

- **API & Dashboard**
  - FastAPI with OpenAPI documentation
  - Real-time prediction endpoints
  - Interactive web dashboard with 5 tabs
  - Risk maps, time series, feature analysis, model comparison

### Technical Implementation
- Production-ready code with comprehensive error handling
- Type hints and docstrings throughout
- Comprehensive logging and monitoring
- Model persistence and loading
- Caching and performance optimization

### Documentation
- Comprehensive README with quick start guide
- API documentation with examples
- Scientific background and references
- Educational focus with clear explanations

## [0.1.0] - 2024-12-18

### Added
- Initial project structure
- Basic NEON data collection module
- Project setup and requirements

---

## Development Notes

This project is designed for educational purposes to demonstrate best practices in:
- Environmental data science
- Machine learning API development
- Data fusion and feature engineering
- Production-ready code patterns

**Not intended for operational fire prediction or emergency response.**

---

*For detailed development history, see the git commit log.*
