# Notebook Development Guide for Claude

This document explains what each Jupyter notebook in the `notebooks/` directory should accomplish. Use this as a reference when developing the empty notebooks.

## Overview

The notebooks form a complete workflow for the NEON AOP Crosswalk system:
1. **Data Exploration** → 2. **Satellite Processing** → 3. **Feature Engineering** → 4. **Model Training** → 5. **Model Evaluation**

## Notebook 01: NEON Data Exploration (`01_neon_data_exploration.ipynb`)

**Status**: **COMPLETE** - 9.3KB, 292 lines

**Purpose**: Explore and understand NEON AOP data structure and content

**What it should do**:
- Load and examine NEON AOP data products (CHM, HSI, RGB, LiDAR)
- Visualize data distributions and spatial patterns
- Demonstrate data quality assessment
- Show basic statistics and metadata

**Key sections**:
- Data loading and inspection
- Spatial visualization with folium/geopandas
- Statistical analysis and distributions
- Quality assessment metrics

## Notebook 02: AOP-Satellite Crosswalk (`02_aop_satellite_crosswalk.ipynb`)

**Status**: **COMPLETE** - 26KB, 626 lines

**Purpose**: Demonstrate the core crosswalk functionality between AOP and satellite data

**What it should do**:
- Load both AOP and satellite data for the same area
- Perform geospatial alignment and resampling
- Extract features from both data sources
- Train and validate crosswalk models
- Show before/after calibration results

**Key sections**:
- Data loading and preprocessing
- Geospatial alignment utilities
- Feature extraction from AOP data
- Crosswalk model training
- Validation and visualization

## Notebook 03: Satellite Data Processing (`02_satellite_data_processing.ipynb`)

**Status**: **EMPTY** - Ready for Claude development

**Purpose**: Process and prepare satellite data for crosswalk analysis

**What it should accomplish**:

### 1. **Data Loading & Setup**
```python
# Import required libraries
import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from datetime import datetime, timedelta

# Initialize Earth Engine
ee.Initialize()
```

### 2. **Sentinel-2 Data Collection**
- Collect Sentinel-2 imagery for target areas
- Filter by date, cloud cover, and quality
- Extract NDVI, NBR, NDWI, EVI indices
- Handle missing data and gaps

### 3. **Landsat Data Collection**
- Collect Landsat 8/9 imagery for comparison
- Extract same vegetation indices
- Perform temporal compositing
- Quality assessment and filtering

### 4. **Data Preprocessing**
- Cloud masking and shadow detection
- Atmospheric correction considerations
- Temporal consistency checks
- Export to analysis-ready format

### 5. **Visualization & Quality Control**
- Interactive maps showing data coverage
- Time series plots of vegetation indices
- Quality metrics and statistics
- Comparison between different satellite sources

### Fire Case Study Focus
- **GRSM**: Pre/post-fire satellite data (2016-2017)
- **SOAP**: Multi-fire progression (2020-2021)
- **SYCA**: Contemporary fire event (2024)
- **Baseline Sites**: SRER, JORN, ONAQ, SJER for comparison

**Expected Outputs**:
- Processed satellite data arrays
- Quality assessment reports
- Interactive maps and visualizations
- Data ready for crosswalk analysis

## Notebook 04: Feature Engineering (`03_feature_engineering.ipynb`)

**Status**: **EMPTY** - Ready for Claude development

**Purpose**: Extract and engineer features from AOP and satellite data

**What it should accomplish**:

### 1. **AOP Feature Extraction**
```python
# Import feature extraction modules
from src.features.aop_features import (
    extract_chm_features, 
    extract_spectral_features,
    extract_texture_features
)
```

- **CHM Features**: Height percentiles, canopy cover, complexity metrics
- **Spectral Features**: NDVI, EVI, NBR, NDWI from hyperspectral data
- **Texture Features**: GLCM-based texture analysis
- **LiDAR Features**: Point density, elevation statistics

### 2. **Satellite Feature Engineering**
- **Vegetation Indices**: NDVI, NBR, NDWI, EVI, SAVI
- **Temporal Features**: Seasonal trends, change detection
- **Spatial Features**: Neighborhood statistics, gradients
- **Quality Features**: Cloud cover, data gaps, uncertainty

### 3. **Feature Selection & Engineering**
- Correlation analysis between features
- Principal component analysis
- Feature importance ranking
- Dimensionality reduction

### 4. **Data Integration**
- Align AOP and satellite features spatially
- Handle different resolutions and projections
- Create training datasets for crosswalk models
- Quality control and validation

### Fire-Specific Features
- **Pre/Post-Fire Comparison**: Feature differences between fire-impacted and baseline sites
- **Fire Severity Indicators**: Features that correlate with fire impact
- **Recovery Metrics**: Temporal change indicators
- **Ecosystem Resilience**: Features that predict recovery potential

**Expected Outputs**:
- Feature matrices for both data sources
- Feature importance rankings
- Integrated training datasets
- Quality assessment reports

## Notebook 05: Model Training (`04_model_training.ipynb`)

**Status**: **EMPTY** - Ready for Claude development

**Purpose**: Train crosswalk models to map satellite features to AOP features

**What it should accomplish**:

### 1. **Data Preparation**
```python
# Import model training modules
from src.features.aop_crosswalk import (
    calibrate_satellite_indices,
    fit_linear_crosswalk,
    fit_ensemble_crosswalk
)
```

- Split data into training/validation sets
- Feature scaling and normalization
- Handle missing values and outliers
- Create cross-validation folds

### 2. **Linear Model Training**
- **Ridge Regression**: Basic linear crosswalk
- **Cross-validation**: Find optimal regularization
- **Feature importance**: Coefficient analysis
- **Performance metrics**: R², MAE, RMSE

### 3. **Ensemble Model Training**
- **Gradient Boosting**: Non-linear relationships
- **Hyperparameter tuning**: Grid search or random search
- **Early stopping**: Prevent overfitting
- **Model persistence**: Save trained models

### 4. **Model Comparison**
- Performance comparison between models
- Feature importance analysis
- Residual analysis and diagnostics
- Cross-validation stability

### Fire-Enhanced Training
- **Site-Specific Models**: Train separate models for fire vs. baseline sites
- **Temporal Validation**: Use pre/post-fire data for validation
- **Severity Calibration**: Calibrate models with fire severity data
- **Recovery Prediction**: Models that predict post-fire conditions

**Expected Outputs**:
- Trained crosswalk models
- Performance metrics and comparisons
- Feature importance rankings
- Model validation reports

## Notebook 06: Model Evaluation (`05_model_evaluation.ipynb`)

**Status**: **EMPTY** - Ready for Claude development

**Purpose**: Comprehensive evaluation of crosswalk model performance

**What it should accomplish**:

### 1. **Model Performance Assessment**
```python
# Import evaluation modules
from src.features.aop_crosswalk import validate_crosswalk
from src.integration.aop_integration import AOPIntegrationManager
```

- **Accuracy Metrics**: R², MAE, RMSE, bias
- **Spatial Analysis**: Performance across different areas
- **Temporal Analysis**: Performance over time
- **Uncertainty Quantification**: Confidence intervals

### 2. **Validation Strategies**
- **Holdout Validation**: Independent test set
- **Cross-validation**: K-fold and spatial CV
- **Bootstrap Validation**: Uncertainty estimation
- **External Validation**: New sites/years

### 3. **Error Analysis**
- Residual plots and diagnostics
- Error spatial distribution
- Feature-specific performance
- Outlier detection and analysis

### 4. **Integration Testing**
- End-to-end workflow testing
- Performance with real satellite data
- Comparison to baseline methods
- Scalability and efficiency

### Fire Research Validation
- **Pre/Post-Fire Performance**: How well models work across fire events
- **Severity Mapping Accuracy**: Validation against fire severity data
- **Recovery Prediction**: Models' ability to predict post-fire conditions
- **Multi-Site Validation**: Performance across different fire types and ecosystems

**Expected Outputs**:
- Comprehensive validation reports
- Performance visualizations
- Error analysis and diagnostics
- Recommendations for improvement

## Technical Requirements

### **Dependencies**
All notebooks should use the existing codebase:
- Import from `src.features.*`, `src.models.*`, `src.integration.*`
- Use configuration from `configs/` directory
- Follow the established code patterns and conventions

### **Data Sources**
- **NEON AOP**: Use the 7 configured sites (3 fire case studies + 4 baseline)
- **Fire Case Studies**: GRSM (2016-2017), SOAP (2020-2021), SYCA (2024)
- **Baseline Sites**: SRER, JORN, ONAQ, SJER for comparison
- **Satellite**: Sentinel-2 and Landsat data via Google Earth Engine
- **Weather**: OpenWeatherMap integration (if needed)

### **Outputs**
- **Interactive visualizations**: Folium maps, Plotly charts
- **Data exports**: CSV, GeoTIFF, or pickle files as appropriate
- **Documentation**: Clear explanations and markdown cells
- **Reproducibility**: All cells should run independently

### **Code Quality**
- **Clear documentation**: Explain what each cell does
- **Error handling**: Graceful handling of missing data
- **Progress indicators**: For long-running operations
- **Modular design**: Break complex operations into functions

## Success Criteria

A notebook is complete when:
1. **All cells run without errors** (assuming data is available)
2. **Clear learning objectives** are met
3. **Interactive visualizations** work properly
4. **Outputs are useful** for understanding the system
5. **Code is well-documented** and explainable
6. **Integration** with the main system is demonstrated
7. **Fire research applications** are clearly showcased

## Getting Started

When developing each notebook:
1. **Start with imports** and setup
2. **Add markdown explanations** for each section
3. **Implement core functionality** step by step
4. **Add visualizations** and interactive elements
5. **Test with sample data** to ensure it works
6. **Document outputs** and next steps
7. **Highlight fire research** applications and case studies

## Fire Research Focus Areas

### **Pre/Post-Fire Analysis**
- Compare vegetation structure before and after fire events
- Map fire severity using high-resolution AOP data
- Track recovery trajectories over time

### **Crosswalk Validation**
- Use fire events as natural experiments
- Validate satellite-AOP relationships across fire-impacted landscapes
- Assess model performance in extreme conditions

### **Ecosystem Resilience**
- Identify features that predict recovery potential
- Map areas of high/low resilience
- Support post-fire management decisions

This will create a comprehensive learning experience that demonstrates the full NEON AOP Crosswalk system with real fire research applications!
