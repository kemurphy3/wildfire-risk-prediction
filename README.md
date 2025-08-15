# Wildfire Risk Prediction System

A comprehensive wildfire risk prediction system that integrates real-time environmental data, satellite imagery, and machine learning models to provide accurate wildfire risk assessments.

## 🌟 Key Features

- **Real-Time Data Integration**: Live weather, satellite, and environmental data
- **Advanced ML Models**: Random Forest, XGBoost, LightGBM, ConvLSTM, and Ensemble methods
- **NEON AOP Crosswalk**: High-resolution airborne data integration for enhanced accuracy
- **Interactive Dashboard**: Plotly Dash-based web interface with real-time updates
- **Comprehensive API**: FastAPI backend with automated model training and validation
- **Modern Research Integration**: Latest scientific findings (2020-2024) and algorithms

## 🚀 NEON AOP Crosswalk System

The system includes a sophisticated NEON AOP (Airborne Observation Platform) crosswalk framework that bridges high-resolution airborne data with satellite observations:

### What is NEON AOP Crosswalk?
- **High-Resolution Ground Truth**: 1-meter resolution airborne data (LiDAR, hyperspectral, RGB)
- **Satellite Calibration**: Calibrates 10-30m satellite indices using airborne measurements
- **Enhanced Accuracy**: Improves wildfire risk prediction through better vegetation characterization
- **Public Data Access**: All NEON AOP data is publicly available through the NEON Data Portal

### Supported NEON Sites
- **SRER**: Santa Rita Experimental Range (Arizona) - Desert grassland ecosystem
- **JORN**: Jornada Experimental Range (New Mexico) - Chihuahuan Desert
- **ONAQ**: Onaqui Airstrip (Utah) - Sagebrush steppe
- **SJER**: San Joaquin Experimental Range (California) - Oak woodland

### Data Products
- **Canopy Height Model (CHM)**: 1m resolution vegetation height
- **Hyperspectral Reflectance**: 426 bands (380-2510nm) for detailed spectral analysis
- **RGB Camera Imagery**: 10cm resolution visual interpretation
- **Discrete Return LiDAR**: High-density 3D point clouds

### Crosswalk Models
- **Linear Models**: Ridge regression for satellite-AOP mapping
- **Ensemble Models**: Gradient Boosting for complex relationships
- **Validation**: Comprehensive testing with R², MAE, and bias metrics

## 🔧 Installation

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/wildfire-risk-prediction.git
cd wildfire-risk-prediction

# Install dependencies
python install_dependencies.py

# Set up environment variables
python create_env_file.py

# Run the dashboard
python run_dashboard.py
```

### Dependencies
- **Core**: `requirements.txt` - Essential packages for basic functionality
- **Minimal**: `requirements-minimal.txt` - Core wildfire prediction capabilities
- **Development**: `requirements-dev.txt` - Testing, linting, and development tools

## 📊 Usage

### NEON AOP Crosswalk
```bash
# Download AOP data for a specific site
make download-SRER

# Process AOP data and extract features
make process

# Train crosswalk models
make calibrate

# Validate models
make validate

# Run complete pipeline
make all
```

### Python API
```python
from src.integration.aop_integration import AOPIntegrationManager

# Initialize integration manager
manager = AOPIntegrationManager('configs/aop_sites.yaml')

# Generate enhanced features
enhanced_data = manager.get_enhanced_features(satellite_data, 'SRER', 2021)

# Validate integration
validation = manager.validate_integration('SRER', 2021)
```

### Dashboard
- **Real-Time Monitoring**: Live environmental data and risk assessments
- **Interactive Maps**: Geographic visualization of risk factors
- **Model Insights**: Feature importance and prediction explanations
- **Data Export**: Download results in multiple formats

## 🔬 Scientific Foundation

The system integrates cutting-edge research from 2020-2024:

### Modern Fire Indices
- **Enhanced Fire Weather Index (FWI)**: Improved meteorological fire danger assessment
- **Hot-Dry-Windy (HDW) Index**: Advanced wind-driven fire potential modeling
- **ML-based Fire Potential Index (FPI)**: Machine learning enhanced fire risk prediction

### Advanced Satellite Features
- **VIIRS Fire Detections**: Real-time fire hotspot identification
- **Sentinel-5P CO Levels**: Atmospheric carbon monoxide monitoring
- **ECOSTRESS Water Stress**: Evapotranspiration-based drought assessment

### Environmental Integration
- **Wildland-Urban Interface (WUI)**: Proximity to human development
- **Social Vulnerability Index (SVI)**: Community resilience assessment
- **Lightning Strike Density**: Natural ignition probability

## 📁 Project Structure

```
wildfire-risk-prediction/
├── src/
│   ├── dashboard/          # Plotly Dash web interface
│   ├── data_collection/    # Data acquisition modules
│   ├── features/           # Feature engineering and AOP crosswalk
│   ├── models/             # Machine learning models
│   ├── integration/        # AOP integration and system coordination
│   └── utils/              # Utility functions and geospatial tools
├── configs/                # Configuration files
├── tests/                  # Comprehensive test suite
├── notebooks/              # Jupyter notebooks and examples
├── data/                   # Data storage (not in Git)
├── logs/                   # Processing logs
└── docs/                   # Documentation
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
# All tests
python -m pytest

# NEON AOP specific tests
python -m pytest tests/test_aop_integration.py -v

# Specific component tests
python -m pytest tests/test_features/ -v
```

## 📚 Documentation

- **Quick Start**: `QUICK_START.md` - Get up and running quickly
- **Project Structure**: `PROJECT_STRUCTURE.md` - Detailed component overview
- **NEON AOP**: `NEON_AOP_IMPLEMENTATION_SUMMARY.md` - Crosswalk system details
- **API Reference**: `docs/api/` - Complete API documentation

## 🌐 Data Sources

### Public Data (No API Keys Required)
- **NEON AOP Data**: Publicly available through NEON Data Portal
- **OpenWeatherMap**: Free tier available (API key required)
- **Google Earth Engine**: Free for research and education

### Data Privacy
- **No Sensitive Data**: Repository contains only code and configuration
- **Public Access**: All referenced data sources are publicly available
- **User Control**: Users download and process their own data locally

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and code of conduct.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
black src/ tests/
flake8 src/ tests/

# Run tests with coverage
pytest --cov=src --cov-report=html
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NEON**: National Ecological Observatory Network for AOP data
- **OpenWeatherMap**: Weather data and API
- **Google Earth Engine**: Satellite data and processing platform
- **Research Community**: Latest wildfire science and methodologies

## 📞 Support

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and community support
- **Documentation**: Comprehensive guides and examples in the docs folder

---

**Note**: This is a public repository. All NEON AOP data referenced is publicly available through the NEON Data Portal. No private or sensitive data is included in this repository.