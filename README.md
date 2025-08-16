# Wildfire Risk Prediction System

Real-time wildfire risk prediction system using machine learning, satellite data, and environmental sensors.

## Features

- **Real-time data** - Live weather, satellite imagery, and sensor data integration
- **ML models** - Random Forest, XGBoost, LightGBM, ConvLSTM, and ensemble methods
- **NEON AOP crosswalk** - Airborne observation platform data for improved accuracy
- **Fire case studies** - Historical fire data for testing and validation
- **Dashboard** - Real-time Plotly Dash visualization interface
- **API** - FastAPI backend with automated training and validation
- **Latest research** - Implementation of recent papers from 2020-2024

## NEON AOP Crosswalk System

The system uses NEON's airborne observation platform data to calibrate satellite imagery by comparing high-resolution aerial scans with satellite data:

### NEON AOP Crosswalk
- **High-resolution ground truth** - 1m resolution from aircraft (LiDAR, hyperspectral, RGB cameras)
- **Satellite calibration** - Trains 10-30m satellite data using airborne measurements
- **Improved accuracy** - Enhanced detection of fire-prone vegetation patterns
- **Public data** - All NEON data is openly available

### Fire Case Study Sites (High Priority)
- **GRSM**: Great Smoky Mountains - 2016 Chimney Tops 2 Fire (11k hectares)
- **SOAP**: Soaproot Saddle - 2020 Creek Fire (153k ha) & 2021 Blue Fire (8.5k ha)
- **SYCA**: Sycamore Creek - 2024 Sand Stone Fire (3.2k ha)

### Ecosystem Diversity Sites (Baseline)
- **SRER**: Santa Rita Range, AZ - desert grassland
- **JORN**: Jornada Range, NM - Chihuahuan Desert
- **ONAQ**: Onaqui, UT - Sagebrush steppe
- **SJER**: San Joaquin, CA - Oak woodlands

### Data Products
- **CHM** - 1m res vegetation height maps
- **Hyperspectral** - 426 spectral bands (380-2510nm)
- **RGB imagery** - 10cm resolution orthoimagery
- **LiDAR point clouds** - High-density 3D data

### Crosswalk Models
- **Linear** - Ridge regression baseline
- **Ensemble** - Gradient Boosting for complex relationships
- **Validation** - R², MAE, bias metrics

### Fire Research Capabilities
- **Before/after analysis** - Change detection from fire events
- **Severity mapping** - Quantify burn severity across affected areas
- **Recovery tracking** - Monitor vegetation recovery over time
- **Temporal validation** - Verify model performance across different fire events

## Installation

### Quick Start
```bash
git clone https://github.com/yourusername/wildfire-risk-prediction.git
cd wildfire-risk-prediction

python install_dependencies.py
python create_env_file.py
python run_dashboard.py
```

### Dependencies
- **Core**: `requirements.txt` - Main dependencies
- **Minimal**: `requirements-minimal.txt` - Minimum requirements for predictions only
- **Dev**: `requirements-dev.txt` - Development and testing tools

## Usage

### NEON AOP Crosswalk
```bash
# Download fire site data
make download-GRSM    # Great Smoky Mountains
make download-SOAP    # Soaproot Saddle
make download-SYCA    # Sycamore Creek

# Download baseline sites
make download-SRER    # Santa Rita
make download-JORN    # Jornada

# Process data
make process          # Extract features
make calibrate        # Train calibration models
make validate         # Validate results

# Run complete pipeline
make all
```

### Python API
```python
from src.integration.aop_integration import AOPIntegrationManager

# Initialize manager
manager = AOPIntegrationManager('configs/aop_sites.yaml')

# Get enhanced features for pre/post fire analysis
enhanced_data = manager.get_enhanced_features(satellite_data, 'GRSM', 2016)  # Pre-fire
enhanced_data = manager.get_enhanced_features(satellite_data, 'GRSM', 2017)  # Post-fire

# Validate integration
validation = manager.validate_integration('SOAP', 2020)
```

### Dashboard
- **Real-time monitoring** - Live fire risk tracking
- **Interactive maps** - Pan, zoom, and layer controls
- **Model insights** - Feature importance visualization
- **Export data** - CSV, JSON export options
- **Fire impact visualization** - Before/after comparisons

## Technical Implementation

The system implements recent research advances (2020-2024) in wildfire prediction:

### Fire Indices
- **Enhanced FWI** - Improved weather-based fire danger rating
- **HDW Index** - Hot-dry-windy conditions for extreme fire behavior
- **ML Fire Potential** - Machine learning-based fire potential index

### Satellite Features
- **VIIRS** - Active fire detection
- **Sentinel-5P** - Carbon monoxide tracking for smoke plumes
- **ECOSTRESS** - Plant water stress measurements

### Environmental Factors
- **WUI** - Wildland-urban interface proximity
- **SVI** - Social vulnerability index
- **Lightning density** - Natural ignition sources

### Research Applications
- **Severity assessment** - Burn severity mapping using AOP data
- **Recovery monitoring** - Vegetation recovery tracking
- **Fuel load estimation** - Quantify available biomass
- **Ecosystem resilience** - Post-fire recovery potential

## Project Structure

```
wildfire-risk-prediction/
├── src/
│   ├── dashboard/          # Plotly Dash web interface
│   ├── data_collection/    # Data acquisition modules
│   ├── features/           # Feature engineering and AOP processing
│   ├── models/             # Machine learning models
│   ├── integration/        # AOP integration layer
│   └── utils/              # Utilities and geographic tools
├── configs/                # Configuration files
├── tests/                  # Test suite
├── notebooks/              # Jupyter notebooks for analysis
├── data/                   # Data directory (gitignored)
├── logs/                   # Application logs
└── docs/                   # Documentation
```

## Testing

Run tests:
```bash
# Run everything
python -m pytest

# Just AOP tests
python -m pytest tests/test_aop_integration.py -v

# Feature tests only
python -m pytest tests/test_features/ -v
```

## Docs

- **Quick Start**: `QUICK_START.md`
- **Project Structure**: `PROJECT_STRUCTURE.md`
- **NEON AOP**: `PUBLIC_DATA_ACCESS.md`
- **API Documentation**: `docs/api/`

## Data Sources

### Public Data Sources
- **NEON AOP** - Public access through NEON data portal
- **Fire case studies** - Historical fire events with temporal coverage
- **OpenWeatherMap** - Free tier available (API key required)
- **Google Earth Engine** - Free for research and education

### Data Privacy
- **No sensitive data** - Repository contains only code and configurations
- **Public sources only** - No proprietary data included
- **Local processing** - All data processing happens on your infrastructure

## Contributing

Contributions are welcome. Please review the contributing guidelines before submitting PRs.

### Dev Setup
```bash
# Get development dependencies
pip install -r requirements-dev.txt

# Code formatting
black src/ tests/
flake8 src/ tests/

# Test w/ coverage
pytest --cov=src --cov-report=html
```

## License

MIT License - see LICENSE file.

## Acknowledgments

- **NEON** - National Ecological Observatory Network for AOP data
- **Fire agencies** - CAL FIRE, USFS, NPS, AZ Forestry
- **OpenWeatherMap** - Weather data API
- **Google Earth Engine** - Satellite data processing platform
- **Research community** - Publications and datasets that made this possible

## Support

- **Bug reports and feature requests** - GitHub Issues
- **Questions and discussions** - GitHub Discussions
- **Documentation** - See docs/ directory

---

**Note**: All NEON data used is publicly available. Fire case studies are based on documented wildfire events.