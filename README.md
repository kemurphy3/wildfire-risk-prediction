# Wildfire Risk Prediction System

Real-time wildfire risk predictions using ML, satellite data, and environmental sensors. Built this to help predict fire risks before they get out of hand.

## What's in here?

- **Real-time data** - pulls live weather, satellite imagery, and sensor data
- **ML models** - Random Forest, XGBoost, LightGBM, ConvLSTM, and ensemble stuff
- **NEON AOP crosswalk** - fancy airborne data that makes predictions way more accurate
- **Fire case studies** - actual fire data for testing and validation
- **Dashboard** - Plotly Dash UI that updates in real-time 
- **API** - FastAPI backend, handles training and validation automatically
- **Latest research** - implemented papers from 2020-2024

## NEON AOP Crosswalk System

We use NEON's airplane data to calibrate satellite imagery. This approach teaches satellites to see better by comparing with super high-resolution airplane scans:

### What's NEON AOP Crosswalk?
- **High-res ground truth** - 1m resolution from planes (LiDAR, hyperspectral, RGB cameras)
- **Satellite calibration** - trains 10-30m satellite data using the airplane measurements
- **Better accuracy** - way better at spotting fire-prone vegetation patterns
- **Free data!** - all NEON data is public (thank u science)

### Fire Case Study Sites (High Priority)
- **GRSM**: Great Smoky Mountains - 2016 Chimney Tops 2 Fire (11k hectares)
- **SOAP**: Soaproot Saddle - 2020 Creek Fire (153k ha) & 2021 Blue Fire (8.5k ha)
- **SYCA**: Sycamore Creek - 2024 Sand Stone Fire (3.2k ha)

### Ecosystem Diversity Sites (Baseline)
- **SRER**: Santa Rita Range, AZ - desert grassland
- **JORN**: Jornada Range, NM - Chihuahuan Desert (hot & dry)
- **ONAQ**: Onaqui, UT - sagebrush everywhere
- **SJER**: San Joaquin, CA - oak trees & golden hills

### Data Products
- **CHM** - 1m res vegetation height maps
- **Hyperspectral** - 426 bands (380-2510nm) for detailed analysis
- **RGB imagery** - 10cm resolution (you can see individual bushes)
- **LiDAR point clouds** - super dense 3D data

### Crosswalk Models
- **Linear** - Ridge regression (simple but effective)
- **Ensemble** - Gradient Boosting for the tricky stuff
- **Validation** - R², MAE, bias... all the metrics

### Fire Research Capabilities
- **Before/after analysis** - see what changed from fires
- **Severity mapping** - quantify burn severity across affected areas
- **Recovery tracking** - watch nature bounce back over time
- **Temporal validation** - make sure our models work across different fire events

## Installation

### Quick Start
```bash
# Grab the code
git clone https://github.com/yourusername/wildfire-risk-prediction.git
cd wildfire-risk-prediction

# Install dependencies
python install_dependencies.py

# Setup env vars
python create_env_file.py

# Start the application
python run_dashboard.py
```

### Dependencies
- **Core**: `requirements.txt` - the essentials
- **Minimal**: `requirements-minimal.txt` - minimum requirements to run predictions
- **Dev**: `requirements-dev.txt` - testing, linting, and development tools

## Usage

### NEON AOP Crosswalk
```bash
# Get fire site data (warning: these are BIG downloads)
make download-GRSM    # Smoky Mountains fire
make download-SOAP    # Soaproot fire (the big one)
make download-SYCA    # Sycamore Creek

# Get baseline sites
make download-SRER    # Santa Rita
make download-JORN    # Jornada 

# Process everything
make process          # extract features
make calibrate        # train models
make validate         # check if it actually works

# Or run everything at once
make all             # does everything above
```

### Python API
```python
from src.integration.aop_integration import AOPIntegrationManager

# Setup manager
manager = AOPIntegrationManager('configs/aop_sites.yaml')

# Get enhanced features (before & after fire)
enhanced_data = manager.get_enhanced_features(satellite_data, 'GRSM', 2016)  # before
enhanced_data = manager.get_enhanced_features(satellite_data, 'GRSM', 2017)  # after

# Check if everything's working
validation = manager.validate_integration('SOAP', 2020)  # that massive Creek Fire
```

### Dashboard
- **Real-time monitoring** - watch fire risk live
- **Interactive maps** - fully interactive with zoom and pan capabilities
- **Model insights** - see what features matter most
- **Export data** - CSV, JSON, whatever you need
- **Fire impact viz** - before/after comparisons

## The Science Behind It

We've implemented a bunch of recent research (2020-2024). Here's what we're using:

### Modern Fire Indices
- **Enhanced FWI** - better weather-based fire danger (finally!)
- **HDW Index** - for those crazy wind-driven fires
- **ML Fire Potential** - because why not throw ML at everything?

### Satellite Features
- **VIIRS** - spots active fires in real-time
- **Sentinel-5P** - tracks CO levels (smoke detection)
- **ECOSTRESS** - measures plant water stress (thirsty plants = fire risk)

### Environmental Factors
- **WUI** - proximity to homes/buildings (critical factor)
- **SVI** - which communities are most at risk
- **Lightning density** - nature's fire starters

### Fire Research Apps
- **Severity assessment** - detailed burn maps from AOP data
- **Recovery monitoring** - watch the green come back
- **Fuel load** - how much stuff is there to burn?
- **Ecosystem resilience** - will it bounce back? (hopefully)

## Project Structure

```
wildfire-risk-prediction/
├── src/
│   ├── dashboard/          # web UI (Plotly Dash)
│   ├── data_collection/    # data downloaders
│   ├── features/           # feature eng + AOP stuff
│   ├── models/             # ML models
│   ├── integration/        # glue code for AOP
│   └── utils/              # helpers, geo tools, etc
├── configs/                # yaml configs
├── tests/                  # tests (we actually have them!)
├── notebooks/              # jupyter experiments
├── data/                   # big files (gitignored)
├── logs/                   # debug logs
└── docs/                   # more docs
```

## Testing

Run tests (yes, we have tests!):
```bash
# Run everything
python -m pytest

# Just AOP tests
python -m pytest tests/test_aop_integration.py -v

# Feature tests only
python -m pytest tests/test_features/ -v
```

## Docs

- **Quick Start**: `QUICK_START.md` - get going fast
- **Project Structure**: `PROJECT_STRUCTURE.md` - what's where
- **NEON AOP**: `PUBLIC_DATA_ACCESS.md` - how to get the data
- **API Docs**: `docs/api/` - all the functions

## Data Sources

### Public Data (mostly free!)
- **NEON AOP** - all public through their portal
- **Fire case studies** - real fires with before/after data
- **OpenWeatherMap** - free tier (need API key tho)
- **Google Earth Engine** - free for research/edu

### Privacy stuff
- **No sensitive data** - just code & configs here
- **All public sources** - nothing proprietary 
- **Your data stays local** - download & process on your machine

## Contributing

PRs welcome! Check out contributing guidelines first.

### Dev Setup
```bash
# Get development dependencies
pip install -r requirements-dev.txt

# Format code
black src/ tests/
flake8 src/ tests/

# Test w/ coverage
pytest --cov=src --cov-report=html
```

## License

MIT License - see LICENSE file.

## Thanks to

- **NEON** - for the amazing AOP data
- **Fire agencies** - CAL FIRE, USFS, NPS, AZ Forestry
- **OpenWeatherMap** - weather data
- **Google Earth Engine** - satellite processing power
- **Researchers** - standing on the shoulders of giants here

## Need Help?

- **Bugs/features** - GitHub Issues
- **Questions** - GitHub Discussions 
- **Docs** - check the docs/ folder

---

**Note**: All the NEON data we use is publicly available. The fire case studies are based on real events.