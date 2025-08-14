# Setup Guide for Real Data Integration

This guide will help you set up the wildfire risk prediction system with real data sources.

## Prerequisites

- Python 3.8+
- Git
- Internet connection for API access

## Installation

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

## API Keys Setup

### 1. OpenWeather API (Free)
- Go to [OpenWeatherMap](https://openweathermap.org/api)
- Sign up for a free account
- Get your API key
- Set environment variable:
  ```bash
   export OPENWEATHER_API_KEY=your_api_key_here
   ```

### 2. NEON Data Access (Optional)
- Visit [NEON Data Portal](https://data.neonscience.org/)
- Create an account
- Get your API token
- Set environment variable:
  ```bash
   export NEON_API_TOKEN=your_token_here
   ```

### 3. Google Earth Engine (Optional)
- Follow [Google Earth Engine Setup Guide](https://developers.google.com/earth-engine/guides/getstarted)
- Authenticate with `earthengine authenticate`
- Set environment variable:
  ```bash
   export GOOGLE_EARTH_ENGINE_CREDENTIALS=path/to/credentials.json
   ```

## Environment Configuration

Create a `.env` file in the project root:

```bash
# Weather Data API
OPENWEATHER_API_KEY=your_openweather_api_key_here

# NEON Data Access
NEON_API_TOKEN=your_neon_token_here

# Google Earth Engine
GOOGLE_EARTH_ENGINE_CREDENTIALS=path/to/your/credentials.json

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_DEBUG=false

# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050
DASHBOARD_DEBUG=true

# Cache Configuration
CACHE_TTL=3600
CACHE_MAX_SIZE=10000

# Logging
LOG_LEVEL=INFO
LOG_FILE=wildfire_risk.log
```

## Running the System

### 1. Start the Dashboard
```bash
python run_dashboard.py
```
Open: http://localhost:8050

### 2. Start the API Server
```bash
python -m src.api.main
```
API docs: http://localhost:8000/docs

## Data Source Status

The system will automatically detect available data sources:

- **✅ Real Data**: When API keys are properly configured
- **⚠️ Demo Data**: When falling back to synthetic data
- **❌ Error**: When there are configuration issues

## Troubleshooting

### Common Issues

1. **"No OpenWeather API key provided"**
   - Set the `OPENWEATHER_API_KEY` environment variable
   - Or add it to your `.env` file

2. **"Satellite client not available"**
   - Install Google Earth Engine: `pip install earthengine-api`
   - Authenticate: `earthengine authenticate`

3. **"Geopandas import error"**
   - Install system dependencies: `conda install geopandas`
   - Or use: `pip install geopandas --no-binary geopandas`

### Performance Tips

- **Grid Resolution**: Adjust `grid_size` in `get_grid_risk_assessment()` for performance vs. detail
- **Cache Settings**: Modify `CACHE_TTL` for data freshness vs. API rate limits
- **Update Frequency**: Adjust data source update frequencies in `config.py`

## Data Quality

### Real Data Sources
- **Weather**: Hourly updates from OpenWeatherMap
- **Satellite**: Daily updates from Google Earth Engine
- **Topography**: Static data from USGS (yearly updates)
- **Fire History**: Monthly updates from CAL FIRE

### Demo Data
- **Weather**: Realistic California weather patterns
- **Satellite**: Location-based vegetation indices
- **Topography**: Coastal-to-mountain elevation gradients
- **Fire History**: Southern California fire patterns

## Next Steps

1. **Customize Risk Factors**: Modify weights in `config.py`
2. **Add New Data Sources**: Extend `RealDataIntegration` class
3. **Deploy to Production**: Use Docker and production web servers
4. **Add Authentication**: Implement user management and access control

## Support

- **Issues**: Use GitHub Issues for bug reports
- **Documentation**: Check docstrings and this guide
- **Community**: Join discussions for questions and ideas
