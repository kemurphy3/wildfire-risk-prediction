"""
Configuration Template for Wildfire Risk Prediction System

This is a template file. Copy this to 'config.py' and fill in your own API keys.

IMPORTANT: Never commit your actual API keys to Git!
"""

import os
from typing import Dict, Any
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_file = Path('.env')
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment variables from {env_file}")
    else:
        print("No .env file found. Run: python create_env_file.py")
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")

# API Keys and External Services
# Replace these with your actual API keys from the respective services
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', 'your_openweather_api_key_here')
NEON_API_TOKEN = os.getenv('NEON_API_TOKEN', 'your_neon_api_token_here')

# Google Earth Engine (FREE for Research)
# Get access from: https://earthengine.google.com/
# After approval, run: earthengine authenticate
# For personal auth: leave empty (run 'earthengine authenticate' in terminal)
# For service account: provide path to JSON file
GOOGLE_EARTH_ENGINE_CREDENTIALS = os.getenv('GOOGLE_EARTH_ENGINE_CREDENTIALS', '')

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'host': '0.0.0.0',
    'port': 8050,
    'debug': True,
    'title': 'Wildfire Risk Prediction Dashboard',
    'theme': 'cosmo'
}

# Data Source Configuration
DATA_SOURCES = {
    'openweather': {
        'base_url': 'https://api.openweathermap.org/data/2.5',
        'api_key': OPENWEATHER_API_KEY,
        'units': 'metric'
    },
    'neon': {
        'base_url': 'https://data.neonscience.org/api/v0',
        'api_token': NEON_API_TOKEN
    },
    'google_earth_engine': {
        'credentials': GOOGLE_EARTH_ENGINE_CREDENTIALS,
        'project': 'wildfire-risk'
    }
}

# Geographic Configuration
GEO_CONFIG = {
    'default_region': 'California',
    'bounding_box': {
        'min_lat': 32.5,
        'max_lat': 42.0,
        'min_lon': -124.5,
        'max_lon': -114.0
    },
    'default_center': {
        'lat': 37.0,
        'lon': -119.0
    }
}

# Machine Learning Configuration
ML_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'convlstm': {
        'time_steps': 10,
        'spatial_dims': (32, 32),
        'channels': 10,
        'filters': 64,
        'kernel_size': 3
    }
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'weather_features': ['temperature', 'humidity', 'wind_speed', 'precipitation'],
    'vegetation_features': ['ndvi', 'nbr', 'ndwi', 'evi'],
    'topographical_features': ['elevation', 'slope', 'aspect', 'roughness'],
    'fire_history_features': ['fire_frequency', 'years_since_fire', 'fire_severity']
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'workers': 4,
    'debug': False,
    'title': 'Wildfire Risk Prediction API',
    'version': '1.0.0',
    'description': 'Comprehensive wildfire risk prediction API with machine learning models'
}

# Cache Configuration
CACHE_CONFIG = {
    'ttl': 3600,  # 1 hour
    'max_size': 10000,
    'backend': 'memory'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/wildfire_risk.log'
}

def get_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary."""
    return {
        'api_keys': {
            'openweather': OPENWEATHER_API_KEY,
            'neon': NEON_API_TOKEN,
            'google_earth_engine': GOOGLE_EARTH_ENGINE_CREDENTIALS
        },
        'dashboard': DASHBOARD_CONFIG,
        'data_sources': DATA_SOURCES,
        'geo': GEO_CONFIG,
        'ml': ML_CONFIG,
        'features': FEATURE_CONFIG,
        'api': API_CONFIG,
        'cache': CACHE_CONFIG,
        'logging': LOGGING_CONFIG
    }

def validate_config() -> bool:
    """Validate that required configuration is present."""
    config = get_config()
    
    # Check if API keys are configured
    api_keys = config['api_keys']
    
    if not api_keys['openweather'] or api_keys['openweather'] == 'your_openweather_api_key_here':
        print("OpenWeather API key not configured")
        return False
    
    if not api_keys['neon'] or api_keys['neon'] == 'your_neon_api_token_here':
        print("NEON API token not configured")
        return False
    
    print("All required API keys are configured")
    return True

if __name__ == "__main__":
    print("Configuration Template")
    print("=" * 50)
    print("This is a template file. To use it:")
    print("1. Copy this file to 'config.py'")
    print("2. Run 'python create_env_file.py' to set up your API keys")
    print("3. Or manually edit the values in config.py")
    print()
    
    if validate_config():
        print("Configuration is valid!")
    else:
        print("Please configure your API keys before running the system")
