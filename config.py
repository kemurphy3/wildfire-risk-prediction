# Configuration settings for the wildfire prediction application
# Contains API keys, dashboard settings, and service configurations

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
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')
NEON_API_TOKEN = os.getenv('NEON_API_TOKEN', '')
# Google Earth Engine (FREE for Research)
# Get access from: https://earthengine.google.com/
# After approval, run: earthengine authenticate
# For personal auth: leave empty (run 'earthengine authenticate' in terminal)
# For service account: provide path to JSON file
GOOGLE_EARTH_ENGINE_CREDENTIALS = os.getenv('GOOGLE_EARTH_ENGINE_CREDENTIALS', '')

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'title': 'Wildfire Risk Prediction Dashboard',
    'description': 'Real-time wildfire risk assessment using satellite data and environmental monitoring',
    'theme': {
        'primary_color': '#2E8B57',  # Sea Green
        'secondary_color': '#FF6B35',  # Orange
        'background_color': '#f8f9fa',
        'text_color': '#333333'
    },
    'map': {
        'default_center': {'lat': 37.7749, 'lon': -122.4194},  # San Francisco
        'default_zoom': 6,
        'mapbox_style': 'open-street-map'
    },
    'risk_thresholds': {
        'low': 30,
        'moderate': 60,
        'high': 80,
        'extreme': 100
    }
}

# Data Source Configuration
DATA_SOURCES = {
    'satellite': {
        'enabled': True,
        'collections': ['sentinel2', 'modis', 'landsat'],
        'update_frequency': 'daily',
        'cloud_filter': 30.0,
        'temporal_composite': 'monthly'
    },
    'weather': {
        'enabled': True,
        'provider': 'openweathermap',
        'update_frequency': 'hourly',
        'cache_ttl': 3600
    },
    'topography': {
        'enabled': True,
        'data_source': 'usgs',
        'resolution': '30m',
        'update_frequency': 'yearly'
    },
    'fire_history': {
        'enabled': True,
        'data_source': 'cal_fire',
        'update_frequency': 'monthly',
        'search_radius_km': 100
    }
}

# Machine Learning Model Configuration
ML_CONFIG = {
    'models': {
        'random_forest': {
            'enabled': True,
            'n_estimators': 200,
            'max_depth': 20,
            'random_state': 42
        },
        'xgboost': {
            'enabled': True,
            'n_estimators': 200,
            'early_stopping_rounds': 10,
            'random_state': 42
        },
        'convlstm': {
            'enabled': False,  # Disabled until fully tested
            'time_steps': 10,
            'spatial_dims': (32, 32),
            'channels': 10
        },
        'ensemble': {
            'enabled': True,
            'method': 'stacking',
            'base_models': ['random_forest', 'xgboost']
        }
    },
    'training': {
        'test_size': 0.2,
        'validation_size': 0.2,
        'cross_validation_folds': 5,
        'hyperparameter_tuning': True
    }
}

# API Configuration
API_CONFIG = {
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', 8000)),
    'workers': int(os.getenv('API_WORKERS', 4)),
    'debug': os.getenv('API_DEBUG', 'false').lower() == 'true',
    'cors_origins': ['http://localhost:8050', 'http://127.0.0.1:8050'],
    'rate_limit': {
        'requests_per_minute': 100,
        'burst_size': 20
    }
}

# Dashboard Server Configuration
DASHBOARD_SERVER_CONFIG = {
    'host': os.getenv('DASHBOARD_HOST', '0.0.0.0'),
    'port': int(os.getenv('DASHBOARD_PORT', 8050)),
    'debug': os.getenv('DASHBOARD_DEBUG', 'true').lower() == 'true',
    'auto_reload': True
}

# Cache Configuration
CACHE_CONFIG = {
    'enabled': True,
    'ttl': int(os.getenv('CACHE_TTL', 3600)),  # 1 hour
    'max_size': int(os.getenv('CACHE_MAX_SIZE', 10000)),
    'backend': 'memory'  # Options: memory, redis, filesystem
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.getenv('LOG_FILE', 'wildfire_risk.log'),
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Geographic Configuration
GEO_CONFIG = {
    'default_region': 'california',
    'bounding_box': {
        'min_lat': 32.5,
        'max_lat': 42.0,
        'min_lon': -124.5,
        'max_lon': -114.0
    },
    'grid_resolution': {
        'default': 50,  # 50x50 grid
        'high_resolution': 100,  # 100x100 grid
        'low_resolution': 25     # 25x25 grid
    }
}

# Risk Assessment Configuration
RISK_CONFIG = {
    'factors': {
        'weather': {
            'weight': 0.35,
            'subfactors': ['temperature', 'humidity', 'wind_speed', 'precipitation']
        },
        'vegetation': {
            'weight': 0.25,
            'subfactors': ['ndvi', 'nbr', 'ndwi', 'fuel_moisture']
        },
        'topography': {
            'weight': 0.20,
            'subfactors': ['elevation', 'slope', 'aspect', 'roughness']
        },
        'fire_history': {
            'weight': 0.20,
            'subfactors': ['frequency', 'severity', 'years_since_fire']
        }
    },
    'categories': {
        'low': {'min': 0, 'max': 30, 'color': '#00FF00'},
        'moderate': {'min': 30, 'max': 60, 'color': '#FFFF00'},
        'high': {'min': 60, 'max': 80, 'color': '#FFA500'},
        'extreme': {'min': 80, 'max': 100, 'color': '#FF0000'}
    }
}

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.
    
    Returns:
        Dictionary containing all configuration settings
    """
    return {
        'dashboard': DASHBOARD_CONFIG,
        'data_sources': DATA_SOURCES,
        'ml_config': ML_CONFIG,
        'api': API_CONFIG,
        'dashboard_server': DASHBOARD_SERVER_CONFIG,
        'cache': CACHE_CONFIG,
        'logging': LOGGING_CONFIG,
        'geo': GEO_CONFIG,
        'risk': RISK_CONFIG,
        'api_keys': {
            'openweather': OPENWEATHER_API_KEY,
            'neon': NEON_API_TOKEN,
            'google_earth_engine': GOOGLE_EARTH_ENGINE_CREDENTIALS
        }
    }

def validate_config() -> bool:
    """
    Validate the configuration settings.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        config = get_config()
        
        # Check required API keys
        if DATA_SOURCES['weather']['enabled'] and not OPENWEATHER_API_KEY:
            print("Warning: OpenWeather API key not provided, weather data will use demo data")
        
        if DATA_SOURCES['satellite']['enabled'] and not GOOGLE_EARTH_ENGINE_CREDENTIALS:
            print("Warning: Google Earth Engine credentials not provided, satellite data will use demo data")
        
        # Validate port numbers
        if not (1 <= API_CONFIG['port'] <= 65535):
            print(f"Error: Invalid API port number: {API_CONFIG['port']}")
            return False
        
        if not (1 <= DASHBOARD_SERVER_CONFIG['port'] <= 65535):
            print(f"Error: Invalid dashboard port number: {DASHBOARD_SERVER_CONFIG['port']}")
            return False
        
        # Validate risk weights sum to 1.0
        total_weight = sum(factor['weight'] for factor in RISK_CONFIG['factors'].values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"Warning: Risk factor weights sum to {total_weight}, should be 1.0")
        
        print("Configuration validation completed successfully")
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    # Validate configuration when run directly
    validate_config()
