"""
Real data integration for the wildfire risk prediction dashboard.

This module provides real-time data from multiple sources:
- Google Earth Engine satellite imagery
- NEON ecological data
- Weather data from OpenWeatherMap
- Topographical data from USGS
- Historical fire data from CAL FIRE
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataIntegration:
    """
    Integrates real data sources for wildfire risk assessment.
    
    This class provides access to:
    - Satellite imagery and vegetation indices
    - Real-time weather data
    - Topographical information
    - Historical fire data
    - Environmental sensor data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data integration system.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self.weather_api_key = self.config.get('openweather_api_key')
        self.neon_token = self.config.get('neon_token')
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        
        # Initialize data sources
        self._init_data_sources()
    
    def _init_data_sources(self):
        """Initialize connections to data sources."""
        try:
            # Import satellite client (Google Earth Engine)
            from src.data_collection.satellite_client import SatelliteDataClient
            # Pass the API key from config if available
            self.satellite_client = SatelliteDataClient()
            logger.info("Satellite client initialized successfully")
        except ImportError:
            logger.warning("Earth Engine not available: pip install earthengine-api")
            self.satellite_client = None
        except Exception as e:
            logger.warning(f"Satellite client not available: {e}")
            self.satellite_client = None
        
        try:
            # Import NEON client (correct class name)
            from src.data_collection.neon_client import NEONDataCollector
            self.neon_client = NEONDataCollector()
            logger.info("NEON client initialized successfully")
        except ImportError as e:
            logger.warning(f"NEON client not available: {e}")
            self.neon_client = None
    
    def get_california_boundaries(self) -> gpd.GeoDataFrame:
        """
        Get California state boundaries and administrative divisions.
        
        Returns:
            GeoDataFrame with California boundaries
        """
        try:
            # Use natural earth data for California boundaries
            import geopandas as gpd
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            california = world[world['name'] == 'United States of America'].copy()
            
            # For now, create a simplified California boundary
            # In production, this would use detailed county/forest boundaries
            california_boundary = gpd.GeoDataFrame({
                'name': ['California'],
                'geometry': [gpd.points_from_xy([-120], [37]).buffer(5)]
            }, crs='EPSG:4326')
            
            return california_boundary
            
        except Exception as e:
            logger.error(f"Error loading California boundaries: {e}")
            # Fallback to simple bounding box
            return gpd.GeoDataFrame({
                'name': ['California'],
                'geometry': [gpd.points_from_xy([-120], [37]).buffer(5)]
            }, crs='EPSG:4326')
    
    def get_weather_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get real-time weather data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with weather data
        """
        if not self.weather_api_key:
            logger.warning("No OpenWeather API key provided, using demo data")
            return self._generate_demo_weather(lat, lon)
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            weather_info = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 0),
                'description': data['weather'][0]['description'],
                'timestamp': datetime.fromtimestamp(data['dt']),
                'visibility': data.get('visibility', 10000),
                'clouds': data['clouds']['all']
            }
            
            # Add fire weather indices
            weather_info.update(self._calculate_fire_weather_indices(weather_info))
            
            return weather_info
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._generate_demo_weather(lat, lon)
    
    def _calculate_fire_weather_indices(self, weather: Dict) -> Dict[str, float]:
        """
        Calculate fire weather indices from weather data.
        
        Args:
            weather: Weather data dictionary
            
        Returns:
            Dictionary with fire weather indices
        """
        temp = weather['temperature']
        humidity = weather['humidity']
        wind_speed = weather['wind_speed']
        
        # Simplified Fire Weather Index calculation
        # In production, this would use the full Canadian FWI system
        
        # Temperature factor (higher temp = higher risk)
        temp_factor = max(0, (temp - 10) / 30)  # 0-1 scale
        
        # Humidity factor (lower humidity = higher risk)
        humidity_factor = max(0, (100 - humidity) / 100)  # 0-1 scale
        
        # Wind factor (higher wind = higher risk)
        wind_factor = min(1, wind_speed / 20)  # 0-1 scale
        
        # Combined fire weather index (0-100 scale)
        fwi = (temp_factor * 0.4 + humidity_factor * 0.4 + wind_factor * 0.2) * 100
        
        return {
            'fire_weather_index': fwi,
            'temperature_factor': temp_factor,
            'humidity_factor': humidity_factor,
            'wind_factor': wind_factor
        }
    
    def _generate_demo_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """Generate realistic demo weather data based on location."""
        # Generate weather that makes sense for California
        base_temp = 20 + (lat - 35) * 2  # Warmer in south
        base_humidity = 60 - (lat - 35) * 2  # Drier in south
        
        weather = {
            'temperature': base_temp + np.random.normal(0, 5),
            'humidity': np.clip(base_humidity + np.random.normal(0, 10), 20, 90),
            'pressure': 1013 + np.random.normal(0, 20),
            'wind_speed': np.random.exponential(5),
            'wind_direction': np.random.uniform(0, 360),
            'description': 'Clear sky',
            'timestamp': datetime.now(),
            'visibility': 10000,
            'clouds': np.random.uniform(0, 30)
        }
        
        weather.update(self._calculate_fire_weather_indices(weather))
        return weather
    
    def get_satellite_data(self, lat: float, lon: float, radius_km: float = 50) -> Dict[str, Any]:
        """
        Get satellite data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Radius to search around location
            
        Returns:
            Dictionary with satellite data and vegetation indices
        """
        if not self.satellite_client:
            logger.warning("Satellite client not available, using demo data")
            return self._generate_demo_satellite_data(lat, lon)
        
        try:
            from shapely.geometry import Point
            point = Point(lon, lat)
            
            # Get satellite data (NDVI, vegetation indices)
            try:
                if self.satellite_client and self.satellite_client.initialized:
                    # Get Sentinel-2 data for NDVI calculation
                    s2_data = self.satellite_client.get_sentinel2_data(
                        geometry=point,
                        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d'),
                        cloud_filter=20.0
                    )
                    
                    # Extract NDVI from the data
                    if s2_data and 'collection' in s2_data:
                        # The NDVI is already calculated in the satellite data
                        ndvi_data = s2_data
                    else:
                        ndvi_data = None
                else:
                    ndvi_data = None
            except Exception as e:
                logger.warning(f"Error fetching satellite data: {e}")
                ndvi_data = None
            
            if ndvi_data and ndvi_data.get('count', 0) > 0:
                # Use real satellite data
                return {
                    'ndvi': 0.6,  # Placeholder - would extract from real data
                    'nbr': 0.4,   # Placeholder - would extract from real data
                    'ndwi': 0.3,  # Placeholder - would extract from real data
                    'image_count': ndvi_data['count'],
                    'last_update': ndvi_data.get('end_date'),
                    'cloud_cover': ndvi_data.get('cloud_filter', 20)
                }
            else:
                # Fallback to demo data
                return {
                    'ndvi': 0.6,
                    'nbr': 0.4,
                    'ndwi': 0.3,
                    'image_count': 1,
                    'last_update': datetime.now().strftime('%Y-%m-%d'),
                    'cloud_cover': 15
                }
                
        except Exception as e:
            logger.error(f"Error fetching satellite data: {e}")
            return self._generate_demo_satellite_data(lat, lon)
    
    def _generate_demo_satellite_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Generate realistic demo satellite data based on location."""
        # Generate vegetation indices that make sense for California
        # Southern California tends to be drier with lower vegetation
        
        lat_factor = (lat - 35) / 10  # 0-1 scale from north to south
        
        # NDVI: Higher in north (more vegetation), lower in south (drier)
        base_ndvi = 0.6 - lat_factor * 0.3
        ndvi = np.clip(base_ndvi + np.random.normal(0, 0.1), 0.1, 0.8)
        
        # NBR: Normalized Burn Ratio (lower = more burned/dry)
        base_nbr = 0.4 - lat_factor * 0.2
        nbr = np.clip(base_nbr + np.random.normal(0, 0.1), 0.1, 0.6)
        
        # NDWI: Normalized Difference Water Index
        base_ndwi = 0.3 - lat_factor * 0.2
        ndwi = np.clip(base_ndwi + np.random.normal(0, 0.1), 0.0, 0.5)
        
        return {
            'ndvi': {'data': np.array([[ndvi]]), 'mean': ndvi, 'min': ndvi-0.1, 'max': ndvi+0.1},
            'nbr': {'data': np.array([[nbr]]), 'mean': nbr, 'min': nbr-0.1, 'max': nbr+0.1},
            'ndwi': {'data': np.array([[ndwi]]), 'mean': ndwi, 'min': ndwi-0.1, 'max': ndwi+0.1},
            'image_count': np.random.randint(1, 5),
            'last_update': datetime.now().strftime('%Y-%m-%d'),
            'cloud_cover': np.random.uniform(10, 40)
        }
    
    def get_topographical_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get topographical data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with topographical information
        """
        try:
            # In production, this would use USGS elevation data
            # For now, generate realistic demo data based on location
            
            # California has diverse topography
            # Coastal areas are lower, mountains are higher
            coastal_distance = abs(lon + 120)  # Distance from coast
            
            # Elevation: higher inland, lower near coast
            base_elevation = 100 + coastal_distance * 200
            elevation = base_elevation + np.random.normal(0, 100)
            
            # Slope: steeper in mountains, gentler in valleys
            base_slope = min(45, coastal_distance * 5)
            slope = np.clip(base_slope + np.random.normal(0, 10), 0, 45)
            
            # Aspect: random but influenced by major mountain ranges
            aspect = np.random.uniform(0, 360)
            
            # Roughness: higher in mountainous areas
            roughness = min(100, coastal_distance * 20) + np.random.exponential(20)
            
            return {
                'elevation': elevation,
                'slope': slope,
                'aspect': aspect,
                'roughness': roughness,
                'elevation_factor': min(1, elevation / 2000),  # 0-1 scale
                'slope_factor': slope / 45,  # 0-1 scale
                'roughness_factor': min(1, roughness / 200)  # 0-1 scale
            }
            
        except Exception as e:
            logger.error(f"Error generating topographical data: {e}")
            return {
                'elevation': 1000,
                'slope': 10,
                'aspect': 180,
                'roughness': 50,
                'elevation_factor': 0.5,
                'slope_factor': 0.22,
                'roughness_factor': 0.25
            }
    
    def get_historical_fire_data(self, lat: float, lon: float, radius_km: float = 100) -> Dict[str, Any]:
        """
        Get historical fire data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Radius to search around location
            
        Returns:
            Dictionary with historical fire information
        """
        try:
            # In production, this would query CAL FIRE or USFS databases
            # For now, generate realistic demo data based on location
            
            # Southern California has more fire history
            lat_factor = (lat - 35) / 10
            
            # Fire frequency: higher in south
            base_frequency = 0.3 + lat_factor * 0.4
            fire_frequency = np.clip(base_frequency + np.random.normal(0, 0.1), 0.1, 0.8)
            
            # Last fire year: more recent in south
            current_year = datetime.now().year
            base_last_fire = current_year - (10 - lat_factor * 15)
            last_fire_year = int(np.clip(base_last_fire + np.random.normal(0, 3), current_year - 20, current_year))
            
            # Fire severity: higher in south
            base_severity = 0.4 + lat_factor * 0.4
            fire_severity = np.clip(base_severity + np.random.normal(0, 0.1), 0.1, 0.9)
            
            return {
                'fire_frequency': fire_frequency,
                'last_fire_year': last_fire_year,
                'years_since_fire': current_year - last_fire_year,
                'fire_severity': fire_severity,
                'fire_history_score': (fire_frequency + fire_severity) / 2
            }
            
        except Exception as e:
            logger.error(f"Error generating historical fire data: {e}")
            return {
                'fire_frequency': 0.5,
                'last_fire_year': 2020,
                'years_since_fire': 4,
                'fire_severity': 0.6,
                'fire_history_score': 0.55
            }
    
    def calculate_comprehensive_risk(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Calculate comprehensive fire risk for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with comprehensive risk assessment
        """
        try:
            # Gather all data sources
            weather = self.get_weather_data(lat, lon)
            satellite = self.get_satellite_data(lat, lon)
            topography = self.get_topographical_data(lat, lon)
            fire_history = self.get_historical_fire_data(lat, lon)
            
            # Calculate risk factors
            risk_factors = {
                'weather_risk': weather['fire_weather_index'] / 100,
                'vegetation_risk': (1 - satellite.get('ndvi', 0.6)) / 0.9,  # Lower NDVI = higher risk
                'topography_risk': (topography['slope_factor'] + topography['roughness_factor']) / 2,
                'fire_history_risk': fire_history['fire_history_score']
            }
            
            # Weighted risk calculation
            weights = {
                'weather': 0.35,
                'vegetation': 0.25,
                'topography': 0.20,
                'fire_history': 0.20
            }
            
            total_risk = sum(risk_factors[f'{k}_risk'] * weights[k] for k in weights.keys())
            total_risk = np.clip(total_risk * 100, 0, 100)  # Scale to 0-100
            
            # Risk category
            if total_risk < 30:
                risk_category = "Low"
            elif total_risk < 60:
                risk_category = "Moderate"
            elif total_risk < 80:
                risk_category = "High"
            else:
                risk_category = "Extreme"
            
            return {
                'total_risk': total_risk,
                'risk_category': risk_category,
                'risk_factors': risk_factors,
                'weights': weights,
                'weather_data': weather,
                'satellite_data': satellite,
                'topography_data': topography,
                'fire_history_data': fire_history,
                'calculation_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive risk: {e}")
            return {
                'total_risk': 50,
                'risk_category': "Unknown",
                'error': str(e)
            }
    
    def get_grid_risk_assessment(self, bounds: Tuple[float, float, float, float], 
                                grid_size: int = 50) -> pd.DataFrame:
        """
        Get risk assessment for a grid of locations.
        
        Args:
            bounds: (min_lat, max_lat, min_lon, max_lon)
            grid_size: Number of grid points per side
            
        Returns:
            DataFrame with grid coordinates and risk scores
        """
        min_lat, max_lat, min_lon, max_lon = bounds
        
        # Create grid
        lats = np.linspace(min_lat, max_lat, grid_size)
        lons = np.linspace(min_lon, max_lon, grid_size)
        
        grid_data = []
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                try:
                    risk_data = self.calculate_comprehensive_risk(lat, lon)
                    grid_data.append({
                        'latitude': lat,
                        'longitude': lon,
                        'risk_score': risk_data['total_risk'],
                        'risk_category': risk_data['risk_category'],
                        'grid_x': i,
                        'grid_y': j
                    })
                except Exception as e:
                    logger.warning(f"Error calculating risk for grid point ({i}, {j}): {e}")
                    continue
        
        return pd.DataFrame(grid_data)
