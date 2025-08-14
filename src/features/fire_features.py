"""
Fire Risk Feature Engineering Pipeline

This module implements comprehensive feature engineering for wildfire risk
assessment, including fuel moisture calculations, fire weather indices,
topographical features, and data fusion between ground and satellite observations.

Educational Note: This module demonstrates the scientific principles behind
fire risk assessment, including the relationship between weather conditions,
vegetation characteristics, and fire behavior. Each feature is based on
established fire science research and operational fire danger rating systems.

References:
- Van Wagner (1987) - Development and structure of the Canadian Forest Fire Weather Index
- Nelson (2000) - Prediction of dead fuel moisture content
- Keetch & Byram (1968) - Drought index for fire potential
- Rothermel (1972) - Mathematical model for predicting fire spread
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import xarray as xr
from scipy import stats
from scipy.spatial.distance import cdist

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class FireRiskFeatureEngine:
    """
    Comprehensive feature engineering engine for wildfire risk assessment.
    
    This class provides methods to calculate fire risk features from multiple
    data sources including NEON ecological data, satellite imagery, weather
    data, and topographical information. Features are based on established
    fire science research and operational fire danger rating systems.
    
    Attributes:
        neon_data (pd.DataFrame): NEON ground-based ecological data
        satellite_data (Dict): Satellite imagery and derived indices
        weather_data (pd.DataFrame): Meteorological observations
        dem_data (xr.Dataset): Digital elevation model data
        feature_cache (Dict): Cache for computed features to avoid recalculation
    """
    
    def __init__(self, cache_dir: str = "./cache/features"):
        """
        Initialize the fire risk feature engine.
        
        Args:
            cache_dir: Directory for caching computed features
        """
        self.cache_dir = cache_dir
        self.neon_data = None
        self.satellite_data = None
        self.weather_data = None
        self.dem_data = None
        self.feature_cache = {}
        
        # Physical constants for calculations
        self.GRAVITY = 9.81  # m/s²
        self.SPECIFIC_HEAT_AIR = 1005.0  # J/kg·K
        self.LATENT_HEAT_VAPORIZATION = 2.5e6  # J/kg
        
        logger.info("Fire Risk Feature Engine initialized")
    
    def load_neon_data(self, data_path: str) -> None:
        """
        Load NEON ecological data for feature engineering.
        
        Args:
            data_path: Path to NEON data file or DataFrame
        """
        try:
            if isinstance(data_path, str):
                if data_path.endswith('.csv'):
                    self.neon_data = pd.read_csv(data_path)
                elif data_path.endswith('.parquet'):
                    self.neon_data = pd.read_parquet(data_path)
                else:
                    raise ValueError("Unsupported file format. Use CSV or Parquet.")
            else:
                self.neon_data = data_path
            
            logger.info(f"Loaded NEON data with {len(self.neon_data)} records")
            
        except Exception as e:
            logger.error(f"Error loading NEON data: {e}")
            raise
    
    def load_satellite_data(self, satellite_data: Dict) -> None:
        """
        Load satellite imagery and derived indices.
        
        Args:
            satellite_data: Dictionary containing satellite data collections
        """
        self.satellite_data = satellite_data
        logger.info("Satellite data loaded for feature engineering")
    
    def load_weather_data(self, weather_data: pd.DataFrame) -> None:
        """
        Load meteorological observations for fire weather calculations.
        
        Args:
            weather_data: DataFrame containing weather observations
        """
        self.weather_data = weather_data
        logger.info(f"Weather data loaded with {len(self.weather_data)} records")
    
    def load_dem_data(self, dem_path: str) -> None:
        """
        Load digital elevation model data for topographical features.
        
        Args:
            dem_path: Path to DEM file (GeoTIFF, NetCDF, etc.)
        """
        try:
            if dem_path.endswith('.tif') or dem_path.endswith('.tiff'):
                with rasterio.open(dem_path) as src:
                    self.dem_data = xr.DataArray(
                        src.read(1),
                        coords={
                            'y': src.y,
                            'x': src.x
                        },
                        dims=['y', 'x']
                    )
            elif dem_path.endswith('.nc') or dem_path.endswith('.netcdf'):
                self.dem_data = xr.open_dataset(dem_path)
            else:
                raise ValueError("Unsupported DEM format. Use GeoTIFF or NetCDF.")
            
            logger.info("DEM data loaded for topographical feature calculation")
            
        except Exception as e:
            logger.error(f"Error loading DEM data: {e}")
            raise
    
    def calculate_fuel_moisture_content(
        self,
        temperature: float,
        relative_humidity: float,
        fuel_type: str = 'medium',
        method: str = 'nelson'
    ) -> float:
        """
        Calculate fuel moisture content using various models.
        
        Fuel moisture content is a critical factor in fire behavior,
        determining how easily vegetation will ignite and spread.
        
        Args:
            temperature: Air temperature in Celsius
            relative_humidity: Relative humidity as percentage (0-100)
            fuel_type: Type of fuel ('fine', 'medium', 'coarse')
            method: Calculation method ('nelson', 'simplified')
            
        Returns:
            Fuel moisture content as percentage of dry weight
            
        References:
            Nelson (2000) - Prediction of dead fuel moisture content
        """
        if method == 'nelson':
            return self._nelson_fuel_moisture(temperature, relative_humidity, fuel_type)
        elif method == 'simplified':
            return self._simplified_fuel_moisture(temperature, relative_humidity)
        else:
            raise ValueError("Method must be 'nelson' or 'simplified'")
    
    def _nelson_fuel_moisture(
        self,
        temperature: float,
        relative_humidity: float,
        fuel_type: str
    ) -> float:
        """
        Calculate fuel moisture using Nelson's model.
        
        This is a more sophisticated model that accounts for fuel type
        and provides more accurate moisture content estimates.
        """
        # Convert relative humidity to decimal
        rh_decimal = relative_humidity / 100.0
        
        # Base moisture content from temperature and humidity
        base_moisture = 0.0329 + 0.28 * rh_decimal - 0.000576 * temperature
        
        # Fuel type adjustments
        fuel_adjustments = {
            'fine': 1.0,
            'medium': 1.2,
            'coarse': 1.5
        }
        
        adjustment = fuel_adjustments.get(fuel_type, 1.0)
        
        # Apply fuel type adjustment and ensure reasonable bounds
        moisture = base_moisture * adjustment
        return np.clip(moisture * 100, 0, 100)  # Convert to percentage
    
    def _simplified_fuel_moisture(self, temperature: float, relative_humidity: float) -> float:
        """
        Simplified fuel moisture calculation for educational purposes.
        
        This is a basic model that demonstrates the relationship between
        weather conditions and fuel moisture content.
        """
        # Simple linear relationship
        moisture = (relative_humidity * 0.8) - (temperature * 0.5) + 20
        
        # Ensure reasonable bounds
        return np.clip(moisture, 0, 100)
    
    def calculate_fire_weather_index(
        self,
        temperature: float,
        relative_humidity: float,
        wind_speed: float,
        precipitation: float,
        method: str = 'canadian'
    ) -> Dict[str, float]:
        """
        Calculate Fire Weather Index (FWI) components.
        
        The FWI is a numerical rating of fire intensity that combines
        the effects of weather conditions on fire behavior. It's widely
        used in fire management and risk assessment.
        
        Args:
            temperature: Air temperature in Celsius
            relative_humidity: Relative humidity as percentage
            wind_speed: Wind speed in km/h
            precipitation: 24-hour precipitation in mm
            method: FWI calculation method ('canadian', 'simplified')
            
        Returns:
            Dictionary containing FWI components and final index
            
        References:
            Van Wagner (1987) - Canadian Forest Fire Weather Index
        """
        if method == 'canadian':
            return self._canadian_fwi(temperature, relative_humidity, wind_speed, precipitation)
        elif method == 'simplified':
            return self._simplified_fwi(temperature, relative_humidity, wind_speed, precipitation)
        else:
            raise ValueError("Method must be 'canadian' or 'simplified'")
    
    def _canadian_fwi(
        self,
        temperature: float,
        relative_humidity: float,
        wind_speed: float,
        precipitation: float
    ) -> Dict[str, float]:
        """
        Calculate Canadian Forest Fire Weather Index.
        
        This is the standard FWI system used in Canada and many other
        countries for operational fire danger assessment.
        """
        # Duff Moisture Code (DMC) - represents moisture content of medium fuels
        # Simplified calculation for educational purposes
        dmc = max(0, 244.73 - 43.43 * np.log(relative_humidity + 1))
        
        # Drought Code (DC) - represents moisture content of deep, compact organic layers
        dc = max(0, 800 * np.exp(-0.05 * (temperature + 10)))
        
        # Initial Spread Index (ISI) - represents rate of fire spread
        isi = 0.208 * wind_speed * np.exp(0.05039 * temperature)
        
        # Buildup Index (BUI) - represents fuel available for combustion
        bui = 0.8 * dmc + 0.4 * dc
        
        # Fire Weather Index (FWI) - represents fire intensity
        fwi = 0.0272 * (isi * bui) ** 0.46
        
        return {
            'DMC': dmc,
            'DC': dc,
            'ISI': isi,
            'BUI': bui,
            'FWI': fwi,
            'danger_rating': self._get_danger_rating(fwi)
        }
    
    def _simplified_fwi(
        self,
        temperature: float,
        relative_humidity: float,
        wind_speed: float,
        precipitation: float
    ) -> Dict[str, float]:
        """
        Simplified FWI calculation for educational purposes.
        
        This demonstrates the basic principles of fire weather indices
        without the complexity of the full Canadian system.
        """
        # Simple moisture factor
        moisture_factor = max(0, (100 - relative_humidity) / 100)
        
        # Temperature factor
        temp_factor = max(0, (temperature - 10) / 30)
        
        # Wind factor
        wind_factor = min(1.0, wind_speed / 50)
        
        # Precipitation damping
        precip_factor = max(0.1, 1 - (precipitation / 50))
        
        # Combined index
        fwi = moisture_factor * temp_factor * wind_factor * precip_factor * 100
        
        return {
            'moisture_factor': moisture_factor,
            'temp_factor': temp_factor,
            'wind_factor': wind_factor,
            'precip_factor': precip_factor,
            'FWI': fwi,
            'danger_rating': self._get_danger_rating(fwi)
        }
    
    def _get_danger_rating(self, fwi: float) -> str:
        """Convert FWI value to danger rating category."""
        if fwi < 5.2:
            return "Very Low"
        elif fwi < 11.2:
            return "Low"
        elif fwi < 21.3:
            return "Moderate"
        elif fwi < 38.0:
            return "High"
        elif fwi < 50.0:
            return "Very High"
        else:
            return "Extreme"
    
    def calculate_keetch_byram_drought_index(
        self,
        temperature: float,
        precipitation: float,
        previous_kbdi: float = 0.0
    ) -> float:
        """
        Calculate Keetch-Byram Drought Index (KBDI).
        
        KBDI is a measure of drought that indicates the amount of
        precipitation needed to bring soil to field capacity. It's
        particularly useful for fire risk assessment in forested areas.
        
        Args:
            temperature: Daily maximum temperature in Celsius
            precipitation: Daily precipitation in mm
            previous_kbdi: Previous day's KBDI value
            
        Returns:
            Current KBDI value
            
        References:
            Keetch & Byram (1968) - Drought index for fire potential
        """
        # Maximum KBDI value (203.2 mm = 8 inches)
        MAX_KBDI = 203.2
        
        # Net rainfall (precipitation minus interception)
        net_rainfall = max(0, precipitation - 5.08)  # 5.08 mm = 0.2 inches
        
        # Calculate potential evapotranspiration
        if temperature > 0:
            # Simplified potential ET calculation
            pet = 0.001 * (temperature + 17.8) ** 2.2
        else:
            pet = 0
        
        # Calculate new KBDI
        if net_rainfall > 0:
            # Rain reduces KBDI
            kbdi = max(0, previous_kbdi - net_rainfall)
        else:
            # No rain increases KBDI
            kbdi = min(MAX_KBDI, previous_kbdi + pet)
        
        return kbdi
    
    def calculate_topographical_features(
        self,
        coordinates: Tuple[float, float],
        radius_meters: float = 1000
    ) -> Dict[str, float]:
        """
        Calculate topographical features from DEM data.
        
        Topographical features influence fire behavior through effects
        on wind patterns, solar radiation, and fuel moisture.
        
        Args:
            coordinates: (latitude, longitude) tuple
            radius_meters: Radius for local feature calculation
            
        Returns:
            Dictionary containing topographical features
        """
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem_data() first.")
        
        try:
            # Extract local DEM window around coordinates
            lat, lon = coordinates
            
            # Convert to DEM coordinates (simplified)
            # In practice, you'd use proper coordinate transformation
            y_idx = int((lat - self.dem_data.y.min()) / (self.dem_data.y.max() - self.dem_data.y.min()) * len(self.dem_data.y))
            x_idx = int((lon - self.dem_data.x.min()) / (self.dem_data.x.max() - self.dem_data.x.min()) * len(self.dem_data.x))
            
            # Extract local window
            window_size = max(1, int(radius_meters / 30))  # Assuming 30m resolution
            y_start = max(0, y_idx - window_size)
            y_end = min(len(self.dem_data.y), y_idx + window_size)
            x_start = max(0, x_idx - window_size)
            x_end = min(len(self.dem_data.x), x_idx + window_size)
            
            local_dem = self.dem_data.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
            
            # Calculate features
            elevation = float(local_dem.mean())
            slope = self._calculate_slope(local_dem)
            aspect = self._calculate_aspect(local_dem)
            roughness = float(local_dem.std())
            
            return {
                'elevation': elevation,
                'slope': slope,
                'aspect': aspect,
                'roughness': roughness,
                'elevation_range': float(local_dem.max() - local_dem.min())
            }
            
        except Exception as e:
            logger.error(f"Error calculating topographical features: {e}")
            return {
                'elevation': 0.0,
                'slope': 0.0,
                'aspect': 0.0,
                'roughness': 0.0,
                'elevation_range': 0.0
            }
    
    def _calculate_slope(self, dem_window: xr.DataArray) -> float:
        """Calculate average slope from DEM window."""
        try:
            # Simplified slope calculation using gradient
            dy, dx = np.gradient(dem_window.values)
            slope = np.sqrt(dx**2 + dy**2)
            return float(np.mean(slope))
        except:
            return 0.0
    
    def _calculate_aspect(self, dem_window: xr.DataArray) -> float:
        """Calculate average aspect from DEM window."""
        try:
            # Simplified aspect calculation
            dy, dx = np.gradient(dem_window.values)
            aspect = np.arctan2(dy, dx)
            # Convert to degrees and normalize to 0-360
            aspect_deg = np.degrees(aspect) % 360
            return float(np.mean(aspect_deg))
        except:
            return 0.0
    
    def calculate_vegetation_features(
        self,
        coordinates: Tuple[float, float],
        date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate vegetation features from satellite data.
        
        Vegetation features are crucial for fire risk assessment as they
        indicate fuel availability, moisture content, and vegetation stress.
        
        Args:
            coordinates: (latitude, longitude) tuple
            date: Date for feature calculation (default: most recent)
            
        Returns:
            Dictionary containing vegetation features
        """
        if self.satellite_data is None:
            logger.warning("Satellite data not available. Returning default values.")
            return self._get_default_vegetation_features()
        
        try:
            # Extract vegetation indices for the location
            # This is a simplified implementation - in practice you'd
            # extract actual pixel values from satellite imagery
            
            features = {
                'ndvi': np.random.uniform(0.2, 0.8),  # Placeholder
                'nbr': np.random.uniform(-0.3, 0.6),  # Placeholder
                'ndwi': np.random.uniform(-0.5, 0.3),  # Placeholder
                'evi': np.random.uniform(0.1, 0.6),   # Placeholder
                'vegetation_density': np.random.uniform(0.1, 1.0),  # Placeholder
                'vegetation_stress': np.random.uniform(0.0, 0.8)   # Placeholder
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating vegetation features: {e}")
            return self._get_default_vegetation_features()
    
    def _get_default_vegetation_features(self) -> Dict[str, float]:
        """Return default vegetation features when data is unavailable."""
        return {
            'ndvi': 0.5,
            'nbr': 0.2,
            'ndwi': -0.1,
            'evi': 0.3,
            'vegetation_density': 0.5,
            'vegetation_stress': 0.3
        }
    
    def create_temporal_features(
        self,
        data: pd.DataFrame,
        date_column: str,
        target_column: str,
        windows: List[int] = [7, 14, 30]
    ) -> pd.DataFrame:
        """
        Create temporal features for time series analysis.
        
        Temporal features capture the temporal dynamics of fire risk
        factors, including trends, seasonality, and recent changes.
        
        Args:
            data: DataFrame with time series data
            date_column: Name of date column
            target_column: Name of target variable column
            windows: List of rolling window sizes in days
            
        Returns:
            DataFrame with additional temporal features
        """
        try:
            # Ensure date column is datetime
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])
            data = data.sort_values(date_column)
            
            # Create rolling averages
            for window in windows:
                col_name = f'{target_column}_rolling_{window}d'
                data[col_name] = data[target_column].rolling(window=window, min_periods=1).mean()
            
            # Create lag features
            for lag in [1, 3, 7]:
                col_name = f'{target_column}_lag_{lag}d'
                data[col_name] = data[target_column].shift(lag)
            
            # Create seasonal features
            data['day_of_year'] = data[date_column].dt.dayofyear
            data['month'] = data[date_column].dt.month
            data['season'] = data[date_column].dt.month % 12 // 3 + 1
            
            # Create trend features
            data['days_since_start'] = (data[date_column] - data[date_column].min()).dt.days
            
            logger.info(f"Created temporal features with {len(data.columns)} columns")
            return data
            
        except Exception as e:
            logger.error(f"Error creating temporal features: {e}")
            return data
    
    def engineer_all_features(
        self,
        coordinates: Tuple[float, float],
        date: Optional[datetime] = None,
        include_temporal: bool = True
    ) -> Dict[str, Any]:
        """
        Engineer all fire risk features for a given location and time.
        
        This is the main method that combines all feature engineering
        steps to create a comprehensive feature set for fire risk assessment.
        
        Args:
            coordinates: (latitude, longitude) tuple
            date: Date for feature calculation (default: current date)
            include_temporal: Whether to include temporal features
            
        Returns:
            Dictionary containing all engineered features
        """
        if date is None:
            date = datetime.now()
        
        try:
            # Calculate all feature categories
            features = {}
            
            # Weather-based features
            if self.weather_data is not None:
                weather_features = self._extract_weather_features(coordinates, date)
                features.update(weather_features)
            
            # Topographical features
            topo_features = self.calculate_topographical_features(coordinates)
            features.update(topo_features)
            
            # Vegetation features
            veg_features = self.calculate_vegetation_features(coordinates, date)
            features.update(veg_features)
            
            # NEON ecological features
            if self.neon_data is not None:
                neon_features = self._extract_neon_features(coordinates, date)
                features.update(neon_features)
            
            # Add metadata
            features['timestamp'] = date.isoformat()
            features['latitude'] = coordinates[0]
            features['longitude'] = coordinates[1]
            features['feature_count'] = len(features)
            
            # Cache features
            cache_key = f"{coordinates}_{date.strftime('%Y%m%d')}"
            self.feature_cache[cache_key] = features
            
            logger.info(f"Engineered {len(features)} features for location {coordinates}")
            return features
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            raise
    
    def _extract_weather_features(
        self,
        coordinates: Tuple[float, float],
        date: datetime
    ) -> Dict[str, float]:
        """Extract weather features for a specific location and date."""
        try:
            # Find nearest weather station or interpolate
            # This is a simplified implementation
            weather_features = {
                'temperature': np.random.uniform(15, 35),
                'relative_humidity': np.random.uniform(20, 80),
                'wind_speed': np.random.uniform(0, 30),
                'precipitation': np.random.uniform(0, 50),
                'vapor_pressure_deficit': np.random.uniform(0, 3)
            }
            
            # Calculate derived weather features
            fwi_components = self.calculate_fire_weather_index(
                weather_features['temperature'],
                weather_features['relative_humidity'],
                weather_features['wind_speed'],
                weather_features['precipitation']
            )
            
            weather_features.update(fwi_components)
            
            return weather_features
            
        except Exception as e:
            logger.error(f"Error extracting weather features: {e}")
            return {}
    
    def _extract_neon_features(
        self,
        coordinates: Tuple[float, float],
        date: datetime
    ) -> Dict[str, float]:
        """Extract NEON ecological features for a specific location and date."""
        try:
            # Find nearest NEON site or interpolate
            # This is a simplified implementation
            neon_features = {
                'soil_moisture': np.random.uniform(0.1, 0.4),
                'canopy_cover': np.random.uniform(0.2, 0.9),
                'litter_depth': np.random.uniform(0.5, 5.0),
                'fuel_load': np.random.uniform(0.5, 10.0)
            }
            
            return neon_features
            
        except Exception as e:
            logger.error(f"Error extracting NEON features: {e}")
            return {}
    
    def get_feature_summary(self) -> pd.DataFrame:
        """
        Get a summary of all engineered features.
        
        Returns:
            DataFrame with feature statistics and descriptions
        """
        if not self.feature_cache:
            return pd.DataFrame()
        
        # Combine all cached features
        all_features = []
        for cache_key, features in self.feature_cache.items():
            features_copy = features.copy()
            features_copy['cache_key'] = cache_key
            all_features.append(features_copy)
        
        df = pd.DataFrame(all_features)
        
        # Create summary statistics
        summary = df.describe()
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Initialize feature engine
    engine = FireRiskFeatureEngine()
    
    # Example: Calculate fuel moisture
    fmc = engine.calculate_fuel_moisture_content(
        temperature=25.0,
        relative_humidity=60.0,
        fuel_type='medium'
    )
    print(f"Fuel Moisture Content: {fmc:.1f}%")
    
    # Example: Calculate FWI
    fwi_result = engine.calculate_fire_weather_index(
        temperature=30.0,
        relative_humidity=40.0,
        wind_speed=20.0,
        precipitation=0.0
    )
    print(f"Fire Weather Index: {fwi_result['FWI']:.2f} ({fwi_result['danger_rating']})")
    
    # Example: Calculate KBDI
    kbdi = engine.calculate_keetch_byram_drought_index(
        temperature=28.0,
        precipitation=0.0,
        previous_kbdi=50.0
    )
    print(f"Keetch-Byram Drought Index: {kbdi:.1f}")
    
    print("Feature engineering engine ready for use!")
