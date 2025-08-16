# Satellite data collection using Google Earth Engine
# Retrieves data from Sentinel, MODIS, Landsat and other sources

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import ee

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Earth Engine warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ee")


class SatelliteDataClient:
    # Google Earth Engine client for satellite imagery retrieval
    
    def __init__(self, cache_dir: str = "./cache/satellite", max_retries: int = 3):
        # Initialize Google Earth Engine connection
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.initialized = False
        
        print("Initializing Satellite Data Client...")
        
        # Initialize Earth Engine
        try:
            print("   Checking for service account credentials...")
            # Import from config to get the loaded environment variables
            import os
            from config import GOOGLE_EARTH_ENGINE_CREDENTIALS
            credentials_path = GOOGLE_EARTH_ENGINE_CREDENTIALS
            
            if credentials_path:
                print(f"   Found credentials path: {credentials_path}")
                
                if os.path.exists(credentials_path):
                    print("   Credentials file exists, loading service account...")
                    # Use service account authentication
                    import ee
                    from google.oauth2 import service_account
                    
                    # Load service account credentials
                    print("   Loading service account credentials...")
                    credentials = service_account.Credentials.from_service_account_file(
                        credentials_path,
                        scopes=['https://www.googleapis.com/auth/earthengine']
                    )
                    print("   Service account credentials loaded successfully")
                    
                    # Initialize Earth Engine with service account
                    print("   Initializing Earth Engine with service account...")
                    ee.Initialize(credentials=credentials)
                    self.initialized = True
                    print("   Google Earth Engine initialized successfully with service account!")
                    logger.info("Google Earth Engine initialized successfully with service account")
                    
                else:
                    print(f"   Credentials file not found: {credentials_path}")
                    print("   Falling back to personal authentication...")
                    # Fallback to personal authentication
                    import ee
                    ee.Initialize()
                    self.initialized = True
                    print("   Google Earth Engine initialized successfully with personal authentication!")
                    logger.info("Google Earth Engine initialized successfully with personal authentication")
                    
            else:
                print("   No service account credentials found")
                print("   Using personal authentication...")
                # Fallback to personal authentication
                import ee
                ee.Initialize()
                self.initialized = True
                print("   Google Earth Engine initialized successfully with personal authentication!")
                logger.info("Google Earth Engine initialized successfully with personal authentication")
                
        except Exception as e:
            print(f"   Error during initialization: {e}")
            if "not registered to use Earth Engine" in str(e):
                print("   Project not registered for Earth Engine. Using demo data.")
                print("   To enable real satellite data:")
                print("      1. Visit: https://code.earthengine.google.com/register?project=wildfire-risk")
                print("      2. Or enable Earth Engine API in Google Cloud Console")
                logger.warning("Project not registered for Earth Engine. Using demo data.")
                self.initialized = False
            else:
                print("   Failed to initialize Google Earth Engine")
                print("   Please check your service account credentials or authenticate with 'earthengine authenticate'")
                logger.error(f"Failed to initialize Google Earth Engine: {e}")
                self.initialized = False
        
        if self.initialized:
            print("   Satellite client ready for real data!")
        else:
            print("   Satellite client using demo data mode")
    
    def _validate_initialization(self) -> None:
        """Ensure Earth Engine is initialized before use."""
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized. "
                             "Please authenticate and initialize first.")
    
    def get_sentinel2_data(
        self,
        geometry: Union[Point, Polygon, gpd.GeoDataFrame],
        start_date: str,
        end_date: str,
        cloud_filter: float = 20.0,
        include_metadata: bool = True
    ) -> Dict:
        """
        Collect Sentinel-2 Level-2A data for specified area and time period.
        
        Sentinel-2 provides high-resolution (10-20m) multispectral imagery
        with bands useful for vegetation analysis and fire risk assessment.
        
        Args:
            geometry: Geographic area of interest
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            cloud_filter: Maximum cloud coverage percentage (0-100)
            include_metadata: Whether to include image metadata
            
        Returns:
            Dictionary containing image collection and metadata
            
        Raises:
            RuntimeError: If Earth Engine is not initialized
            ValueError: If dates are invalid or geometry is unsupported
        """
        self._validate_initialization()
        
        # Validate inputs
        if not self._is_valid_date_range(start_date, end_date):
            raise ValueError("Invalid date range. End date must be after start date.")
        
        # Convert geometry to Earth Engine format
        ee_geometry = self._convert_geometry(geometry)
        
        try:
            # Get Sentinel-2 Level-2A collection
            s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                           .filterBounds(ee_geometry)
                           .filterDate(start_date, end_date)
                           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))
                           .filter(ee.Filter.lt('CLOUD_COVERAGE_ASSESSMENT', cloud_filter))
                           .sort('system:time_start'))
            
            # Apply cloud masking
            s2_cloud_masked = s2_collection.map(self._mask_sentinel2_clouds)
            
            # Calculate vegetation indices
            s2_with_indices = s2_cloud_masked.map(self._calculate_sentinel2_indices)
            
            result = {
                'collection': s2_with_indices,
                'count': s2_with_indices.size().getInfo(),
                'date_range': [start_date, end_date],
                'cloud_filter': cloud_filter
            }
            
            if include_metadata:
                result['metadata'] = self._get_collection_metadata(s2_with_indices)
            
            logger.info(f"Collected {result['count']} Sentinel-2 images")
            return result
            
        except ee_exception.EEException as e:
            logger.error(f"Earth Engine error collecting Sentinel-2 data: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error collecting Sentinel-2 data: {e}")
            raise
    
    def get_modis_data(
        self,
        geometry: Union[Point, Polygon, gpd.GeoDataFrame],
        start_date: str,
        end_date: str,
        product: str = 'MOD13Q1.061',
        include_metadata: bool = True
    ) -> Dict:
        """
        Collect MODIS data for specified area and time period.
        
        MODIS provides global coverage with moderate resolution (250m-1km)
        and daily to 16-day temporal resolution, useful for large-scale
        vegetation monitoring and fire risk assessment.
        
        Args:
            geometry: Geographic area of interest
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            product: MODIS product ID (default: MOD13Q1.061 for NDVI)
            include_metadata: Whether to include image metadata
            
        Returns:
            Dictionary containing image collection and metadata
        """
        self._validate_initialization()
        
        # Validate inputs
        if not self._is_valid_date_range(start_date, end_date):
            raise ValueError("Invalid date range. End date must be after start date.")
        
        # Convert geometry to Earth Engine format
        ee_geometry = self._convert_geometry(geometry)
        
        try:
            # Get MODIS collection
            modis_collection = (ee.ImageCollection(product)
                              .filterBounds(ee_geometry)
                              .filterDate(start_date, end_date)
                              .sort('system:time_start'))
            
            # Apply quality filtering for MOD13Q1
            if 'MOD13Q1' in product:
                modis_filtered = modis_collection.map(self._filter_modis_quality)
            else:
                modis_filtered = modis_collection
            
            result = {
                'collection': modis_filtered,
                'count': modis_filtered.size().getInfo(),
                'date_range': [start_date, end_date],
                'product': product
            }
            
            if include_metadata:
                result['metadata'] = self._get_collection_metadata(modis_filtered)
            
            logger.info(f"Collected {result['count']} MODIS images")
            return result
            
        except ee_exception.EEException as e:
            logger.error(f"Earth Engine error collecting MODIS data: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error collecting MODIS data: {e}")
            raise
    
    def get_landsat_data(
        self,
        geometry: Union[Point, Polygon, gpd.GeoDataFrame],
        start_date: str,
        end_date: str,
        collection: str = 'LANDSAT/LC08/C02/T1_L2',
        cloud_filter: float = 20.0,
        include_metadata: bool = True
    ) -> Dict:
        """
        Collect Landsat data for specified area and time period.
        
        Landsat provides long-term historical data (since 1972) with
        30m resolution, useful for temporal analysis and change detection
        in fire risk assessment.
        
        Args:
            geometry: Geographic area of interest
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            collection: Landsat collection ID
            cloud_filter: Maximum cloud coverage percentage (0-100)
            include_metadata: Whether to include image metadata
            
        Returns:
            Dictionary containing image collection and metadata
        """
        self._validate_initialization()
        
        # Validate inputs
        if not self._is_valid_date_range(start_date, end_date):
            raise ValueError("Invalid date range. End date must be after start date.")
        
        # Convert geometry to Earth Engine format
        ee_geometry = self._convert_geometry(geometry)
        
        try:
            # Get Landsat collection
            landsat_collection = (ee.ImageCollection(collection)
                                .filterBounds(ee_geometry)
                                .filterDate(start_date, end_date)
                                .filter(ee.Filter.lt('CLOUD_COVER', cloud_filter))
                                .sort('system:time_start'))
            
            # Apply cloud masking
            landsat_cloud_masked = landsat_collection.map(self._mask_landsat_clouds)
            
            # Calculate vegetation indices
            landsat_with_indices = landsat_cloud_masked.map(self._calculate_landsat_indices)
            
            result = {
                'collection': landsat_with_indices,
                'count': landsat_with_indices.size().getInfo(),
                'date_range': [start_date, end_date],
                'collection_id': collection,
                'cloud_filter': cloud_filter
            }
            
            if include_metadata:
                result['metadata'] = self._get_collection_metadata(landsat_with_indices)
            
            logger.info(f"Collected {result['count']} Landsat images")
            return result
            
        except ee_exception.EEException as e:
            logger.error(f"Earth Engine error collecting Landsat data: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error collecting Landsat data: {e}")
            raise
    
    def create_temporal_composite(
        self,
        collection: ee.ImageCollection,
        method: str = 'median',
        period: str = 'month'
    ) -> ee.ImageCollection:
        """
        Create temporal composites from image collections.
        
        Temporal compositing reduces noise and creates representative
        images for analysis periods, improving the quality of derived
        vegetation indices and fire risk assessments.
        
        Args:
            collection: Earth Engine image collection
            method: Compositing method ('median', 'mean', 'max', 'min')
            period: Time period for compositing ('day', 'week', 'month', 'year')
            
        Returns:
            Composited image collection
        """
        self._validate_initialization()
        
        if method not in ['median', 'mean', 'max', 'min']:
            raise ValueError("Method must be one of: median, mean, max, min")
        
        if period not in ['day', 'week', 'month', 'year']:
            raise ValueError("Period must be one of: day, week, month, year")
        
        try:
            if method == 'median':
                composite = collection.median()
            elif method == 'mean':
                composite = collection.mean()
            elif method == 'max':
                composite = collection.max()
            else:  # min
                composite = collection.min()
            
            logger.info(f"Created {method} composite for {period} period")
            return composite
            
        except Exception as e:
            logger.error(f"Error creating temporal composite: {e}")
            raise
    
    def _mask_sentinel2_clouds(self, image: ee.Image) -> ee.Image:
        """
        Apply cloud masking to Sentinel-2 images using QA60 band.
        
        The QA60 band provides cloud and cirrus cloud information
        that can be used to create cloud masks for analysis.
        
        Args:
            image: Sentinel-2 image
            
        Returns:
            Cloud-masked image
        """
        # Get QA60 band
        qa = image.select('QA60')
        
        # Create cloud mask (bits 6 and 7 for cloud and cirrus)
        cloud_mask = qa.bitwiseAnd(3).eq(0)  # Clear pixels
        
        # Apply mask to all bands
        return image.updateMask(cloud_mask)
    
    def _mask_landsat_clouds(self, image: ee.Image) -> ee.Image:
        """
        Apply cloud masking to Landsat images using QA_PIXEL band.
        
        The QA_PIXEL band provides detailed quality information
        including cloud, cloud shadow, and snow/ice detection.
        
        Args:
            image: Landsat image
            
        Returns:
            Cloud-masked image
        """
        # Get QA_PIXEL band
        qa = image.select('QA_PIXEL')
        
        # Create cloud mask (bits 3 and 4 for cloud and cloud shadow)
        cloud_mask = qa.bitwiseAnd(24).eq(0)  # Clear pixels
        
        # Apply mask to all bands
        return image.updateMask(cloud_mask)
    
    def _filter_modis_quality(self, image: ee.Image) -> ee.Image:
        """
        Apply quality filtering to MODIS NDVI data using reliability band.
        
        The reliability band indicates data quality and should be
        used to filter out poor quality observations.
        
        Args:
            image: MODIS image
            
        Returns:
            Quality-filtered image
        """
        # Get reliability band
        reliability = image.select('SummaryQA')
        
        # Keep only good quality pixels (value 0)
        quality_mask = reliability.eq(0)
        
        # Apply mask to all bands
        return image.updateMask(quality_mask)
    
    def _calculate_sentinel2_indices(self, image: ee.Image) -> ee.Image:
        """
        Calculate vegetation indices for Sentinel-2 images.
        
        These indices are crucial for fire risk assessment:
        - NDVI: Vegetation health and density
        - NBR: Burn severity and vegetation stress
        - NDWI: Vegetation water content
        - EVI: Enhanced vegetation index for dense vegetation
        
        Args:
            image: Sentinel-2 image with bands B2, B3, B4, B8, B11, B12
            
        Returns:
            Image with additional vegetation index bands
        """
        # Normalized Difference Vegetation Index (NDVI)
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # Normalized Burn Ratio (NBR)
        nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
        
        # Normalized Difference Water Index (NDWI)
        ndwi = image.normalizedDifference(['B3', 'B11']).rename('NDWI')
        
        # Enhanced Vegetation Index (EVI)
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }
        ).rename('EVI')
        
        # Add indices to image
        return image.addBands([ndvi, nbr, ndwi, evi])
    
    def _calculate_landsat_indices(self, image: ee.Image) -> ee.Image:
        """
        Calculate vegetation indices for Landsat images.
        
        Landsat provides similar indices to Sentinel-2 but with
        different band configurations and historical coverage.
        
        Args:
            image: Landsat image with bands SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7
            
        Returns:
            Image with additional vegetation index bands
        """
        # Normalized Difference Vegetation Index (NDVI)
        ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        
        # Normalized Burn Ratio (NBR)
        nbr = image.normalizedDifference(['SR_B5', 'SR_B7']).rename('NBR')
        
        # Normalized Difference Water Index (NDWI)
        ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
        
        # Soil Adjusted Vegetation Index (SAVI)
        savi = image.expression(
            '1.5 * ((NIR - RED) / (NIR + RED + 0.5))',
            {
                'NIR': image.select('SR_B5'),
                'RED': image.select('SR_B4')
            }
        ).rename('SAVI')
        
        # Add indices to image
        return image.addBands([ndvi, nbr, ndwi, savi])
    
    def _convert_geometry(self, geometry: Union[Point, Polygon, gpd.GeoDataFrame]) -> ee.Geometry:
        """
        Convert various geometry types to Earth Engine format.
        
        Args:
            geometry: Input geometry (Point, Polygon, or GeoDataFrame)
            
        Returns:
            Earth Engine geometry object
        """
        if isinstance(geometry, gpd.GeoDataFrame):
            # Convert GeoDataFrame to Earth Engine geometry
            geojson = geometry.__geo_interface__
            return ee.Geometry(geojson)
        elif isinstance(geometry, (Point, Polygon)):
            # Convert Shapely geometry to Earth Engine geometry
            geojson = geometry.__geo_interface__
            return ee.Geometry(geojson)
        else:
            raise ValueError("Geometry must be Point, Polygon, or GeoDataFrame")
    
    def _is_valid_date_range(self, start_date: str, end_date: str) -> bool:
        """
        Validate that the date range is logical.
        
        Args:
            start_date: Start date string
            end_date: End date string
            
        Returns:
            True if valid, False otherwise
        """
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            return start < end
        except ValueError:
            return False
    
    def _get_collection_metadata(self, collection: ee.ImageCollection) -> Dict:
        """
        Extract metadata from an image collection.
        
        Args:
            collection: Earth Engine image collection
            
        Returns:
            Dictionary containing metadata
        """
        try:
            # Get basic collection info
            info = collection.getInfo()
            
            metadata = {
                'size': info.get('features', []),
                'bands': info.get('bands', []),
                'properties': info.get('properties', {})
            }
            
            return metadata
        except Exception as e:
            logger.warning(f"Could not extract metadata: {e}")
            return {}
    
    def download_to_local(
        self,
        image: ee.Image,
        geometry: Union[Point, Polygon, gpd.GeoDataFrame],
        scale: int = 30,
        format: str = 'GeoTIFF'
    ) -> str:
        """
        Download an Earth Engine image to local storage.
        
        Args:
            image: Earth Engine image to download
            geometry: Area of interest for download
            scale: Resolution in meters
            format: Output format ('GeoTIFF', 'PNG', 'JPEG')
            
        Returns:
            Path to downloaded file
        """
        # This is a placeholder - actual download implementation
        # would require additional setup and authentication
        logger.info("Download functionality requires additional setup")
        return ""


# Example usage and testing
if __name__ == "__main__":
    # Initialize client
    client = SatelliteDataClient()
    
    # Example: Get Sentinel-2 data for a point
    point = Point(-122.4194, 37.7749)  # San Francisco
    
    try:
        s2_data = client.get_sentinel2_data(
            geometry=point,
            start_date='2023-06-01',
            end_date='2023-06-30',
            cloud_filter=20.0
        )
        print(f"Collected {s2_data['count']} Sentinel-2 images")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure Earth Engine is authenticated and initialized")
