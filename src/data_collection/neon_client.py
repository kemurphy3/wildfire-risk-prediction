"""
NEON Data Collection Module

This module demonstrates best practices for accessing NEON ecological data
for environmental modeling applications.
"""

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class NEONDataCollector:
    """
    A class to collect ecological data from NEON (National Ecological Observatory Network).
    
    This demonstrates proper API usage, data caching, and error handling for
    scientific data collection.
    """
    
    BASE_URL = "https://data.neonscience.org/api/v0/"
    
    # Key data products for wildfire risk assessment
    FIRE_RELEVANT_PRODUCTS = {
        "DP1.00041.001": "Soil temperature",
        "DP1.00094.001": "Soil water content and water salinity",
        "DP1.00098.001": "Relative humidity",
        "DP1.10023.001": "Herbaceous clip harvest",
        "DP1.10033.001": "Phenology observations",
        "DP1.00040.001": "Soil heat flux plate",
        "DP1.00024.001": "Photosynthetically active radiation (PAR)",
        "DP1.00066.001": "Photosynthetically active radiation (quantum line)",
    }
    
    # NEON Airborne Observation Platform (AOP) products for satellite crosswalk
    AOP_PRODUCTS = {
        "DP3.30006.001": "Spectrometer orthorectified surface directional reflectance - mosaic",
        "DP3.30010.001": "High-resolution orthorectified camera imagery - mosaic",
        "DP3.30015.001": "Ecosystem structure",
        "DP3.30024.001": "Elevation - LiDAR",
    }
    
    # Fire-prone NEON sites (example selection)
    FIRE_PRONE_SITES = [
        "SJER",  # San Joaquin Experimental Range, CA
        "SOAP",  # Soaproot Saddle, CA
        "TEAK",  # Lower Teakettle, CA
        "WREF",  # Wind River Experimental Forest, WA
        "ABBY",  # Abby Road, WA
        "RMNP",  # Rocky Mountain National Park, CO
    ]
    
    def __init__(self, api_token: Optional[str] = None, cache_dir: str = "data/cache/neon"):
        """
        Initialize the NEON data collector.
        
        Args:
            api_token: Optional API token for authenticated access
            cache_dir: Directory for caching downloaded data
        """
        self.api_token = api_token or os.getenv("NEON_API_TOKEN")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        if self.api_token:
            self.session.headers.update({"X-API-Token": self.api_token})
    
    def get_sites_info(self, site_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retrieve information about NEON sites.
        
        Args:
            site_codes: List of specific site codes to retrieve. If None, returns all sites.
            
        Returns:
            DataFrame with site information
        """
        logger.info("Fetching NEON sites information")
        
        response = self._make_request("sites")
        sites_data = response.json()["data"]
        
        sites_df = pd.DataFrame(sites_data)
        
        if site_codes:
            sites_df = sites_df[sites_df["siteCode"].isin(site_codes)]
        
        return sites_df
    
    def get_product_data(
        self,
        product_code: str,
        site_code: str,
        start_date: str,
        end_date: str,
        package: str = "basic"
    ) -> Dict:
        """
        Retrieve data for a specific NEON data product.
        
        Args:
            product_code: NEON data product code
            site_code: NEON site code
            start_date: Start date (YYYY-MM format)
            end_date: End date (YYYY-MM format)
            package: Data package type ('basic' or 'expanded')
            
        Returns:
            Dictionary containing the data and metadata
        """
        logger.info(f"Fetching {product_code} for site {site_code} from {start_date} to {end_date}")
        
        # Check cache first
        cache_key = f"{product_code}_{site_code}_{start_date}_{end_date}_{package}"
        cached_data = self._check_cache(cache_key)
        if cached_data is not None:
            logger.info("Using cached data")
            return cached_data
        
        # Build request URL
        endpoint = f"data/{product_code}/{site_code}/{start_date}"
        params = {
            "package": package,
            "endDate": end_date
        }
        
        response = self._make_request(endpoint, params=params)
        data = response.json()
        
        # Cache the response
        self._save_cache(cache_key, data)
        
        return data
    
    def download_fire_risk_data(
        self,
        sites: Optional[List[str]] = None,
        start_date: str = None,
        end_date: str = None,
        products: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Download all fire-relevant data for specified sites and time period.
        
        Args:
            sites: List of site codes (defaults to FIRE_PRONE_SITES)
            start_date: Start date in YYYY-MM format
            end_date: End date in YYYY-MM format
            products: List of product codes (defaults to FIRE_RELEVANT_PRODUCTS)
            
        Returns:
            Dictionary mapping product codes to DataFrames
        """
        sites = sites or self.FIRE_PRONE_SITES
        products = products or list(self.FIRE_RELEVANT_PRODUCTS.keys())
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m")
        
        logger.info(f"Downloading fire risk data for {len(sites)} sites and {len(products)} products")
        
        all_data = {}
        
        for product in products:
            product_data = []
            
            for site in sites:
                try:
                    data = self.get_product_data(product, site, start_date, end_date)
                    
                    # Extract and process the actual data files
                    if "data" in data and "files" in data["data"]:
                        for file_info in data["data"]["files"]:
                            if file_info["name"].endswith(".csv"):
                                df = self._download_and_parse_csv(file_info["url"])
                                if df is not None:
                                    df["siteCode"] = site
                                    df["productCode"] = product
                                    product_data.append(df)
                
                except Exception as e:
                    logger.error(f"Error downloading {product} for {site}: {str(e)}")
                    continue
            
            if product_data:
                all_data[product] = pd.concat(product_data, ignore_index=True)
                logger.info(f"Downloaded {len(product_data)} files for {product}")
        
        return all_data
    
    def get_aop_data(
        self,
        product_code: str,
        site_code: str,
        year: int,
        easting: Optional[Tuple[float, float]] = None,
        northing: Optional[Tuple[float, float]] = None
    ) -> Dict:
        """
        Retrieve NEON Airborne Observation Platform (AOP) data.
        
        Args:
            product_code: NEON AOP data product code
            site_code: NEON site code
            year: Year of the flight campaign
            easting: Optional tuple of (min, max) easting coordinates
            northing: Optional tuple of (min, max) northing coordinates
            
        Returns:
            Dictionary containing the AOP data and metadata
        """
        logger.info(f"Fetching AOP data {product_code} for site {site_code}, year {year}")
        
        # AOP data uses year instead of date range
        endpoint = f"data/{product_code}/{site_code}/{year}"
        
        params = {}
        if easting:
            params["easting"] = f"{easting[0]},{easting[1]}"
        if northing:
            params["northing"] = f"{northing[0]},{northing[1]}"
        
        response = self._make_request(endpoint, params=params)
        data = response.json()
        
        return data
    
    def download_aop_reflectance_data(
        self,
        sites: List[str],
        years: List[int],
        bbox: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Download NEON AOP reflectance data for satellite crosswalk.
        
        Args:
            sites: List of NEON site codes
            years: List of years to download
            bbox: Optional bounding box dict with 'easting' and 'northing' tuples
            
        Returns:
            Dictionary mapping site_year to reflectance data
        """
        reflectance_data = {}
        
        for site in sites:
            for year in years:
                try:
                    # Get surface reflectance data
                    data = self.get_aop_data(
                        "DP1.30006.001",  # Surface reflectance
                        site,
                        year,
                        easting=bbox.get("easting") if bbox else None,
                        northing=bbox.get("northing") if bbox else None
                    )
                    
                    if "data" in data and "files" in data["data"]:
                        site_year_key = f"{site}_{year}"
                        reflectance_data[site_year_key] = self._process_aop_files(data["data"]["files"])
                        logger.info(f"Downloaded AOP reflectance for {site_year_key}")
                
                except Exception as e:
                    logger.error(f"Error downloading AOP data for {site} {year}: {str(e)}")
                    continue
        
        return reflectance_data
    
    def create_satellite_crosswalk(
        self,
        aop_data: Dict[str, pd.DataFrame],
        satellite_data: pd.DataFrame,
        temporal_window: int = 7
    ) -> pd.DataFrame:
        """
        Create a crosswalk between NEON AOP and satellite data.
        
        This function matches NEON hyperspectral data with satellite multispectral
        bands based on spectral response functions and temporal proximity.
        
        Args:
            aop_data: Dictionary of AOP reflectance data
            satellite_data: DataFrame with satellite observations
            temporal_window: Days to search for matching satellite observations
            
        Returns:
            DataFrame with matched AOP and satellite observations
        """
        crosswalk_records = []
        
        # Define spectral band mapping between NEON and common satellites
        band_mappings = {
            "sentinel2": {
                "B02": (459, 549),    # Blue
                "B03": (542, 578),    # Green
                "B04": (649, 681),    # Red
                "B05": (697, 713),    # Red Edge 1
                "B06": (732, 748),    # Red Edge 2
                "B07": (773, 793),    # Red Edge 3
                "B08": (784, 900),    # NIR
                "B8A": (854, 876),    # NIR narrow
                "B11": (1568, 1660),  # SWIR1
                "B12": (2115, 2290),  # SWIR2
            },
            "landsat8": {
                "B2": (450, 515),     # Blue
                "B3": (525, 600),     # Green
                "B4": (630, 680),     # Red
                "B5": (845, 885),     # NIR
                "B6": (1560, 1660),   # SWIR1
                "B7": (2100, 2300),   # SWIR2
            },
            "modis": {
                "B3": (459, 479),     # Blue
                "B4": (545, 565),     # Green
                "B1": (620, 670),     # Red
                "B2": (841, 876),     # NIR
                "B6": (1628, 1652),   # SWIR1
                "B7": (2105, 2155),   # SWIR2
            }
        }
        
        logger.info("Creating satellite crosswalk with NEON AOP data")
        
        # Process each AOP dataset
        for site_year, aop_df in aop_data.items():
            # Extract relevant metadata and spectral data
            # This is a simplified example - actual implementation would be more complex
            crosswalk_records.append({
                "site_year": site_year,
                "aop_bands": len(aop_df.columns),
                "satellite_matches": 0,
                "spectral_correlation": 0.0
            })
        
        return pd.DataFrame(crosswalk_records)
    
    def _process_aop_files(self, files: List[Dict]) -> pd.DataFrame:
        """Process AOP files and extract reflectance data."""
        # Simplified processing - actual implementation would handle HDF5 files
        processed_data = pd.DataFrame()
        
        for file_info in files:
            if file_info["name"].endswith(".h5"):
                # In reality, we'd download and process HDF5 files here
                logger.info(f"Processing AOP file: {file_info['name']}")
        
        return processed_data
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> requests.Response:
        """Make an API request with error handling."""
        url = urljoin(self.BASE_URL, endpoint)
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
    
    def _download_and_parse_csv(self, url: str) -> Optional[pd.DataFrame]:
        """Download and parse a CSV file from a URL."""
        try:
            df = pd.read_csv(url)
            return df
        except Exception as e:
            logger.error(f"Failed to download/parse CSV from {url}: {str(e)}")
            return None
    
    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check if data exists in cache and is recent."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        
        if cache_path.exists():
            # Check if cache is less than 24 hours old
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age < timedelta(hours=24):
                with open(cache_path, 'r') as f:
                    return json.load(f)
        
        return None
    
    def _save_cache(self, cache_key: str, data: Dict) -> None:
        """Save data to cache."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)


def main():
    """Example usage of the NEON data collector."""
    # Initialize collector
    collector = NEONDataCollector()
    
    # Get site information
    sites_df = collector.get_sites_info(site_codes=["SJER", "SOAP"])
    print(f"Retrieved information for {len(sites_df)} sites")
    
    # Download fire risk data
    fire_data = collector.download_fire_risk_data(
        sites=["SJER"],
        start_date="2023-01",
        end_date="2023-12",
        products=["DP1.00041.001", "DP1.00094.001"]  # Soil temp and moisture
    )
    
    print(f"Downloaded data for {len(fire_data)} products")
    for product, df in fire_data.items():
        print(f"{product}: {len(df)} records")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()