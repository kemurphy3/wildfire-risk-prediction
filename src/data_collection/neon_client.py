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