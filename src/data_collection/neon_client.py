"""
Enhanced NEON client for downloading and processing AOP data.

This module provides comprehensive access to NEON AOP data including
canopy height models, hyperspectral reflectance, and LiDAR data.
"""

import os
import requests
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
from datetime import datetime, timedelta
import time
import zipfile
import shutil

logger = logging.getLogger(__name__)


class NEONDataCollector:
    """Enhanced NEON data collector with AOP support."""
    
    def __init__(self, config: Dict):
        """
        Initialize NEON data collector.
        
        Args:
            config: Configuration dictionary with NEON API settings
        """
        self.config = config
        self.base_url = config.get('neon_api', {}).get('base_url', 'https://data.neonscience.org/api/v0')
        self.token = os.environ.get('NEON_API_TOKEN')
        self.timeout = config.get('neon_api', {}).get('timeout', 300)
        self.max_retries = config.get('neon_api', {}).get('max_retries', 3)
        
        if not self.token:
            logger.warning("No NEON API token found. Set NEON_API_TOKEN environment variable.")
        
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({'Authorization': f'Bearer {self.token}'})
    
    def get_sites(self) -> List[Dict]:
        """
        Get list of available NEON sites.
        
        Returns:
            List of site information dictionaries
        """
        logger.info("Fetching NEON sites...")
        
        try:
            response = self.session.get(f"{self.base_url}/sites", timeout=self.timeout)
            response.raise_for_status()
            
            sites = response.json()['data']
            logger.info(f"Found {len(sites)} NEON sites")
            return sites
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching sites: {e}")
            return []
    
    def get_site_info(self, site_code: str) -> Optional[Dict]:
        """
        Get detailed information for a specific site.
        
        Args:
            site_code: NEON site code
            
        Returns:
            Site information dictionary or None if not found
        """
        logger.info(f"Fetching site info for {site_code}")
        
        try:
            response = self.session.get(f"{self.base_url}/sites/{site_code}", timeout=self.timeout)
            response.raise_for_status()
            
            site_info = response.json()['data']
            logger.info(f"Retrieved info for site {site_code}")
            return site_info
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching site info for {site_code}: {e}")
            return None
    
    def get_available_products(self, site_code: str) -> List[Dict]:
        """
        Get available data products for a site.
        
        Args:
            site_code: NEON site code
            
        Returns:
            List of available data products
        """
        logger.info(f"Fetching available products for {site_code}")
        
        try:
            response = self.session.get(f"{self.base_url}/sites/{site_code}/dataProducts", timeout=self.timeout)
            response.raise_for_status()
            
            products = response.json()['data']
            logger.info(f"Found {len(products)} products for {site_code}")
            return products
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching products for {site_code}: {e}")
            return []
    
    def get_aop_data_info(self, site_code: str, year: int) -> List[Dict]:
        """
        Get AOP data information for a specific site and year.
        
        Args:
            site_code: NEON site code
            year: Data year
            
        Returns:
            List of AOP data information
        """
        logger.info(f"Fetching AOP data info for {site_code} {year}")
        
        try:
            # Get available products for the site
            products = self.get_available_products(site_code)
            
            # Filter for AOP products
            aop_products = []
            for product in products:
                if product.get('productCode') in ['DP3.30024.001', 'DP3.30026.001', 'DP3.30010.001', 'DP1.30003.001']:
                    # Get data availability for this product
                    product_code = product['productCode']
                    response = self.session.get(
                        f"{self.base_url}/sites/{site_code}/dataProducts/{product_code}/available",
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    
                    available_data = response.json()['data']
                    
                    # Filter for the specific year
                    year_data = [d for d in available_data if d.get('year') == year]
                    
                    for data in year_data:
                        aop_products.append({
                            'productCode': product_code,
                            'productName': product.get('productName', ''),
                            'year': year,
                            'month': data.get('month'),
                            'day': data.get('day'),
                            'url': data.get('downloadLink'),
                            'fileSize': data.get('fileSize'),
                            'checksum': data.get('checksum')
                        })
            
            logger.info(f"Found {len(aop_products)} AOP data files for {site_code} {year}")
            return aop_products
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching AOP data info for {site_code} {year}: {e}")
            return []
    
    def download_aop_data(self, site_code: str, year: int, output_dir: Path, 
                         products: Optional[List[str]] = None) -> bool:
        """
        Download AOP data for a specific site and year.
        
        Args:
            site_code: NEON site code
            year: Data year
            output_dir: Output directory for downloaded data
            products: List of product codes to download (None = all available)
            
        Returns:
            True if download successful, False otherwise
        """
        logger.info(f"Downloading AOP data for {site_code} {year}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get available AOP data
        aop_data = self.get_aop_data_info(site_code, year)
        
        if not aop_data:
            logger.warning(f"No AOP data available for {site_code} {year}")
            return False
        
        # Filter products if specified
        if products:
            aop_data = [d for d in aop_data if d['productCode'] in products]
        
        # Download each product
        success_count = 0
        for data in aop_data:
            product_code = data['productCode']
            download_url = data['url']
            
            if not download_url:
                logger.warning(f"No download URL for {product_code}")
                continue
            
            # Create product directory
            product_dir = output_dir / product_code
            product_dir.mkdir(exist_ok=True)
            
            # Download file
            if self._download_file(download_url, product_dir, data):
                success_count += 1
        
        logger.info(f"Successfully downloaded {success_count}/{len(aop_data)} AOP products for {site_code} {year}")
        return success_count > 0
    
    def _download_file(self, url: str, output_dir: Path, file_info: Dict) -> bool:
        """
        Download a single file from NEON.
        
        Args:
            url: Download URL
            output_dir: Output directory
            file_info: File information dictionary
            
        Returns:
            True if download successful, False otherwise
        """
        product_code = file_info['productCode']
        file_size = file_info.get('fileSize', 0)
        
        # Generate filename
        filename = f"{product_code}_{file_info['year']}_{file_info.get('month', '01')}_{file_info.get('day', '01')}.zip"
        file_path = output_dir / filename
        
        logger.info(f"Downloading {filename} ({file_size} bytes)")
        
        try:
            # Download with progress tracking
            response = self.session.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress for large files
                        if file_size > 0 and downloaded % (1024 * 1024) == 0:  # Every MB
                            progress = (downloaded / file_size) * 100
                            logger.info(f"Download progress: {progress:.1f}%")
            
            # Verify file size
            actual_size = file_path.stat().st_size
            if file_size > 0 and abs(actual_size - file_size) > 1024:  # Allow 1KB difference
                logger.warning(f"File size mismatch: expected {file_size}, got {actual_size}")
            
            # Extract zip file
            if file_path.suffix == '.zip':
                logger.info(f"Extracting {filename}")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                
                # Remove zip file after extraction
                file_path.unlink()
            
            logger.info(f"Successfully downloaded and extracted {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            if file_path.exists():
                file_path.unlink()
            return False
    
    def get_aop_metadata(self, site_code: str, year: int) -> Dict:
        """
        Get metadata for AOP data.
        
        Args:
            site_code: NEON site code
            year: Data year
            
        Returns:
            Metadata dictionary
        """
        logger.info(f"Fetching AOP metadata for {site_code} {year}")
        
        try:
            # Get site info
            site_info = self.get_site_info(site_code)
            if not site_info:
                return {}
            
            # Get AOP data info
            aop_data = self.get_aop_data_info(site_code, year)
            
            metadata = {
                'site_code': site_code,
                'site_name': site_info.get('siteName', ''),
                'year': year,
                'download_timestamp': datetime.now().isoformat(),
                'products': aop_data,
                'site_info': {
                    'latitude': site_info.get('location', {}).get('latitude'),
                    'longitude': site_info.get('location', {}).get('longitude'),
                    'elevation': site_info.get('location', {}).get('elevation'),
                    'domain': site_info.get('domainCode', ''),
                    'state': site_info.get('stateCode', '')
                }
            }
            
            logger.info(f"Retrieved metadata for {site_code} {year}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error fetching metadata for {site_code} {year}: {e}")
            return {}
    
    def validate_download(self, output_dir: Path, metadata: Dict) -> Dict:
        """
        Validate downloaded AOP data.
        
        Args:
            output_dir: Directory containing downloaded data
            metadata: Metadata dictionary
            
        Returns:
            Validation results dictionary
        """
        logger.info("Validating downloaded AOP data")
        
        validation_results = {
            'total_products': len(metadata.get('products', [])),
            'downloaded_products': 0,
            'valid_files': 0,
            'missing_files': [],
            'corrupted_files': [],
            'validation_passed': False
        }
        
        for product in metadata.get('products', []):
            product_code = product['productCode']
            product_dir = output_dir / product_code
            
            if product_dir.exists():
                validation_results['downloaded_products'] += 1
                
                # Check for valid files
                valid_files = list(product_dir.glob('*'))
                if valid_files:
                    validation_results['valid_files'] += 1
                else:
                    validation_results['missing_files'].append(product_code)
            else:
                validation_results['missing_files'].append(product_code)
        
        # Determine if validation passed
        if validation_results['downloaded_products'] > 0 and validation_results['valid_files'] > 0:
            validation_results['validation_passed'] = True
        
        logger.info(f"Validation results: {validation_results['validation_passed']}")
        return validation_results
    
    def cleanup_temp_files(self, output_dir: Path):
        """
        Clean up temporary files after processing.
        
        Args:
            output_dir: Output directory to clean
        """
        logger.info("Cleaning up temporary files")
        
        # Remove temporary zip files and other temp files
        temp_patterns = ['*.tmp', '*.temp', '*.zip']
        
        for pattern in temp_patterns:
            for temp_file in output_dir.glob(pattern):
                try:
                    temp_file.unlink()
                    logger.debug(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Could not remove temporary file {temp_file}: {e}")
    
    def download_aop_data_batch(self, sites: List[str], years: List[int], 
                               output_root: Path, products: Optional[List[str]] = None) -> Dict:
        """
        Download AOP data for multiple sites and years.
        
        Args:
            sites: List of site codes
            years: List of years
            output_root: Root output directory
            products: List of product codes to download
            
        Returns:
            Summary of download results
        """
        logger.info(f"Starting batch download for {len(sites)} sites and {len(years)} years")
        
        results = {
            'total_sites': len(sites),
            'total_years': len(years),
            'successful_downloads': 0,
            'failed_downloads': 0,
            'download_details': {}
        }
        
        start_time = time.time()
        
        for site in sites:
            results['download_details'][site] = {}
            
            for year in years:
                logger.info(f"Processing {site} {year}")
                
                # Create site-year directory
                site_year_dir = output_root / site / str(year)
                
                try:
                    # Download data
                    success = self.download_aop_data(site, year, site_year_dir, products)
                    
                    if success:
                        results['successful_downloads'] += 1
                        results['download_details'][site][year] = 'success'
                        
                        # Get metadata and validate
                        metadata = self.get_aop_metadata(site, year)
                        validation = self.validate_download(site_year_dir, metadata)
                        
                        # Save metadata
                        metadata_file = site_year_dir / 'metadata.json'
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        # Cleanup temp files
                        self.cleanup_temp_files(site_year_dir)
                        
                    else:
                        results['failed_downloads'] += 1
                        results['download_details'][site][year] = 'failed'
                    
                except Exception as e:
                    logger.error(f"Error processing {site} {year}: {e}")
                    results['failed_downloads'] += 1
                    results['download_details'][site][year] = 'error'
                
                # Add delay between downloads to be respectful
                time.sleep(1)
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        results['average_time_per_download'] = total_time / (len(sites) * len(years))
        
        logger.info(f"Batch download completed in {total_time:.1f} seconds")
        logger.info(f"Successful: {results['successful_downloads']}, Failed: {results['failed_downloads']}")
        
        return results


def download_aop_data_cli():
    """CLI interface for downloading AOP data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download NEON AOP data")
    parser.add_argument("--site", required=True, help="NEON site code")
    parser.add_argument("--years", nargs="+", type=int, required=True, help="Years to download")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--config", default="configs/aop_sites.yaml", help="Configuration file")
    parser.add_argument("--products", nargs="+", help="Specific products to download")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize collector
    collector = NEONDataCollector(config)
    
    # Download data
    output_dir = Path(args.output_dir)
    success = collector.download_aop_data_batch(
        [args.site], args.years, output_dir, args.products
    )
    
    if success['successful_downloads'] > 0:
        print(f"Successfully downloaded AOP data for {args.site}")
        print(f"Output directory: {output_dir}")
    else:
        print(f"Failed to download AOP data for {args.site}")
        exit(1)


if __name__ == "__main__":
    download_aop_data_cli()