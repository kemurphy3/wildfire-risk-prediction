"""
NEON data downloader for AOP data retrieval.

Manages download of airborne observation platform data including LiDAR, 
hyperspectral, and RGB imagery. Note: These files can be several gigabytes in size.
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
    """Downloads NEON data, handles retries, unzips files, the works."""
    
    def __init__(self, config: Dict):
        """
        Set up the NEON downloader.
        
        Args:
            config: dict with API urls, timeouts, etc
        """
        self.config = config
        self.base_url = config.get('neon_api', {}).get('base_url', 'https://data.neonscience.org/api/v0')
        self.token = os.environ.get('NEON_API_TOKEN')
        self.timeout = config.get('neon_api', {}).get('timeout', 300)
        self.max_retries = config.get('neon_api', {}).get('max_retries', 3)
        
        if not self.token:
            logger.warning("No API token! Set NEON_API_TOKEN env var")
        
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({'Authorization': f'Bearer {self.token}'})
    
    def get_sites(self) -> List[Dict]:
        """
        Get all NEON sites.
        
        Returns:
            list of site info dicts
        """
        logger.info("Getting site list...")
        
        try:
            response = self.session.get(f"{self.base_url}/sites", timeout=self.timeout)
            response.raise_for_status()
            
            sites = response.json()['data']
            logger.info(f"Got {len(sites)} sites")
            return sites
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed getting sites: {e}")
            return []
    
    def get_site_info(self, site_code: str) -> Optional[Dict]:
        """
        Get details for one site.
        
        Args:
            site_code: 4-letter site code
            
        Returns:
            site info dict or None
        """
        logger.info(f"Getting info for {site_code}")
        
        try:
            response = self.session.get(f"{self.base_url}/sites/{site_code}", timeout=self.timeout)
            response.raise_for_status()
            
            site_info = response.json()['data']
            logger.info(f"Got {site_code} info")
            return site_info
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Couldn't get {site_code}: {e}")
            return None
    
    def get_available_products(self, site_code: str) -> List[Dict]:
        """
        See what data products exist for a site.
        
        Args:
            site_code: which site
            
        Returns:
            list of available products
        """
        logger.info(f"Checking products for {site_code}")
        
        try:
            response = self.session.get(f"{self.base_url}/sites/{site_code}/dataProducts", timeout=self.timeout)
            response.raise_for_status()
            
            products = response.json()['data']
            logger.info(f"{len(products)} products available")
            return products
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Product fetch failed: {e}")
            return []
    
    def get_aop_data_info(self, site_code: str, year: int) -> List[Dict]:
        """
        Find what AOP data is available for site/year.
        
        Args:
            site_code: which site
            year: which year
            
        Returns:
            list of available AOP files
        """
        logger.info(f"Looking for AOP data: {site_code} {year}")
        
        try:
            # get all products first
            products = self.get_available_products(site_code)
            
            # filter to just AOP stuff
            aop_products = []
            for product in products:
                # these are the AOP product codes we want
                if product.get('productCode') in ['DP3.30024.001', 'DP3.30026.001', 'DP3.30010.001', 'DP1.30003.001']:
                    product_code = product['productCode']
                    response = self.session.get(
                        f"{self.base_url}/sites/{site_code}/dataProducts/{product_code}/available",
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    
                    available_data = response.json()['data']
                    
                    # get just the year we want
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
            
            logger.info(f"Found {len(aop_products)} AOP files")
            return aop_products
            
        except requests.exceptions.RequestException as e:
            logger.error(f"AOP info failed: {e}")
            return []
    
    def download_aop_data(self, site_code: str, year: int, output_dir: Path, 
                         products: Optional[List[str]] = None) -> bool:
        """
        Download AOP data files (warning: BIG downloads!).
        
        Args:
            site_code: which site
            year: which year
            output_dir: where to save
            products: specific products only (None = all)
            
        Returns:
            True if got something, False if failed
        """
        logger.info(f"Starting download: {site_code} {year}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # see what's available
        aop_data = self.get_aop_data_info(site_code, year)
        
        if not aop_data:
            logger.warning(f"No data for {site_code} {year}")
            return False
        
        # filter if requested
        if products:
            aop_data = [d for d in aop_data if d['productCode'] in products]
        
        # Download all files
        success_count = 0
        for data in aop_data:
            product_code = data['productCode']
            download_url = data['url']
            
            if not download_url:
                logger.warning(f"No URL for {product_code}")
                continue
            
            # make product folder
            product_dir = output_dir / product_code
            product_dir.mkdir(exist_ok=True)
            
            # download it!
            if self._download_file(download_url, product_dir, data):
                success_count += 1
        
        logger.info(f"Downloaded {success_count}/{len(aop_data)} products")
        return success_count > 0
    
    def _download_file(self, url: str, output_dir: Path, file_info: Dict) -> bool:
        """
        Actually download one file.
        
        Args:
            url: download link
            output_dir: where to save
            file_info: metadata about the file
            
        Returns:
            True if worked, False if not
        """
        product_code = file_info['productCode']
        file_size = file_info.get('fileSize', 0)
        
        # make filename
        filename = f"{product_code}_{file_info['year']}_{file_info.get('month', '01')}_{file_info.get('day', '01')}.zip"
        file_path = output_dir / filename
        
        logger.info(f"Downloading {filename} ({file_size} bytes)")
        
        try:
            # stream download (these files are huge)
            response = self.session.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # show progress every MB
                        if file_size > 0 and downloaded % (1024 * 1024) == 0:
                            progress = (downloaded / file_size) * 100
                            logger.info(f"Progress: {progress:.1f}%")
            
            # check size is right
            actual_size = file_path.stat().st_size
            if file_size > 0 and abs(actual_size - file_size) > 1024:  # 1KB tolerance
                logger.warning(f"Size mismatch! Expected {file_size}, got {actual_size}")
            
            # unzip it
            if file_path.suffix == '.zip':
                logger.info(f"Unzipping {filename}...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                
                # delete zip to save space
                file_path.unlink()
            
            logger.info(f"Done with {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed for {filename}: {e}")
            if file_path.exists():
                file_path.unlink()  # cleanup partial download
            return False
    
    def get_aop_metadata(self, site_code: str, year: int) -> Dict:
        """
        Get metadata about AOP data.
        
        Args:
            site_code: site
            year: year
            
        Returns:
            metadata dict
        """
        logger.info(f"Getting metadata: {site_code} {year}")
        
        try:
            # get site details
            site_info = self.get_site_info(site_code)
            if not site_info:
                return {}
            
            # get AOP info
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
            
            logger.info(f"Got metadata")
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata failed: {e}")
            return {}
    
    def validate_download(self, output_dir: Path, metadata: Dict) -> Dict:
        """
        Check if downloads worked properly.
        
        Args:
            output_dir: where we downloaded to
            metadata: expected metadata
            
        Returns:
            validation results
        """
        logger.info("Checking downloads...")
        
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
                
                # make sure files exist
                valid_files = list(product_dir.glob('*'))
                if valid_files:
                    validation_results['valid_files'] += 1
                else:
                    validation_results['missing_files'].append(product_code)
            else:
                validation_results['missing_files'].append(product_code)
        
        # did we get anything?
        if validation_results['downloaded_products'] > 0 and validation_results['valid_files'] > 0:
            validation_results['validation_passed'] = True
        
        logger.info(f"Validation: {'PASS' if validation_results['validation_passed'] else 'FAIL'}")
        return validation_results
    
    def cleanup_temp_files(self, output_dir: Path):
        """
        Delete temp files to save space.
        
        Args:
            output_dir: dir to clean
        """
        logger.info("Cleaning up temp files...")
        
        # Remove temporary files
        temp_patterns = ['*.tmp', '*.temp', '*.zip']
        
        for pattern in temp_patterns:
            for temp_file in output_dir.glob(pattern):
                try:
                    temp_file.unlink()
                    logger.debug(f"Deleted: {temp_file}")
                except Exception as e:
                    logger.warning(f"Couldn't delete {temp_file}: {e}")
    
    def download_aop_data_batch(self, sites: List[str], years: List[int], 
                               output_root: Path, products: Optional[List[str]] = None) -> Dict:
        """
        Batch download for multiple sites/years.
        
        Args:
            sites: site codes
            years: years to download
            output_root: base output dir
            products: specific products (None = all)
            
        Returns:
            summary of what worked/failed
        """
        logger.info(f"Batch download: {len(sites)} sites x {len(years)} years")
        
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
                logger.info(f"Working on {site} {year}")
                
                # make folder
                site_year_dir = output_root / site / str(year)
                
                try:
                    # try to download
                    success = self.download_aop_data(site, year, site_year_dir, products)
                    
                    if success:
                        results['successful_downloads'] += 1
                        results['download_details'][site][year] = 'success'
                        
                        # validate & save metadata
                        metadata = self.get_aop_metadata(site, year)
                        validation = self.validate_download(site_year_dir, metadata)
                        
                        # save metadata
                        metadata_file = site_year_dir / 'metadata.json'
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        # cleanup
                        self.cleanup_temp_files(site_year_dir)
                        
                    else:
                        results['failed_downloads'] += 1
                        results['download_details'][site][year] = 'failed'
                    
                except Exception as e:
                    logger.error(f"Failed {site} {year}: {e}")
                    results['failed_downloads'] += 1
                    results['download_details'][site][year] = 'error'
                
                # be nice to NEON servers
                time.sleep(1)
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        results['average_time_per_download'] = total_time / (len(sites) * len(years))
        
        logger.info(f"Batch done in {total_time:.1f}s")
        logger.info(f"Success: {results['successful_downloads']}, Failed: {results['failed_downloads']}")
        
        return results


def download_aop_data_cli():
    """Quick CLI for downloading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NEON AOP downloader")
    parser.add_argument("--site", required=True, help="NEON site code")
    parser.add_argument("--years", nargs="+", type=int, required=True, help="Years to download")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--config", default="configs/aop_sites.yaml", help="Configuration file")
    parser.add_argument("--products", nargs="+", help="Specific products to download")
    
    args = parser.parse_args()
    
    # load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # setup downloader
    collector = NEONDataCollector(config)
    
    # go!
    output_dir = Path(args.output_dir)
    success = collector.download_aop_data_batch(
        [args.site], args.years, output_dir, args.products
    )
    
    if success['successful_downloads'] > 0:
        print(f"Downloaded {args.site} data!")
        print(f"Saved to: {output_dir}")
    else:
        print(f"Download failed for {args.site} :(")
        exit(1)


if __name__ == "__main__":
    download_aop_data_cli()