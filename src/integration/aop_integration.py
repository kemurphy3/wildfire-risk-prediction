"""
AOP Integration - connects airplane data with wildfire predictions.

Glues together the NEON crosswalk stuff with our main prediction pipeline.
Makes everything work better by using high-res data to fix satellite blindspots.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import yaml
import joblib
from datetime import datetime

# Import AOP modules
from ..features.aop_crosswalk import load_crosswalk_models, apply_crosswalk_models
from ..features.aop_features import create_aop_bundle, process_aop_to_grid
from ..utils.geoalign import validate_alignment

# Import main system modules
from ..features.fire_features import FireRiskFeatureEngine
from ..models.ensemble import EnsembleFireRiskModel

logger = logging.getLogger(__name__)


class AOPIntegrationManager:
    """Main integration manager - handles all the AOP/satellite mashup stuff."""
    
    def __init__(self, config_path: str):
        """
        Setup the integration manager.
        
        Args:
            config_path: yaml config file location
        """
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = Path(self.config['paths']['models_root'])
        self.processed_dir = Path(self.config['paths']['processed_data_root'])
        self.outputs_dir = Path(self.config['paths']['outputs_root'])
        
        # make sure dirs exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # load any existing models
        self.crosswalk_models = self._load_crosswalk_models()
        
        # init feature engine
        self.feature_engine = FireRiskFeatureEngine()
        
        logger.info("Integration manager ready!")
    
    def _load_crosswalk_models(self) -> Dict:
        """Load any existing crosswalk models."""
        if not self.models_dir.exists():
            logger.warning("No models dir yet")
            return {}
        
        try:
            models = load_crosswalk_models(self.models_dir)
            logger.info(f"Found {len(models)} models")
            return models
        except Exception as e:
            logger.error(f"Couldn't load models: {e}")
            return {}
    
    def get_enhanced_features(self, satellite_data: pd.DataFrame, 
                             site_code: str, year: int) -> pd.DataFrame:
        """
        Enhance satellite data with AOP-calibrated features.
        
        Args:
            satellite_data: regular satellite indices
            site_code: which NEON site
            year: what year
            
        Returns:
            DataFrame with extra calibrated columns
        """
        logger.info(f"Enhancing features for {site_code} {year}")
        
        # work on a copy
        enhanced_data = satellite_data.copy()
        
        # apply crosswalk if we have models
        if self.crosswalk_models:
            enhanced_data = apply_crosswalk_models(satellite_data, self.crosswalk_models)
            logger.info(f"Applied {len(self.crosswalk_models)} crosswalk models")
        else:
            logger.warning("No models - using raw satellite data")
        
        # try to add direct AOP features
        aop_features = self._get_aop_features(site_code, year)
        if aop_features is not None:
            enhanced_data = self._merge_aop_features(enhanced_data, aop_features)
            logger.info("Added AOP features")
        
        # add metadata about data quality
        enhanced_data = self._add_quality_indicators(enhanced_data, site_code, year)
        
        return enhanced_data
    
    def _get_aop_features(self, site_code: str, year: int) -> Optional[pd.DataFrame]:
        """Try to load processed AOP features."""
        aop_dir = self.processed_dir / site_code / str(year)
        
        if not aop_dir.exists():
            logger.debug(f"No AOP data for {site_code} {year}")
            return None
        
        # look for processed features
        feature_files = list(aop_dir.glob("*_features.gpkg"))
        if not feature_files:
            logger.debug(f"No features in {aop_dir}")
            return None
        
        try:
            import geopandas as gpd
            features_df = gpd.read_file(feature_files[0])
            logger.info(f"Got AOP features: {list(features_df.columns)}")
            return features_df
        except Exception as e:
            logger.error(f"Failed loading AOP features: {e}")
            return None
    
    def _merge_aop_features(self, satellite_data: pd.DataFrame, 
                           aop_features: pd.DataFrame) -> pd.DataFrame:
        """Merge AOP with satellite data (TODO: implement spatial join)."""
        # disabled for now - would explode the columns
        logger.debug("AOP merge disabled (not implemented yet)")
        return satellite_data
    
    def _add_quality_indicators(self, data: pd.DataFrame, site_code: str, 
                               year: int) -> pd.DataFrame:
        """Add metadata about data quality/availability."""
        # work on copy
        data = data.copy()
        
        # how many models do we have?
        data['crosswalk_models_available'] = len(self.crosswalk_models)
        
        # is AOP data there?
        aop_dir = self.processed_dir / site_code / str(year)
        data['aop_data_available'] = aop_dir.exists()
        
        # timestamp
        data['enhancement_timestamp'] = datetime.now().isoformat()
        
        return data
    
    def validate_integration(self, site_code: str, year: int) -> Dict:
        """
        Check if everything's working properly.
        
        Args:
            site_code: NEON site
            year: year to check
            
        Returns:
            dict with validation results
        """
        logger.info(f"Checking integration: {site_code} {year}")
        
        validation_results = {
            'site_code': site_code,
            'year': year,
            'timestamp': datetime.now().isoformat(),
            'crosswalk_models': len(self.crosswalk_models),
            'aop_data_available': False,
            'feature_enhancement': False,
            'integration_status': 'unknown'
        }
        
        # check for AOP data
        aop_dir = self.processed_dir / site_code / str(year)
        if aop_dir.exists():
            validation_results['aop_data_available'] = True
            
            # count feature files
            feature_files = list(aop_dir.glob("*_features.gpkg"))
            validation_results['feature_files_count'] = len(feature_files)
            
            # metadata?
            metadata_files = list(aop_dir.glob("metadata.json"))
            validation_results['metadata_available'] = len(metadata_files) > 0
        
        # check models
        if self.crosswalk_models:
            validation_results['feature_enhancement'] = True
            
            # get model stats
            model_performance = {}
            for target_var, model_info in self.crosswalk_models.items():
                if hasattr(model_info, 'metrics_'):
                    model_performance[target_var] = {
                        'test_r2': model_info.metrics_.get('test_r2', 0),
                        'test_mae': model_info.metrics_.get('test_mae', float('inf'))
                    }
                else:
                    model_performance[target_var] = {'status': 'no_metrics'}
            
            validation_results['model_performance'] = model_performance
        
        # figure out overall status
        if validation_results['aop_data_available'] and validation_results['feature_enhancement']:
            validation_results['integration_status'] = 'fully_integrated'
        elif validation_results['aop_data_available']:
            validation_results['integration_status'] = 'aop_only'
        elif validation_results['feature_enhancement']:
            validation_results['integration_status'] = 'crosswalk_only'
        else:
            validation_results['integration_status'] = 'not_integrated'
        
        logger.info(f"Status: {validation_results['integration_status']}")
        return validation_results
    
    def generate_integration_report(self, site_codes: List[str], years: List[int]) -> str:
        """
        Generate a nice HTML report of what's working.
        
        Args:
            site_codes: which sites to check
            years: which years
            
        Returns:
            HTML report as string
        """
        logger.info("Making integration report...")
        
        # check all combos
        validation_results = []
        for site in site_codes:
            for year in years:
                validation = self.validate_integration(site, year)
                validation_results.append(validation)
        
        # make the HTML
        html_content = self._generate_html_report(validation_results)
        
        # save it
        report_path = self.outputs_dir / f"aop_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report saved: {report_path}")
        return html_content
    
    def _generate_html_report(self, validation_results: List[Dict]) -> str:
        """Build the actual HTML report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AOP Integration Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .status-fully_integrated {{ color: green; font-weight: bold; }}
                .status-aop_only {{ color: orange; font-weight: bold; }}
                .status-crosswalk_only {{ color: blue; font-weight: bold; }}
                .status-not_integrated {{ color: red; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AOP Integration Report</h1>
                <p>Generated: {timestamp}</p>
            </div>
        """
        
        # calc summary stats
        total_sites = len(set(r['site_code'] for r in validation_results))
        total_years = len(set(r['year'] for r in validation_results))
        fully_integrated = sum(1 for r in validation_results if r['integration_status'] == 'fully_integrated')
        aop_only = sum(1 for r in validation_results if r['integration_status'] == 'aop_only')
        crosswalk_only = sum(1 for r in validation_results if r['integration_status'] == 'crosswalk_only')
        not_integrated = sum(1 for r in validation_results if r['integration_status'] == 'not_integrated')
        
        html += f"""
            <div class="summary">
                <h2>Integration Summary</h2>
                <p><strong>Total Sites:</strong> {total_sites}</p>
                <p><strong>Total Years:</strong> {total_years}</p>
                <p><strong>Fully Integrated:</strong> {fully_integrated}</p>
                <p><strong>AOP Only:</strong> {aop_only}</p>
                <p><strong>Crosswalk Only:</strong> {crosswalk_only}</p>
                <p><strong>Not Integrated:</strong> {not_integrated}</p>
            </div>
        """
        
        # Detailed results table
        html += """
            <h2>Detailed Integration Status</h2>
            <table>
                <tr>
                    <th>Site</th>
                    <th>Year</th>
                    <th>Crosswalk Models</th>
                    <th>AOP Data Available</th>
                    <th>Feature Enhancement</th>
                    <th>Integration Status</th>
                </tr>
        """
        
        for result in validation_results:
            status_class = f"status-{result['integration_status']}"
            html += f"""
                <tr>
                    <td>{result['site_code']}</td>
                    <td>{result['year']}</td>
                    <td>{result['crosswalk_models']}</td>
                    <td>{'Yes' if result['aop_data_available'] else 'No'}</td>
                    <td>{'Yes' if result['feature_enhancement'] else 'No'}</td>
                    <td class="{status_class}">{result['integration_status'].replace('_', ' ').title()}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Recommendations</h2>
            <ul>
                <li><strong>Fully Integrated:</strong> These sites are ready for enhanced wildfire risk prediction</li>
                <li><strong>AOP Only:</strong> Download and process AOP data to enable crosswalk models</li>
                <li><strong>Crosswalk Only:</strong> AOP data needed for validation and improvement</li>
                <li><strong>Not Integrated:</strong> Start with AOP data download and processing</li>
            </ul>
            
            <h2>Next Steps</h2>
            <ol>
                <li>For sites with 'Not Integrated' status: Run 'make download-SITE' to get AOP data</li>
                <li>For sites with 'AOP Only' status: Run 'make process' to extract features</li>
                <li>For sites with 'Crosswalk Only' status: Download AOP data for validation</li>
                <li>For all sites: Run 'make calibrate' to train crosswalk models</li>
                <li>Validate integration with 'make validate'</li>
            </ol>
        </body>
        </html>
        """
        
        return html
    
    def export_enhanced_features(self, site_code: str, year: int, 
                                output_format: str = 'csv') -> Optional[Path]:
        """
        Export enhanced features to file.
        
        Args:
            site_code: which site
            year: which year
            output_format: 'csv', 'parquet', or 'geopackage'
            
        Returns:
            Path to saved file (or None if it failed)
        """
        logger.info(f"Exporting features: {site_code} {year}")
        
        # TODO: load real satellite data from main system
        # using fake data for now
        satellite_data = self._create_sample_satellite_data()
        
        # enhance the features
        enhanced_data = self.get_enhanced_features(satellite_data, site_code, year)
        
        # export to requested format
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_format == 'csv':
            output_path = self.outputs_dir / f"{site_code}_{year}_enhanced_features_{timestamp}.csv"
            enhanced_data.to_csv(output_path, index=False)
        elif output_format == 'parquet':
            output_path = self.outputs_dir / f"{site_code}_{year}_enhanced_features_{timestamp}.parquet"
            enhanced_data.to_parquet(output_path, index=False)
        elif output_format == 'geopackage':
            output_path = self.outputs_dir / f"{site_code}_{year}_enhanced_features_{timestamp}.gpkg"
            # need geometry column for geopackage
            if 'geometry' in enhanced_data.columns:
                import geopandas as gpd
                gdf = gpd.GeoDataFrame(enhanced_data)
                gdf.to_file(output_path, driver='GPKG')
            else:
                logger.warning("No geometry - falling back to CSV")
                enhanced_data.to_csv(output_path.with_suffix('.csv'), index=False)
                output_path = output_path.with_suffix('.csv')
        else:
            logger.error(f"Unknown format: {output_format}")
            return None
        
        logger.info(f"Saved to {output_path}")
        return output_path
    
    def _create_sample_satellite_data(self) -> pd.DataFrame:
        """Make fake satellite data for testing."""
        # generate some realistic-ish values
        n_samples = 100
        
        data = {
            'ndvi': np.random.uniform(0.1, 0.8, n_samples),
            'nbr': np.random.uniform(-0.3, 0.6, n_samples),
            'ndwi': np.random.uniform(-0.5, 0.4, n_samples),
            'evi': np.random.uniform(0.1, 0.7, n_samples),
            'savi': np.random.uniform(0.1, 0.6, n_samples),
            'msavi': np.random.uniform(0.1, 0.6, n_samples),
            'nbr2': np.random.uniform(-0.2, 0.5, n_samples),
            'ndsi': np.random.uniform(-0.3, 0.4, n_samples)
        }
        
        return pd.DataFrame(data)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AOP Integration CLI")
    parser.add_argument("--config", default="configs/aop_sites.yaml", help="Configuration file")
    parser.add_argument("--sites", nargs="+", help="Site codes to process")
    parser.add_argument("--years", nargs="+", type=int, help="Years to process")
    parser.add_argument("--action", choices=["validate", "report", "export"], default="validate", help="Action to perform")
    parser.add_argument("--output-format", choices=["csv", "parquet", "geopackage"], default="csv", help="Output format for export")
    
    args = parser.parse_args()
    
    # setup manager
    manager = AOPIntegrationManager(args.config)
    
    if args.action == "validate":
        # check each site/year combo
        for site in args.sites:
            for year in args.years:
                validation = manager.validate_integration(site, year)
                print(f"{site} {year}: {validation['integration_status']}")
    
    elif args.action == "report":
        # make report
        report = manager.generate_integration_report(args.sites, args.years)
        print("Report generated!")
    
    elif args.action == "export":
        # export features
        for site in args.sites:
            for year in args.years:
                output_path = manager.export_enhanced_features(site, year, args.output_format)
                if output_path:
                    print(f"Exported {site} {year} -> {output_path}")


if __name__ == "__main__":
    main()
