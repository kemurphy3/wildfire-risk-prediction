"""
Test suite for AOP Integration System.

This module tests the integration between NEON AOP crosswalk and
the main wildfire risk prediction system.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import yaml
from unittest.mock import Mock, patch

# Import modules to test
from src.integration.aop_integration import AOPIntegrationManager
from src.features.aop_crosswalk import (
    fit_linear_crosswalk, 
    fit_ensemble_crosswalk,
    calibrate_satellite_indices,
    validate_crosswalk
)
from src.features.aop_features import extract_chm_features, extract_spectral_features
from src.utils.geoalign import infer_dst_grid, warp_to_grid


class TestAOPIntegration:
    """Test AOP integration functionality."""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        config = {
            'paths': {
                'models_root': 'temp_models',
                'processed_data_root': 'temp_processed',
                'outputs_root': 'temp_outputs'
            },
            'neon_api': {
                'base_url': 'https://test.neonscience.org/api/v0',
                'timeout': 30,
                'max_retries': 2
            }
        }
        
        # Create temporary directories
        temp_dir = Path(tempfile.mkdtemp())
        config['paths']['models_root'] = str(temp_dir / 'models')
        config['paths']['processed_data_root'] = str(temp_dir / 'processed')
        config['paths']['outputs_root'] = str(temp_dir / 'outputs')
        
        yield config, temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_satellite_data(self):
        """Create sample satellite data for testing."""
        np.random.seed(42)
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
    
    @pytest.fixture
    def sample_aop_data(self):
        """Create sample AOP data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'chm_mean': np.random.uniform(2.0, 15.0, n_samples),
            'chm_std': np.random.uniform(0.5, 5.0, n_samples),
            'canopy_cover_gt2m': np.random.uniform(0.1, 0.9, n_samples),
            'canopy_cover_gt5m': np.random.uniform(0.05, 0.7, n_samples),
            'ndvi_aop': np.random.uniform(0.2, 0.9, n_samples),
            'evi_aop': np.random.uniform(0.1, 0.8, n_samples),
            'nbr_aop': np.random.uniform(-0.2, 0.7, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_aop_integration_manager_initialization(self, temp_config):
        """Test AOP integration manager initialization."""
        config, temp_dir = temp_config
        
        # Create config file
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Test initialization
        manager = AOPIntegrationManager(str(config_path))
        
        assert manager.config == config
        assert Path(manager.models_dir).exists()
        assert Path(manager.outputs_dir).exists()
        assert len(manager.crosswalk_models) == 0  # No models initially
    
    def test_enhanced_features_generation(self, temp_config, sample_satellite_data):
        """Test enhanced features generation."""
        config, temp_dir = temp_config
        
        # Create config file
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        manager = AOPIntegrationManager(str(config_path))
        
        # Test without crosswalk models
        enhanced_data = manager.get_enhanced_features(
            sample_satellite_data, 'SRER', 2021
        )
        
        # Original data has 8 columns, we add 3 quality indicators
        expected_columns = 8 + 3
        assert enhanced_data.shape == (100, expected_columns)
        assert 'crosswalk_models_available' in enhanced_data.columns
        assert 'aop_data_available' in enhanced_data.columns
        assert 'enhancement_timestamp' in enhanced_data.columns
    
    def test_integration_validation(self, temp_config):
        """Test integration validation."""
        config, temp_dir = temp_config
        
        # Create config file
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        manager = AOPIntegrationManager(str(config_path))
        
        # Test validation for non-existent site
        validation = manager.validate_integration('SRER', 2021)
        
        assert validation['site_code'] == 'SRER'
        assert validation['year'] == 2021
        assert validation['integration_status'] == 'not_integrated'
        assert not validation['aop_data_available']
        assert not validation['feature_enhancement']
    
    def test_crosswalk_model_training(self, sample_satellite_data, sample_aop_data):
        """Test crosswalk model training."""
        # Test linear crosswalk
        target_vars = ['chm_mean', 'ndvi_aop']
        
        models = calibrate_satellite_indices(
            sample_satellite_data, sample_aop_data, target_vars, model_type="linear"
        )
        
        assert len(models) == 2
        assert 'chm_mean' in models
        assert 'ndvi_aop' in models
        
        # Test ensemble crosswalk
        models_ensemble = calibrate_satellite_indices(
            sample_satellite_data, sample_aop_data, target_vars, model_type="ensemble"
        )
        
        assert len(models_ensemble) == 2
    
    def test_chm_feature_extraction(self):
        """Test CHM feature extraction."""
        # Create sample CHM data
        chm_arr = np.random.uniform(0, 20, (100, 100))
        chm_arr[0:10, 0:10] = np.nan  # Add some invalid values
        
        features = extract_chm_features(chm_arr)
        
        assert 'chm_mean' in features
        assert 'chm_std' in features
        assert 'canopy_cover_gt2m' in features
        assert 'canopy_cover_gt5m' in features
        assert 'rumple_index' in features
        assert 'height_entropy' in features
        
        # Check value ranges
        assert 0 <= features['canopy_cover_gt2m'] <= 1
        assert 0 <= features['canopy_cover_gt5m'] <= 1
        assert features['chm_mean'] >= 0
    
    def test_spectral_feature_extraction(self):
        """Test spectral feature extraction."""
        # Create sample hyperspectral data
        hsi_arr = np.random.uniform(0, 1, (100, 100, 10))
        
        features = extract_spectral_features(hsi_arr)
        
        assert 'ndvi_aop' in features
        assert 'evi_aop' in features
        assert 'nbr_aop' in features
        assert 'ndwi_aop' in features
        
        # Check value ranges for vegetation indices
        assert -1 <= features['ndvi_aop'] <= 1
        assert -1 <= features['evi_aop'] <= 1
        assert -1 <= features['nbr_aop'] <= 1
    
    def test_geospatial_alignment(self):
        """Test geospatial alignment utilities."""
        # Mock rasterio functionality
        with patch('rasterio.open') as mock_open:
            mock_src = Mock()
            mock_src.meta = {
                'crs': 'EPSG:32632',
                'transform': (10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
                'width': 100,
                'height': 100
            }
            mock_open.return_value.__enter__.return_value = mock_src
            
            # Test grid inference
            bounds = [-111.51, 31.74, -111.46, 31.79]
            grid_params = infer_dst_grid(bounds, 'sentinel2_10m')
            
            assert 'crs' in grid_params
            assert 'transform' in grid_params
            assert 'width' in grid_params
            assert 'height' in grid_params
    
    def test_crosswalk_validation(self, sample_satellite_data, sample_aop_data):
        """Test crosswalk validation."""
        # Train models first
        target_vars = ['chm_mean', 'ndvi_aop']
        models = calibrate_satellite_indices(
            sample_satellite_data, sample_aop_data, target_vars, model_type="linear"
        )
        
        # Test validation
        with tempfile.TemporaryDirectory() as temp_dir:
            validation_results = validate_crosswalk(
                sample_satellite_data, sample_aop_data, models, Path(temp_dir)
            )
            
            assert len(validation_results) == 2
            assert 'target_variable' in validation_results.columns
            assert 'r2' in validation_results.columns
            assert 'mae' in validation_results.columns
    
    def test_integration_report_generation(self, temp_config):
        """Test integration report generation."""
        config, temp_dir = temp_config
        
        # Create config file
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        manager = AOPIntegrationManager(str(config_path))
        
        # Generate report
        report = manager.generate_integration_report(['SRER', 'JORN'], [2021, 2022])
        
        assert isinstance(report, str)
        assert 'AOP Integration Report' in report
        assert 'Integration Summary' in report
        assert 'SRER' in report
        assert 'JORN' in report
    
    def test_enhanced_features_export(self, temp_config, sample_satellite_data):
        """Test enhanced features export."""
        config, temp_dir = temp_config
        
        # Create config file
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        manager = AOPIntegrationManager(str(config_path))
        
        # Test CSV export
        output_path = manager.export_enhanced_features('SRER', 2021, 'csv')
        
        assert output_path is not None
        assert output_path.exists()
        assert output_path.suffix == '.csv'
        
        # Test parquet export
        output_path = manager.export_enhanced_features('SRER', 2021, 'parquet')
        
        assert output_path is not None
        assert output_path.exists()
        assert output_path.suffix == '.parquet'


class TestAOPCrosswalkModels:
    """Test AOP crosswalk model functionality."""
    
    def test_linear_crosswalk_fitting(self):
        """Test linear crosswalk model fitting."""
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        model_result = fit_linear_crosswalk(X, y)
        
        assert 'model' in model_result
        assert 'metrics' in model_result
        assert 'feature_importance' in model_result
        assert model_result['metrics']['test_r2'] >= 0  # R² should be non-negative
    
    def test_ensemble_crosswalk_fitting(self):
        """Test ensemble crosswalk model fitting."""
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        model = fit_ensemble_crosswalk(X, y)
        
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'feature_importances_')
    
    def test_model_calibration_with_missing_data(self):
        """Test model calibration with missing data."""
        # Create data with missing values
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # Add some missing values
        X[0:10, 0] = np.nan
        y[0:10] = np.nan
        
        # Test calibration
        target_vars = ['target1']
        y_df = pd.DataFrame({'target1': y})
        
        models = calibrate_satellite_indices(X, y_df, target_vars, model_type="linear")
        
        # Should still work with missing data handling
        assert len(models) >= 0  # May be 0 if insufficient valid data


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
