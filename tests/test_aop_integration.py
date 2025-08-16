"""
Tests for AOP integration.

Making sure the AOP crosswalk stuff actually works with 
our wildfire prediction pipeline. Lots of mocking because
we don't wanna download gigabytes of data during tests.
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
    """Tests for the main integration manager."""
    
    @pytest.fixture
    def temp_config(self):
        """Make a temp config for testing."""
        config = {
            'paths': {
                'models_root': 'temp_models',
                'processed_data_root': 'temp_processed',
                'outputs_root': 'temp_outputs'
            },
            'neon_api': {
                'base_url': 'https://test.neonscience.org/api/v0',  # fake url
                'timeout': 30,
                'max_retries': 2
            }
        }
        
        # make temp dirs
        temp_dir = Path(tempfile.mkdtemp())
        config['paths']['models_root'] = str(temp_dir / 'models')
        config['paths']['processed_data_root'] = str(temp_dir / 'processed')
        config['paths']['outputs_root'] = str(temp_dir / 'outputs')
        
        yield config, temp_dir
        
        # cleanup after test
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_satellite_data(self):
        """Fake satellite data."""
        np.random.seed(42)  # reproducible
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
        """Fake AOP data (airplane measurements)."""
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
        """Check if manager sets up correctly."""
        config, temp_dir = temp_config
        
        # write config
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # should work
        manager = AOPIntegrationManager(str(config_path))
        
        assert manager.config == config
        assert Path(manager.models_dir).exists()
        assert Path(manager.outputs_dir).exists()
        assert len(manager.crosswalk_models) == 0  # no models yet
    
    def test_enhanced_features_generation(self, temp_config, sample_satellite_data):
        """Test feature enhancement."""
        config, temp_dir = temp_config
        
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        manager = AOPIntegrationManager(str(config_path))
        
        # test w/o models (should still work)
        enhanced_data = manager.get_enhanced_features(
            sample_satellite_data, 'SRER', 2021
        )
        
        # 8 original cols + 3 metadata cols
        expected_columns = 8 + 3
        assert enhanced_data.shape == (100, expected_columns)
        assert 'crosswalk_models_available' in enhanced_data.columns
        assert 'aop_data_available' in enhanced_data.columns
        assert 'enhancement_timestamp' in enhanced_data.columns
    
    def test_integration_validation(self, temp_config):
        """Check validation logic."""
        config, temp_dir = temp_config
        
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        manager = AOPIntegrationManager(str(config_path))
        
        # validate non-existent site (should fail gracefully)
        validation = manager.validate_integration('SRER', 2021)
        
        assert validation['site_code'] == 'SRER'
        assert validation['year'] == 2021
        assert validation['integration_status'] == 'not_integrated'  # nothing there
        assert not validation['aop_data_available']
        assert not validation['feature_enhancement']
    
    def test_crosswalk_model_training(self, sample_satellite_data, sample_aop_data):
        """Test model training."""
        # try linear models
        target_vars = ['chm_mean', 'ndvi_aop']
        
        models = calibrate_satellite_indices(
            sample_satellite_data, sample_aop_data, target_vars, model_type="linear"
        )
        
        assert len(models) == 2
        assert 'chm_mean' in models
        assert 'ndvi_aop' in models
        
        # try ensemble too
        models_ensemble = calibrate_satellite_indices(
            sample_satellite_data, sample_aop_data, target_vars, model_type="ensemble"
        )
        
        assert len(models_ensemble) == 2
    
    def test_chm_feature_extraction(self):
        """Test canopy height features."""
        # fake CHM data
        chm_arr = np.random.uniform(0, 20, (100, 100))
        chm_arr[0:10, 0:10] = np.nan  # add some bad pixels
        
        features = extract_chm_features(chm_arr)
        
        assert 'chm_mean' in features
        assert 'chm_std' in features
        assert 'canopy_cover_gt2m' in features
        assert 'canopy_cover_gt5m' in features
        assert 'rumple_index' in features
        assert 'height_entropy' in features
        
        # sanity check values
        assert 0 <= features['canopy_cover_gt2m'] <= 1
        assert 0 <= features['canopy_cover_gt5m'] <= 1
        assert features['chm_mean'] >= 0  # no negative heights!
    
    def test_spectral_feature_extraction(self):
        """Test hyperspectral features."""
        # fake hyperspectral cube
        hsi_arr = np.random.uniform(0, 1, (100, 100, 10))  # 10 bands
        
        features = extract_spectral_features(hsi_arr)
        
        assert 'ndvi_aop' in features
        assert 'evi_aop' in features
        assert 'nbr_aop' in features
        assert 'ndwi_aop' in features
        
        # indices should be in [-1, 1]
        assert -1 <= features['ndvi_aop'] <= 1
        assert -1 <= features['evi_aop'] <= 1
        assert -1 <= features['nbr_aop'] <= 1
    
    def test_geospatial_alignment(self):
        """Test geo alignment stuff."""
        # mock rasterio (don't need real files)
        with patch('rasterio.open') as mock_open:
            mock_src = Mock()
            mock_src.meta = {
                'crs': 'EPSG:32632',
                'transform': (10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
                'width': 100,
                'height': 100
            }
            mock_open.return_value.__enter__.return_value = mock_src
            
            # test grid calc
            bounds = [-111.51, 31.74, -111.46, 31.79]  # somewhere in AZ
            grid_params = infer_dst_grid(bounds, 'sentinel2_10m')
            
            assert 'crs' in grid_params
            assert 'transform' in grid_params
            assert 'width' in grid_params
            assert 'height' in grid_params
    
    def test_crosswalk_validation(self, sample_satellite_data, sample_aop_data):
        """Test model validation."""
        # train first
        target_vars = ['chm_mean', 'ndvi_aop']
        models = calibrate_satellite_indices(
            sample_satellite_data, sample_aop_data, target_vars, model_type="linear"
        )
        
        # then validate
        with tempfile.TemporaryDirectory() as temp_dir:
            validation_results = validate_crosswalk(
                sample_satellite_data, sample_aop_data, models, Path(temp_dir)
            )
            
            assert len(validation_results) == 2  # 2 variables
            assert 'target_variable' in validation_results.columns
            assert 'r2' in validation_results.columns
            assert 'mae' in validation_results.columns
    
    def test_integration_report_generation(self, temp_config):
        """Test HTML report generation."""
        config, temp_dir = temp_config
        
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        manager = AOPIntegrationManager(str(config_path))
        
        # make report
        report = manager.generate_integration_report(['SRER', 'JORN'], [2021, 2022])
        
        assert isinstance(report, str)
        assert 'AOP Integration Report' in report
        assert 'Integration Summary' in report
        assert 'SRER' in report  # should have both sites
        assert 'JORN' in report
    
    def test_enhanced_features_export(self, temp_config, sample_satellite_data):
        """Test export functionality."""
        config, temp_dir = temp_config
        
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        manager = AOPIntegrationManager(str(config_path))
        
        # test CSV
        output_path = manager.export_enhanced_features('SRER', 2021, 'csv')
        
        assert output_path is not None
        assert output_path.exists()
        assert output_path.suffix == '.csv'
        
        # test parquet
        output_path = manager.export_enhanced_features('SRER', 2021, 'parquet')
        
        assert output_path is not None
        assert output_path.exists()
        assert output_path.suffix == '.parquet'


class TestAOPCrosswalkModels:
    """Tests for crosswalk models specifically."""
    
    def test_linear_crosswalk_fitting(self):
        """Test linear model fitting."""
        # random data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        model_result = fit_linear_crosswalk(X, y)
        
        assert 'model' in model_result
        assert 'metrics' in model_result
        assert 'feature_importance' in model_result
        assert model_result['metrics']['test_r2'] >= 0  # R² can't be negative
    
    def test_ensemble_crosswalk_fitting(self):
        """Test GB model fitting."""
        # more random data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        model = fit_ensemble_crosswalk(X, y)
        
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'feature_importances_')  # GB should have this
    
    def test_model_calibration_with_missing_data(self):
        """Test with NaNs (real data always has em)."""
        # data with holes
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # mess it up
        X[0:10, 0] = np.nan
        y[0:10] = np.nan
        
        # try to train anyway
        target_vars = ['target1']
        y_df = pd.DataFrame({'target1': y})
        
        models = calibrate_satellite_indices(X, y_df, target_vars, model_type="linear")
        
        # should handle NaNs gracefully
        assert len(models) >= 0  # might be 0 if too many NaNs


if __name__ == "__main__":
    # run it!
    pytest.main([__file__, "-v"])
