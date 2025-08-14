"""
Tests for XGBoostFireRiskModel.

This module provides comprehensive tests for the XGBoost-based wildfire risk
prediction model, including unit tests and integration tests.
"""

import os
import tempfile
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.exceptions import ConvergenceWarning

from src.models.xgboost_model import XGBoostFireRiskModel


class TestXGBoostFireRiskModel:
    """Test suite for XGBoostFireRiskModel."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample regression data."""
        X, y = make_regression(
            n_samples=1000, n_features=15, n_informative=10,
            n_targets=1, random_state=42
        )
        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)]), y
    
    @pytest.fixture
    def sample_classification_data(self):
        """Generate sample classification data."""
        X, y = make_classification(
            n_samples=1000, n_features=15, n_informative=10,
            n_classes=2, random_state=42
        )
        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)]), y
    
    @pytest.fixture
    def model_regression(self):
        """Create a regression XGBoost model."""
        return XGBoostFireRiskModel(
            model_type='regression',
            n_estimators=50,
            max_depth=4,
            random_state=42
        )
    
    @pytest.fixture
    def model_classification(self):
        """Create a classification XGBoost model."""
        return XGBoostFireRiskModel(
            model_type='classification',
            n_estimators=50,
            max_depth=4,
            random_state=42
        )
    
    def test_initialization(self, model_regression, model_classification):
        """Test model initialization."""
        # Test regression model
        assert model_regression.model_type == 'regression'
        assert model_regression.n_estimators == 50
        assert model_regression.max_depth == 4
        assert model_regression.random_state == 42
        assert not model_regression.is_trained
        
        # Test classification model
        assert model_classification.model_type == 'classification'
        assert model_classification.n_estimators == 50
        assert model_classification.max_depth == 4
        assert model_classification.random_state == 42
        assert not model_classification.is_trained
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError, match="model_type must be 'regression' or 'classification'"):
            XGBoostFireRiskModel(model_type='invalid')
    
    def test_data_preparation(self, model_regression, sample_data):
        """Test data preparation functionality."""
        X, y = sample_data
        
        # Test data preparation
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(
            X, y, test_size=0.2, validation_size=0.2
        )
        
        # Check shapes
        assert X_train.shape[0] == 640  # 1000 * 0.8 * 0.8
        assert X_val.shape[0] == 160    # 1000 * 0.8 * 0.2
        assert X_test.shape[0] == 200   # 1000 * 0.2
        
        # Check that features are scaled
        assert not np.allclose(X_train, X_train.astype(int))
        assert not np.allclose(X_val, X_val.astype(int))
        assert not np.allclose(X_test, X_test.astype(int))
        
        # Check feature names
        assert model_regression.feature_names == [f'feature_{i}' for i in range(15)]
    
    def test_training_regression(self, model_regression, sample_data):
        """Test training for regression model."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
        result = model_regression.train(X_train, y_train, X_val, y_val)
        
        # Check training results
        assert model_regression.is_trained
        assert 'train_metrics' in result
        assert 'val_metrics' in result
        assert 'feature_importance' in result
        
        # Check metrics
        assert 'r2' in result['train_metrics']
        assert 'mse' in result['train_metrics']
        assert 'mae' in result['train_metrics']
        
        # Check that model can make predictions
        predictions = model_regression.predict(X_test[:10])
        assert len(predictions) == 10
        assert not np.any(np.isnan(predictions))
    
    def test_training_classification(self, model_classification, sample_classification_data):
        """Test training for classification model."""
        X, y = sample_classification_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_classification.prepare_data(X, y)
        
        # Train model
        result = model_classification.train(X_train, y_train, X_val, y_val)
        
        # Check training results
        assert model_classification.is_trained
        assert 'train_metrics' in result
        assert 'val_metrics' in result
        
        # Check metrics
        assert 'accuracy' in result['train_metrics']
        assert 'classification_report' in result['train_metrics']
        
        # Check that model can make predictions
        predictions = model_classification.predict(X_test[:10])
        assert len(predictions) == 10
        assert not np.any(np.isnan(predictions))
    
    def test_training_with_hyperparameter_tuning(self, model_regression, sample_data):
        """Test training with hyperparameter tuning."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train with hyperparameter tuning
        result = model_regression.train(
            X_train, y_train, X_val, y_val,
            hyperparameter_tuning=True,
            tuning_method='random',
            n_iter=10
        )
        
        # Check that best parameters were found
        assert len(model_regression.best_params) > 0
        assert model_regression.is_trained
        
        # Check that model can make predictions
        predictions = model_regression.predict(X_test[:10])
        assert len(predictions) == 10
    
    def test_prediction(self, model_regression, sample_data):
        """Test prediction functionality."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Test basic prediction
        predictions = model_regression.predict(X_test[:10])
        assert len(predictions) == 10
        assert not np.any(np.isnan(predictions))
        
        # Test prediction with confidence
        predictions, confidence = model_regression.predict(X_test[:10], return_confidence=True)
        assert len(predictions) == 10
        assert len(confidence) == 10
        assert not np.any(np.isnan(confidence))
    
    def test_prediction_with_intervals(self, model_regression, sample_data):
        """Test prediction with confidence intervals."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Test prediction with intervals
        predictions, lower_bounds, upper_bounds = model_regression.predict_with_intervals(
            X_test[:10], confidence_level=0.95
        )
        
        assert len(predictions) == 10
        assert len(lower_bounds) == 10
        assert len(upper_bounds) == 10
        
        # Check that bounds make sense
        assert np.all(lower_bounds <= predictions)
        assert np.all(predictions <= upper_bounds)
    
    def test_evaluation(self, model_regression, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        metrics = model_regression.evaluate(X_test, y_test)
        
        # Check metrics
        assert 'r2' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        
        # Check metric values are reasonable
        assert 0 <= metrics['r2'] <= 1
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
    
    def test_feature_importance(self, model_regression, sample_data):
        """Test feature importance calculation."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Get feature importance
        importance = model_regression._get_feature_importance()
        
        assert len(importance) == 15  # Number of features
        assert not np.any(np.isnan(importance))
        assert np.all(importance >= 0)
    
    def test_plot_feature_importance(self, model_regression, sample_data):
        """Test feature importance plotting."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Test plotting (should not raise error)
        try:
            model_regression.plot_feature_importance(top_n=10)
        except Exception as e:
            # If matplotlib backend is not available, that's okay
            if "No backend" not in str(e):
                raise
    
    def test_explain_prediction(self, model_regression, sample_data):
        """Test SHAP explanation functionality."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Test explanation
        explanation = model_regression.explain_prediction(X_test[:5], sample_idx=0, plot=False)
        
        # Check explanation structure
        assert 'shap_values' in explanation
        assert 'feature_names' in explanation
        assert 'prediction' in explanation
        assert 'feature_contributions' in explanation
        
        # Check values
        assert len(explanation['shap_values']) == 15
        assert len(explanation['feature_names']) == 15
        assert not np.isnan(explanation['prediction'])
    
    def test_model_persistence(self, model_regression, sample_data):
        """Test model saving and loading."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model_regression.save_model(tmp_file.name)
            
            # Create new model instance
            new_model = XGBoostFireRiskModel(model_type='regression')
            
            # Load model
            new_model.load_model(tmp_file.name)
            
            # Check that models are equivalent
            assert new_model.is_trained == model_regression.is_trained
            assert new_model.model_type == model_regression.model_type
            assert new_model.feature_names == model_regression.feature_names
            
            # Check predictions are the same
            old_pred = model_regression.predict(X_test[:10])
            new_pred = new_model.predict(X_test[:10])
            np.testing.assert_array_almost_equal(old_pred, new_pred)
            
            # Clean up
            try:
                # Add small delay for Windows file handling
                import time
                time.sleep(0.1)
                os.unlink(tmp_file.name)
            except (PermissionError, OSError):
                # Windows sometimes has file permission issues
                # This is a known issue and doesn't affect the actual test functionality
                pass
    
    def test_untrained_model_errors(self, model_regression, sample_data):
        """Test that untrained model raises appropriate errors."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Test prediction without training
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            model_regression.predict(X_test[:10])
        
        # Test evaluation without training
        with pytest.raises(ValueError, match="Model must be trained before evaluation"):
            model_regression.evaluate(X_test, y_test)
        
        # Test feature importance without training
        importance = model_regression._get_feature_importance()
        assert len(importance) == 0
    
    def test_invalid_hyperparameter_tuning_method(self, model_regression, sample_data):
        """Test that invalid tuning method raises error."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        with pytest.raises(ValueError, match="tuning_method must be 'grid' or 'random'"):
            model_regression.train(
                X_train, y_train, X_val, y_val,
                hyperparameter_tuning=True,
                tuning_method='invalid'
            )
    
    def test_early_stopping(self, model_regression, sample_data):
        """Test early stopping functionality."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Create model with early stopping
        model = XGBoostFireRiskModel(
            model_type='regression',
            n_estimators=1000,  # High number to trigger early stopping
            early_stopping_rounds=5,
            random_state=42
        )
        
        # Train model
        result = model.train(X_train, y_train, X_val, y_val)
        
        # Check that early stopping worked
        assert model.is_trained
        if 'best_iteration' in result and result['best_iteration'] is not None:
            assert result['best_iteration'] < 1000


class TestXGBoostFireRiskModelIntegration:
    """Integration tests for the XGBoost model."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Generate data
        X, y = make_regression(
            n_samples=500, n_features=10, n_informative=8,
            random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        
        # Create and train model
        model = XGBoostFireRiskModel(
            model_type='regression',
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X, y)
        
        # Train model
        result = model.train(X_train, y_train, X_val, y_val)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        
        # Check results
        assert model.is_trained
        assert len(predictions) == len(y_test)
        assert 'r2' in metrics
        assert metrics['r2'] > 0.5  # Should have reasonable performance
    
    def test_classification_workflow(self):
        """Test complete classification workflow."""
        # Generate data
        X, y = make_classification(
            n_samples=500, n_features=10, n_informative=8,
            n_classes=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        
        # Create and train model
        model = XGBoostFireRiskModel(
            model_type='classification',
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X, y)
        
        # Train model
        result = model.train(X_train, y_train, X_val, y_val)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        
        # Check results
        assert model.is_trained
        assert len(predictions) == len(y_test)
        assert 'accuracy' in metrics
        assert metrics['accuracy'] > 0.7  # Should have reasonable performance
    
    def test_model_persistence_integration(self):
        """Test model persistence in integration context."""
        # Generate data
        X, y = make_regression(
            n_samples=300, n_features=8, n_informative=6,
            random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
        
        # Create and train model
        model = XGBoostFireRiskModel(
            model_type='regression',
            n_estimators=50,
            max_depth=4,
            random_state=42
        )
        
        # Prepare data and train
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X, y)
        model.train(X_train, y_train, X_val, y_val)
        
        # Save and reload model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model.save_model(tmp_file.name)
            
            new_model = XGBoostFireRiskModel(model_type='regression')
            new_model.load_model(tmp_file.name)
            
            # Verify functionality
            old_pred = model.predict(X_test[:10])
            new_pred = new_model.predict(X_test[:10])
            
            np.testing.assert_array_almost_equal(old_pred, new_pred)
            
            # Clean up
            try:
                # Add small delay for Windows file handling
                import time
                time.sleep(0.1)
                os.unlink(tmp_file.name)
            except (PermissionError, OSError):
                # Windows sometimes has file permission issues
                # This is a known issue and doesn't affect the actual test functionality
                pass
    
    def test_hyperparameter_tuning_integration(self):
        """Test hyperparameter tuning in integration context."""
        # Generate data
        X, y = make_regression(
            n_samples=400, n_features=12, n_informative=10,
            random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(12)])
        
        # Create model
        model = XGBoostFireRiskModel(
            model_type='regression',
            n_estimators=50,
            max_depth=4,
            random_state=42
        )
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X, y)
        
        # Train with hyperparameter tuning
        result = model.train(
            X_train, y_train, X_val, y_val,
            hyperparameter_tuning=True,
            tuning_method='random',
            n_iter=20
        )
        
        # Check results
        assert model.is_trained
        assert len(model.best_params) > 0
        assert 'train_metrics' in result
        
        # Verify model works
        predictions = model.predict(X_test[:10])
        assert len(predictions) == 10
        assert not np.any(np.isnan(predictions))


if __name__ == '__main__':
    pytest.main([__file__])
