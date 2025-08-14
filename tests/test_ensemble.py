"""
Unit tests for the ensemble model.

This module provides comprehensive testing for the EnsembleFireRiskModel
class, including data preparation, training, prediction, evaluation, and
model persistence functionality.

Test Coverage:
- Model initialization and configuration
- Data preparation and validation
- Training with different ensemble methods
- Prediction functionality (single, with confidence, with intervals)
- Model evaluation and metrics calculation
- Base model comparison and visualization
- Model persistence (save/load)
- Error handling and edge cases
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os
from datetime import datetime

# Import the model to test
from src.models.ensemble import EnsembleFireRiskModel
from src.models.baseline_model import RandomForestFireRiskModel
from src.models.xgboost_model import XGBoostFireRiskModel


class TestEnsembleFireRiskModel:
    """Test suite for EnsembleFireRiskModel."""
    
    @pytest.fixture
    def model_regression(self):
        """Create a regression ensemble model for testing."""
        return EnsembleFireRiskModel(
            model_type='regression',
            ensemble_method='voting'
        )
    
    @pytest.fixture
    def model_classification(self):
        """Create a classification ensemble model for testing."""
        return EnsembleFireRiskModel(
            model_type='classification',
            ensemble_method='voting'
        )
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 15
        
        # Create features
        X = np.random.randn(n_samples, n_features)
        
        # Create target variable
        if np.random.choice([True, False]):
            # Regression target
            y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1
        else:
            # Classification target
            y = np.random.randint(0, 3, n_samples)
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        return X_df, y
    
    def test_initialization(self, model_regression, model_classification):
        """Test model initialization."""
        # Test regression model
        assert model_regression.model_type == 'regression'
        assert model_regression.ensemble_method == 'voting'
        assert len(model_regression.base_models) > 0
        assert not model_regression.is_trained
        
        # Test classification model
        assert model_classification.model_type == 'classification'
        assert model_classification.ensemble_method == 'voting'
        assert len(model_classification.base_models) > 0
        assert not model_classification.is_trained
    
    def test_initialization_with_custom_models(self):
        """Test initialization with custom base models."""
        # Create custom base models
        rf_model = RandomForestFireRiskModel(model_type='regression')
        xgb_model = XGBoostFireRiskModel(model_type='regression')
        
        ensemble = EnsembleFireRiskModel(
            model_type='regression',
            ensemble_method='stacking',
            base_models=[rf_model, xgb_model]
        )
        
        assert len(ensemble.base_models) == 2
        assert ensemble.ensemble_method == 'stacking'
    
    def test_data_preparation(self, model_regression, sample_data):
        """Test data preparation functionality."""
        X_df, y = sample_data
        
        # Test data preparation
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Check shapes
        assert X_train.shape[0] == 640  # 1000 * 0.8 * 0.8 = 640
        assert X_val.shape[0] == 160   # 1000 * 0.8 * 0.2 = 160
        assert X_test.shape[0] == 200  # 1000 * 0.2 = 200
        
        # Check that feature names are stored
        assert model_regression.feature_names is not None
        assert len(model_regression.feature_names) == 15
    
    def test_training_voting(self, model_regression, sample_data):
        """Test voting ensemble training."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train ensemble
        result = model_regression.train(X_train, y_train, X_val, y_val)
        
        # Check training result
        assert 'train_metrics' in result
        assert 'val_metrics' in result
        assert model_regression.is_trained
        
        # Check that base models are trained
        for model in model_regression.base_models:
            assert model.is_trained
    
    def test_training_stacking(self, sample_data):
        """Test stacking ensemble training."""
        ensemble = EnsembleFireRiskModel(
            model_type='regression',
            ensemble_method='stacking'
        )
        
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = ensemble.prepare_data(X_df, y)
        
        # Train ensemble
        result = ensemble.train(X_train, y_train, X_val, y_val)
        
        # Check training result
        assert 'train_metrics' in result
        assert 'val_metrics' in result
        assert ensemble.is_trained
        
        # Check that meta-learner is trained
        assert hasattr(ensemble, 'meta_learner')
    
    def test_training_weighted(self, sample_data):
        """Test weighted ensemble training."""
        ensemble = EnsembleFireRiskModel(
            model_type='regression',
            ensemble_method='weighted'
        )
        
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = ensemble.prepare_data(X_df, y)
        
        # Train ensemble
        result = ensemble.train(X_train, y_train, X_val, y_val)
        
        # Check training result
        assert 'train_metrics' in result
        assert 'val_metrics' in result
        assert ensemble.is_trained
        
        # Check that weights are optimized
        assert hasattr(ensemble, 'model_weights')
        assert len(ensemble.model_weights) == len(ensemble.base_models)
    
    def test_prediction(self, model_regression, sample_data):
        """Test prediction functionality."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train ensemble
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Test prediction
        predictions = model_regression.predict(X_test[:5])
        assert len(predictions) == 5
        
        # Test prediction with confidence
        predictions, confidence = model_regression.predict(X_test[:5], return_confidence=True)
        assert len(predictions) == 5
        assert len(confidence) == 5
        assert all(0 <= c <= 1 for c in confidence)
    
    def test_prediction_with_intervals(self, model_regression, sample_data):
        """Test prediction with confidence intervals."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train ensemble
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Test prediction with intervals
        predictions, lower, upper = model_regression.predict_with_intervals(X_test[:5])
        assert len(predictions) == 5
        assert len(lower) == 5
        assert len(upper) == 5
        
        # Check that intervals make sense
        for i in range(5):
            assert lower[i] <= predictions[i] <= upper[i]
    
    def test_evaluation(self, model_regression, sample_data):
        """Test model evaluation."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train ensemble
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Evaluate ensemble
        metrics = model_regression.evaluate(X_test, y_test)
        
        # Check metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        # Check that metrics are reasonable
        assert metrics['r2'] >= -1 and metrics['r2'] <= 1
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
    
    def test_evaluation_classification(self, model_classification, sample_data):
        """Test classification ensemble evaluation."""
        X_df, y = sample_data
        
        # Ensure y is classification target
        if len(np.unique(y)) > 3:
            y = np.random.randint(0, 3, len(y))
        
        X_train, X_val, X_test, y_train, y_val, y_test = model_classification.prepare_data(X_df, y)
        
        # Train ensemble
        model_classification.train(X_train, y_train, X_val, y_val)
        
        # Evaluate ensemble
        metrics = model_classification.evaluate(X_test, y_test)
        
        # Check classification metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1-score' in metrics
        
        # Check that metrics are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1-score'] <= 1
    
    def test_prediction_before_training(self, model_regression, sample_data):
        """Test that prediction fails before training."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Try to predict without training
        with pytest.raises(RuntimeError, match="Model must be trained before making predictions"):
            model_regression.predict(X_test[:5])
    
    def test_evaluation_before_training(self, model_regression, sample_data):
        """Test that evaluation fails before training."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Try to evaluate without training
        with pytest.raises(RuntimeError, match="Model must be trained before evaluation"):
            model_regression.evaluate(X_test, y_test)
    
    def test_model_persistence(self, model_regression, sample_data):
        """Test model save and load functionality."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train ensemble
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model_regression.save_model(tmp_file.name)
            
            # Create new model instance
            new_model = EnsembleFireRiskModel(
                model_type='regression',
                ensemble_method='voting'
            )
            
            # Load model
            new_model.load_model(tmp_file.name)
            
            # Test that loaded model works
            predictions_original = model_regression.predict(X_test[:5])
            predictions_loaded = new_model.predict(X_test[:5])
            
            np.testing.assert_array_almost_equal(predictions_original, predictions_loaded)
            
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
    
    def test_invalid_ensemble_method(self):
        """Test initialization with invalid ensemble method."""
        with pytest.raises(ValueError, match="ensemble_method must be"):
            EnsembleFireRiskModel(
                model_type='regression',
                ensemble_method='invalid_method'
            )
    
    def test_base_model_comparison(self, model_regression, sample_data):
        """Test base model comparison functionality."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train ensemble
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Test base model comparison
        try:
            model_regression.plot_base_model_comparison(X_test, y_test)
        except Exception as e:
            # If matplotlib backend is not available, that's okay
            if "No backend" not in str(e):
                raise
    
    def test_model_attributes_after_training(self, model_regression, sample_data):
        """Test that model attributes are properly set after training."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train ensemble
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Check that all attributes are properly set
        assert model_regression.is_trained == True
        assert model_regression.feature_names is not None
        
        # Check that base models are trained
        for model in model_regression.base_models:
            assert model.is_trained


# Integration tests
class TestEnsembleFireRiskModelIntegration:
    """Integration tests for the ensemble model."""
    
    def test_end_to_end_workflow_voting(self):
        """Test complete end-to-end workflow with voting ensemble."""
        # Create ensemble
        ensemble = EnsembleFireRiskModel(
            model_type='regression',
            ensemble_method='voting'
        )
        
        # Generate data
        np.random.seed(42)
        n_samples = 500
        n_features = 15
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] * 2.0 + X[:, 1] * 1.5 + X[:, 2] * 0.8 + 
             np.random.randn(n_samples) * 0.1)
        y = np.clip((y - y.min()) / (y.max() - y.min()) * 100, 0, 100)
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Complete workflow
        X_train, X_val, X_test, y_train, y_val, y_test = ensemble.prepare_data(X_df, y)
        training_result = ensemble.train(X_train, y_train, X_val, y_val)
        test_metrics = ensemble.evaluate(X_test, y_test)
        predictions = ensemble.predict(X_test[:10])
        
        # Verify results
        assert len(predictions) == 10
        assert 'mse' in test_metrics
        assert 'r2' in test_metrics
        assert ensemble.is_trained
    
    def test_end_to_end_workflow_stacking(self):
        """Test complete end-to-end workflow with stacking ensemble."""
        # Create ensemble
        ensemble = EnsembleFireRiskModel(
            model_type='regression',
            ensemble_method='stacking'
        )
        
        # Generate data
        np.random.seed(42)
        n_samples = 500
        n_features = 15
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] * 2.0 + X[:, 1] * 1.5 + X[:, 2] * 0.8 + 
             np.random.randn(n_samples) * 0.1)
        y = np.clip((y - y.min()) / (y.max() - y.min()) * 100, 0, 100)
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Complete workflow
        X_train, X_val, X_test, y_train, y_val, y_test = ensemble.prepare_data(X_df, y)
        training_result = ensemble.train(X_train, y_train, X_val, y_val)
        test_metrics = ensemble.evaluate(X_test, y_test)
        predictions = ensemble.predict(X_test[:10])
        
        # Verify results
        assert len(predictions) == 10
        assert 'mse' in test_metrics
        assert 'r2' in test_metrics
        assert ensemble.is_trained
        assert hasattr(ensemble, 'meta_learner')
    
    def test_end_to_end_workflow_weighted(self):
        """Test complete end-to-end workflow with weighted ensemble."""
        # Create ensemble
        ensemble = EnsembleFireRiskModel(
            model_type='regression',
            ensemble_method='weighted'
        )
        
        # Generate data
        np.random.seed(42)
        n_samples = 500
        n_features = 15
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] * 2.0 + X[:, 1] * 1.5 + X[:, 2] * 0.8 + 
             np.random.randn(n_samples) * 0.1)
        y = np.clip((y - y.min()) / (y.max() - y.min()) * 100, 0, 100)
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Complete workflow
        X_train, X_val, X_test, y_train, y_val, y_test = ensemble.prepare_data(X_df, y)
        training_result = ensemble.train(X_train, y_train, X_val, y_val)
        test_metrics = ensemble.evaluate(X_test, y_test)
        predictions = ensemble.predict(X_test[:10])
        
        # Verify results
        assert len(predictions) == 10
        assert 'mse' in test_metrics
        assert 'r2' in test_metrics
        assert ensemble.is_trained
        assert hasattr(ensemble, 'model_weights')
    
    def test_model_persistence_integration(self):
        """Test model persistence in integration workflow."""
        # Create ensemble
        ensemble = EnsembleFireRiskModel(
            model_type='regression',
            ensemble_method='voting'
        )
        
        # Generate data
        np.random.seed(42)
        n_samples = 300
        n_features = 15
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] * 2.0 + X[:, 1] * 1.5 + X[:, 2] * 0.8 + 
             np.random.randn(n_samples) * 0.1)
        y = np.clip((y - y.min()) / (y.max() - y.min()) * 100, 0, 100)
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Train and save
        X_train, X_val, X_test, y_train, y_val, y_test = ensemble.prepare_data(X_df, y)
        ensemble.train(X_train, y_train, X_val, y_val)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            ensemble.save_model(tmp_file.name)
            
            # Load and verify
            new_ensemble = EnsembleFireRiskModel(
                model_type='regression',
                ensemble_method='voting'
            )
            new_ensemble.load_model(tmp_file.name)
            
            # Test predictions match
            predictions_original = ensemble.predict(X_test[:5])
            predictions_loaded = new_ensemble.predict(X_test[:5])
            np.testing.assert_array_almost_equal(predictions_original, predictions_loaded)
            
            # Clean up
            try:
                import time
                time.sleep(0.1)
                os.unlink(tmp_file.name)
            except (PermissionError, OSError):
                pass
