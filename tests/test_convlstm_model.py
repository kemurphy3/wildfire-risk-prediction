"""
Unit tests for the ConvLSTM model.

This module provides comprehensive testing for the ConvLSTMFireRiskModel
class, including data preparation, training, prediction, evaluation, and
model persistence functionality.

Test Coverage:
- Model initialization and configuration
- Data preparation and validation
- Training with and without validation data
- Prediction functionality (single, with confidence, with intervals)
- Model evaluation and metrics calculation
- Training history visualization
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
from src.models.convlstm_model import ConvLSTMFireRiskModel


class TestConvLSTMFireRiskModel:
    """Test suite for ConvLSTMFireRiskModel."""
    
    @pytest.fixture
    def model_regression(self):
        """Create a regression ConvLSTM model for testing."""
        return ConvLSTMFireRiskModel(
            model_type='regression',
            time_steps=5,
            spatial_dims=(16, 16),
            channels=8
        )
    
    @pytest.fixture
    def model_classification(self):
        """Create a classification ConvLSTM model for testing."""
        return ConvLSTMFireRiskModel(
            model_type='classification',
            time_steps=5,
            spatial_dims=(16, 16),
            channels=8
        )
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample spatiotemporal data for testing."""
        np.random.seed(42)
        n_samples = 100
        time_steps = 5
        height, width = 16, 16
        channels = 8
        
        # Create 4D spatiotemporal data: (samples, time_steps, height, width, channels)
        X = np.random.randn(n_samples, time_steps, height, width, channels)
        
        # Create target variable
        if np.random.choice([True, False]):
            # Regression target
            y = np.random.uniform(0, 100, n_samples)
        else:
            # Classification target
            y = np.random.randint(0, 3, n_samples)
        
        return X, y
    
    def test_initialization(self, model_regression, model_classification):
        """Test model initialization."""
        # Test regression model
        assert model_regression.model_type == 'regression'
        assert model_regression.time_steps == 5
        assert model_regression.spatial_dims == (16, 16)
        assert model_regression.channels == 8
        assert not model_regression.is_trained
        
        # Test classification model
        assert model_classification.model_type == 'classification'
        assert model_classification.time_steps == 5
        assert model_classification.spatial_dims == (16, 16)
        assert model_classification.channels == 8
        assert not model_classification.is_trained
    
    def test_data_preparation(self, model_regression, sample_data):
        """Test data preparation functionality."""
        X, y = sample_data
        
        # Test data preparation
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Check shapes
        assert X_train.shape[0] == 64  # 100 * 0.8 * 0.8 = 64
        assert X_val.shape[0] == 16   # 100 * 0.8 * 0.2 = 16
        assert X_test.shape[0] == 20  # 100 * 0.2 = 20
        
        # Check that data is 4D
        assert len(X_train.shape) == 4
        assert len(X_val.shape) == 4
        assert len(X_test.shape) == 4
        
        # Check that time_steps dimension is preserved
        assert X_train.shape[1] == 5
        assert X_val.shape[1] == 5
        assert X_test.shape[1] == 5
    
    def test_training_regression(self, model_regression, sample_data):
        """Test regression model training."""
        X, y = sample_data
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
        result = model_regression.train(X_train, y_train, X_val, y_val)
        
        # Check training result
        assert 'train_metrics' in result
        assert 'val_metrics' in result
        assert model_regression.is_trained
        
        # Check metrics
        train_metrics = result['train_metrics']
        assert 'mse' in train_metrics
        assert 'rmse' in train_metrics
        assert 'mae' in train_metrics
        assert 'r2' in train_metrics
    
    def test_training_classification(self, model_classification, sample_data):
        """Test classification model training."""
        X, y = sample_data
        
        # Ensure y is classification target
        if len(np.unique(y)) > 3:
            y = np.random.randint(0, 3, len(y))
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = model_classification.prepare_data(X, y)
        
        # Train model
        result = model_classification.train(X_train, y_train, X_val, y_val)
        
        # Check training result
        assert 'train_metrics' in result
        assert 'val_metrics' in result
        assert model_classification.is_trained
        
        # Check metrics
        train_metrics = result['train_metrics']
        assert 'accuracy' in train_metrics
        assert 'precision' in train_metrics
        assert 'recall' in train_metrics
        assert 'f1-score' in train_metrics
    
    def test_prediction(self, model_regression, sample_data):
        """Test prediction functionality."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
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
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
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
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Evaluate model
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
        """Test classification model evaluation."""
        X, y = sample_data
        
        # Ensure y is classification target
        if len(np.unique(y)) > 3:
            y = np.random.randint(0, 3, len(y))
        
        X_train, X_val, X_test, y_train, y_val, y_test = model_classification.prepare_data(X, y)
        
        # Train model
        model_classification.train(X_train, y_train, X_val, y_val)
        
        # Evaluate model
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
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Try to predict without training
        with pytest.raises(RuntimeError, match="Model must be trained before making predictions"):
            model_regression.predict(X_test[:5])
    
    def test_evaluation_before_training(self, model_regression, sample_data):
        """Test that evaluation fails before training."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Try to evaluate without training
        with pytest.raises(RuntimeError, match="Model must be trained before evaluation"):
            model_regression.evaluate(X_test, y_test)
    
    def test_model_persistence(self, model_regression, sample_data):
        """Test model save and load functionality."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            model_regression.save_model(tmp_file.name)
            
            # Create new model instance
            new_model = ConvLSTMFireRiskModel(
                model_type='regression',
                time_steps=5,
                spatial_dims=(16, 16),
                channels=8
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
    
    def test_invalid_data_preparation(self, model_regression):
        """Test data preparation with invalid inputs."""
        # Test with wrong data shape
        wrong_shape_data = np.random.randn(100, 3, 16, 16)  # Missing time dimension
        
        with pytest.raises(ValueError, match="Input data must be 5D"):
            model_regression.prepare_data(wrong_shape_data, np.random.randn(100))
    
    def test_training_history_plotting(self, model_regression, sample_data):
        """Test training history visualization."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Test plotting (should not raise error)
        try:
            model_regression.plot_training_history()
        except Exception as e:
            # If matplotlib backend is not available, that's okay
            if "No backend" not in str(e):
                raise
    
    def test_model_attributes_after_training(self, model_regression, sample_data):
        """Test that model attributes are properly set after training."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Check that all attributes are properly set
        assert model_regression.is_trained == True
        assert model_regression.feature_names is not None
        assert model_regression.training_history is not None
        
        # Check that training history contains expected keys
        if model_regression.training_history:
            assert 'loss' in model_regression.training_history
            assert 'val_loss' in model_regression.training_history


# Integration tests
class TestConvLSTMFireRiskModelIntegration:
    """Integration tests for the ConvLSTM model."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create model
        model = ConvLSTMFireRiskModel(
            model_type='regression',
            time_steps=5,
            spatial_dims=(16, 16),
            channels=8
        )
        
        # Generate data
        np.random.seed(42)
        n_samples = 200
        time_steps = 5
        height, width = 16, 16
        channels = 8
        
        X = np.random.randn(n_samples, time_steps, height, width, channels)
        y = np.random.uniform(0, 100, n_samples)
        
        # Complete workflow
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X, y)
        training_result = model.train(X_train, y_train, X_val, y_val)
        test_metrics = model.evaluate(X_test, y_test)
        predictions = model.predict(X_test[:10])
        
        # Verify results
        assert len(predictions) == 10
        assert 'mse' in test_metrics
        assert 'r2' in test_metrics
        assert model.is_trained
    
    def test_classification_workflow(self):
        """Test classification workflow."""
        # Create model
        model = ConvLSTMFireRiskModel(
            model_type='classification',
            time_steps=5,
            spatial_dims=(16, 16),
            channels=8
        )
        
        # Generate data
        np.random.seed(42)
        n_samples = 200
        time_steps = 5
        height, width = 16, 16
        channels = 8
        
        X = np.random.randn(n_samples, time_steps, height, width, channels)
        y = np.random.randint(0, 3, n_samples)
        
        # Complete workflow
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X, y)
        training_result = model.train(X_train, y_train, X_val, y_val)
        test_metrics = model.evaluate(X_test, y_test)
        predictions = model.predict(X_test[:10])
        
        # Verify results
        assert len(predictions) == 10
        assert 'accuracy' in test_metrics
        assert 'precision' in test_metrics
        assert model.is_trained
    
    def test_model_persistence_integration(self):
        """Test model persistence in integration workflow."""
        # Create model
        model = ConvLSTMFireRiskModel(
            model_type='regression',
            time_steps=5,
            spatial_dims=(16, 16),
            channels=8
        )
        
        # Generate data
        np.random.seed(42)
        n_samples = 100
        time_steps = 5
        height, width = 16, 16
        channels = 8
        
        X = np.random.randn(n_samples, time_steps, height, width, channels)
        y = np.random.uniform(0, 100, n_samples)
        
        # Train and save
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X, y)
        model.train(X_train, y_train, X_val, y_val)
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            model.save_model(tmp_file.name)
            
            # Load and verify
            new_model = ConvLSTMFireRiskModel(
                model_type='regression',
                time_steps=5,
                spatial_dims=(16, 16),
                channels=8
            )
            new_model.load_model(tmp_file.name)
            
            # Test predictions match
            predictions_original = model.predict(X_test[:5])
            predictions_loaded = new_model.predict(X_test[:5])
            np.testing.assert_array_almost_equal(predictions_original, predictions_loaded)
            
            # Clean up
            try:
                import time
                time.sleep(0.1)
                os.unlink(tmp_file.name)
            except (PermissionError, OSError):
                pass
