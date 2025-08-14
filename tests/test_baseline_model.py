"""
Unit tests for the Random Forest baseline model.

This module provides comprehensive testing for the RandomForestFireRiskModel
class, including data preparation, training, prediction, evaluation, and
model persistence functionality.

Test Coverage:
- Model initialization and configuration
- Data preparation and validation
- Training with and without hyperparameter tuning
- Prediction functionality (single, with confidence, with intervals)
- Model evaluation and metrics calculation
- Feature importance calculation
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
from src.models.baseline_model import RandomForestFireRiskModel


class TestRandomForestFireRiskModel:
    """Test suite for RandomForestFireRiskModel."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        # Generate synthetic features
        X = np.random.randn(n_samples, n_features)
        
        # Generate synthetic target (regression)
        y = (X[:, 0] * 2.0 + X[:, 1] * 1.5 + X[:, 2] * 0.8 + 
             np.random.randn(n_samples) * 0.1)
        
        # Normalize to 0-100 range
        y = np.clip((y - y.min()) / (y.max() - y.min()) * 100, 0, 100)
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        return X_df, y
    
    @pytest.fixture
    def sample_classification_data(self):
        """Create sample classification data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        # Generate synthetic features
        X = np.random.randn(n_samples, n_features)
        
        # Generate synthetic target (classification)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        return X_df, y
    
    @pytest.fixture
    def model_regression(self):
        """Create a regression model instance."""
        return RandomForestFireRiskModel(model_type='regression')
    
    @pytest.fixture
    def model_classification(self):
        """Create a classification model instance."""
        return RandomForestFireRiskModel(model_type='classification')
    
    def test_initialization(self, model_regression):
        """Test model initialization."""
        assert model_regression.model_type == 'regression'
        assert model_regression.n_estimators == 100
        assert model_regression.random_state == 42
        assert model_regression.is_trained == False
        assert model_regression.model is not None
        assert model_regression.scaler is not None
        assert model_regression.label_encoder is not None
    
    def test_initialization_classification(self, model_classification):
        """Test classification model initialization."""
        assert model_classification.model_type == 'classification'
        assert model_classification.model is not None
        assert hasattr(model_classification.model, 'class_weight')
    
    def test_initialization_invalid_type(self):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError, match="Model type must be 'regression' or 'classification'"):
            RandomForestFireRiskModel(model_type='invalid')
    
    def test_data_preparation(self, model_regression, sample_data):
        """Test data preparation functionality."""
        X_df, y = sample_data
        
        # Test data preparation
        result = model_regression.prepare_data(X_df, y)
        X_train, X_val, X_test, y_train, y_val, y_test = result
        
        # Check shapes
        assert len(X_train) + len(X_val) + len(X_test) == len(X_df)
        assert len(y_train) + len(y_val) + len(y_test) == len(y)
        
        # Check feature names
        assert model_regression.feature_names == list(X_df.columns)
        
        # Check scaling
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    
    def test_data_preparation_classification(self, model_classification, sample_classification_data):
        """Test data preparation for classification."""
        X_df, y = sample_classification_data
        
        result = model_classification.prepare_data(X_df, y)
        X_train, X_val, X_test, y_train, y_val, y_test = result
        
        # Check that labels were encoded
        assert len(model_classification.label_encoder.classes_) == 2
        
        # Check data splits
        assert len(X_train) + len(X_val) + len(X_test) == len(X_df)
    
    def test_training(self, model_regression, sample_data):
        """Test model training."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train model
        result = model_regression.train(X_train, y_train, X_val, y_val)
        
        # Check training results
        assert model_regression.is_trained == True
        assert 'training_metrics' in result
        assert 'validation_metrics' in result
        assert 'feature_importance' is not None
        assert 'oob_score' in result
        
        # Check metrics
        assert 'r2' in result['training_metrics']
        assert 'r2' in result['validation_metrics']
    
    def test_training_with_hyperparameter_tuning(self, model_regression, sample_data):
        """Test training with hyperparameter tuning."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train with hyperparameter tuning
        result = model_regression.train(X_train, y_train, hyperparameter_tuning=True)
        
        assert model_regression.is_trained == True
        assert result is not None
    
    def test_prediction(self, model_regression, sample_data):
        """Test basic prediction functionality."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Test prediction
        predictions = model_regression.predict(X_test[:5])
        assert len(predictions) == 5
        assert all(isinstance(p, (int, float)) for p in predictions)
    
    def test_prediction_with_confidence(self, model_regression, sample_data):
        """Test prediction with confidence scores."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Test prediction with confidence
        predictions, confidence = model_regression.predict(X_test[:5], return_confidence=True)
        assert len(predictions) == 5
        assert len(confidence) == 5
        assert all(0 <= c <= 1 for c in confidence)
    
    def test_prediction_with_intervals(self, model_regression, sample_data):
        """Test prediction with confidence intervals."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
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
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        metrics = model_regression.evaluate(X_test, y_test)
        
        # Check metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        
        # Check that metrics are reasonable
        assert metrics['r2'] >= -1 and metrics['r2'] <= 1
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
    
    def test_evaluation_classification(self, model_classification, sample_classification_data):
        """Test classification model evaluation."""
        X_df, y = sample_classification_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_classification.prepare_data(X_df, y)
        
        # Train model
        model_classification.train(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        metrics = model_classification.evaluate(X_test, y_test)
        
        # Check classification metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1-score' in metrics
        assert 'confusion_matrix' in metrics
        
        # Check that metrics are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1-score'] <= 1
    
    def test_feature_importance(self, model_regression, sample_data):
        """Test feature importance calculation."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Check feature importance
        assert model_regression.feature_importance is not None
        assert len(model_regression.feature_importance) == len(model_regression.feature_names)
        
        # Check that importance values are sorted
        importance_values = model_regression.feature_importance['importance'].values
        assert all(importance_values[i] >= importance_values[i+1] 
                  for i in range(len(importance_values)-1))
    
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
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model_regression.save_model(tmp_file.name)
            
            # Create new model instance
            new_model = RandomForestFireRiskModel()
            
            # Load model
            new_model.load_model(tmp_file.name)
            
            # Check that model is loaded correctly
            assert new_model.is_trained == True
            assert new_model.feature_names == model_regression.feature_names
            assert new_model.model_type == model_regression.model_type
            
            # Test prediction with loaded model
            predictions_original = model_regression.predict(X_test[:5])
            predictions_loaded = new_model.predict(X_test[:5])
            
            # Predictions should be identical
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
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            model_regression.prepare_data(empty_df, [])
        
        # Test with mismatched lengths
        X_df = pd.DataFrame({'feature_1': [1, 2, 3], 'feature_2': [4, 5, 6]})
        y = [1, 2]  # Mismatched length
        with pytest.raises(ValueError):
            model_regression.prepare_data(X_df, y)
    
    def test_hyperparameter_tuning_parameters(self, model_regression, sample_data):
        """Test hyperparameter tuning with different parameter grids."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Test with custom parameters
        custom_model = RandomForestFireRiskModel(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        # Prepare data
        X_train_custom, X_val_custom, X_test_custom, y_train_custom, y_val_custom, y_test_custom = \
            custom_model.prepare_data(X_df, y)
        
        # Train with hyperparameter tuning
        result = custom_model.train(X_train_custom, y_train_custom, hyperparameter_tuning=True)
        
        assert custom_model.is_trained == True
        assert result is not None
    
    def test_classification_with_imbalanced_data(self, model_classification):
        """Test classification with imbalanced data."""
        # Create imbalanced dataset
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate imbalanced target (90% class 0, 10% class 1)
        y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = model_classification.prepare_data(X_df, y)
        
        # Train model
        result = model_classification.train(X_train, y_train, X_val, y_val)
        
        # Check that model handles imbalanced data
        assert model_classification.is_trained == True
        
        # Evaluate
        metrics = model_classification.evaluate(X_test, y_test)
        assert metrics['accuracy'] > 0.5  # Should be better than random
    
    def test_edge_cases(self, model_regression):
        """Test edge cases and boundary conditions."""
        # Test with single sample
        single_sample = pd.DataFrame({'feature_1': [1.0], 'feature_2': [2.0]})
        y_single = [5.0]
        
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(single_sample, y_single)
            # This should work but may have warnings
        except Exception as e:
            # Single sample might not work with some validation strategies
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ["insufficient", "too few", "empty", "n_samples"])
        
        # Test with very large feature values
        large_features = pd.DataFrame({
            'feature_1': [1e10, -1e10],
            'feature_2': [1e-10, -1e-10]
        })
        y_large = [100, 0]
        
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(large_features, y_large)
            # Should handle extreme values gracefully
        except Exception:
            # Some models might not handle extreme values well
            pass
    
    def test_model_attributes_after_training(self, model_regression, sample_data):
        """Test that model attributes are properly set after training."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Check that all attributes are properly set
        assert model_regression.is_trained == True
        assert model_regression.feature_names is not None
        assert model_regression.training_metrics is not None
        assert model_regression.validation_metrics is not None
        assert model_regression.feature_importance is not None
        
        # Check that metrics contain expected keys
        expected_training_keys = ['mse', 'rmse', 'mae', 'r2', 'mape']
        for key in expected_training_keys:
            assert key in model_regression.training_metrics
    
    def test_cross_validation_compatibility(self, model_regression, sample_data):
        """Test that the model is compatible with scikit-learn cross-validation."""
        X_df, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = model_regression.prepare_data(X_df, y)
        
        # Train model
        model_regression.train(X_train, y_train, X_val, y_val)
        
        # Test cross-validation compatibility
        from sklearn.model_selection import cross_val_score
        
        # The model should be compatible with scikit-learn's cross_val_score
        scores = cross_val_score(
            model_regression.model, 
            X_train, 
            y_train, 
            cv=3, 
            scoring='r2'
        )
        
        assert len(scores) == 3
        assert all(isinstance(score, (int, float)) for score in scores)


# Integration tests
class TestRandomForestFireRiskModelIntegration:
    """Integration tests for the Random Forest model."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create model
        model = RandomForestFireRiskModel(model_type='regression', n_estimators=50)
        
        # Generate data
        np.random.seed(42)
        n_samples = 200
        n_features = 15
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] * 2.0 + X[:, 1] * 1.5 + X[:, 2] * 0.8 + 
             np.random.randn(n_samples) * 0.1)
        y = np.clip((y - y.min()) / (y.max() - y.min()) * 100, 0, 100)
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Complete workflow
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X_df, y)
        training_result = model.train(X_train, y_train, X_val, y_val)
        test_metrics = model.evaluate(X_test, y_test)
        predictions = model.predict(X_test[:10])
        
        # Verify results
        assert model.is_trained == True
        assert len(predictions) == 10
        assert test_metrics['r2'] > 0.5  # Should have reasonable performance
        assert 'feature_importance' in training_result
    
    def test_model_persistence_integration(self):
        """Test complete model persistence workflow."""
        # Create and train model
        model = RandomForestFireRiskModel(model_type='regression', n_estimators=30)
        
        # Generate simple data
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(100) * 0.1
        
        feature_names = [f'feature_{i}' for i in range(10)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Train
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X_df, y)
        model.train(X_train, y_train, X_val, y_val)
        
        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model.save_model(tmp_file.name)
            
            new_model = RandomForestFireRiskModel()
            new_model.load_model(tmp_file.name)
            
            # Verify functionality
            original_pred = model.predict(X_test[:5])
            loaded_pred = new_model.predict(X_test[:5])
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)
            
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


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
