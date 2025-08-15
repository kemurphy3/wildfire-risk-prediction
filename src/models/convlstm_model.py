"""
ConvLSTM model for spatiotemporal wildfire risk prediction.

This module implements a Convolutional LSTM (ConvLSTM) model for
spatiotemporal prediction of wildfire risk using satellite and
environmental data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Sequential
from typing import Tuple, Dict, Any, Optional, Union, List
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class ConvLSTMFireRiskModel:
    """
    ConvLSTM model for spatiotemporal wildfire risk prediction.
    
    This class provides a ConvLSTM implementation for wildfire risk assessment
    that can handle both spatial and temporal data. It combines convolutional
    layers for spatial feature extraction with LSTM layers for temporal
    dependencies, making it suitable for predicting fire risk across geographic
    areas over time.
    
    Attributes:
        model_type (str): Type of prediction task ('regression' or 'classification')
        model (tf.keras.Model): The trained ConvLSTM model
        scaler (StandardScaler): Feature scaler for input data
        feature_names (List[str]): Names of input features
        is_trained (bool): Whether the model has been trained
        training_history (Dict): Training history and metrics
        input_shape (Tuple): Shape of input data (time_steps, height, width, channels)
        time_steps (int): Number of time steps for temporal sequences
        spatial_dims (Tuple): Spatial dimensions (height, width)
        channels (int): Number of input channels/features
    """
    
    def __init__(
        self,
        model_type: str = 'regression',
        time_steps: int = 10,
        spatial_dims: Tuple[int, int] = (32, 32),
        channels: int = 10,
        filters: int = 64,
        kernel_size: int = 3,
        lstm_units: int = 128,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the ConvLSTM model.
        
        Args:
            model_type: Type of prediction task ('regression' or 'classification')
            time_steps: Number of time steps for temporal sequences
            spatial_dims: Spatial dimensions (height, width)
            channels: Number of input channels/features
            filters: Number of convolutional filters
            kernel_size: Size of convolutional kernels
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type.lower()
        self.time_steps = time_steps
        self.spatial_dims = spatial_dims
        self.channels = channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.random_state = random_state
        
        self.is_trained = False
        self.training_history = {}
        self.feature_names = []
        self.input_shape = (time_steps, spatial_dims[0], spatial_dims[1], channels)
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def _build_model(self) -> tf.keras.Model:
        """Build the ConvLSTM model architecture."""
        model = Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # ConvLSTM layers
            layers.ConvLSTM2D(
                filters=self.filters,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding='same',
                return_sequences=True,
                activation='relu'
            ),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            layers.ConvLSTM2D(
                filters=self.filters // 2,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding='same',
                return_sequences=False,
                activation='relu'
            ),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            # Global pooling to reduce spatial dimensions
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(self.lstm_units, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            layers.Dense(self.lstm_units // 2, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            # Output layer
            layers.Dense(
                1 if self.model_type == 'regression' else 2,
                activation='linear' if self.model_type == 'regression' else 'softmax'
            )
        ])
        
        # Compile model
        if self.model_type == 'regression':
            loss = 'mse'
            metrics = ['mae', 'mse']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def prepare_data(self, features, targets, test_size=0.2, val_size=0.2, random_state=42):
        """
        Prepare data for training, validation, and testing.
        
        Args:
            features: Feature data (DataFrame or numpy array)
            targets: Target data (Series or numpy array)
            test_size: Proportion of data for testing
            val_size: Proportion of remaining data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Convert to numpy arrays if needed
        if hasattr(features, 'values'):
            X = features.values
            if hasattr(features, 'columns'):
                self.feature_names = features.columns.tolist()
            else:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        else:
            X = np.array(features)
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        if hasattr(targets, 'values'):
            y = targets.values
        else:
            y = np.array(targets)
        
        # Validate data shapes
        if len(X.shape) == 2:
            # 2D data: (samples, features) - reshape to 5D for ConvLSTM
            # We'll create a spatial representation by organizing features into a grid
            n_samples, n_features = X.shape
            
            # Calculate grid dimensions (try to make it square-ish)
            grid_size = int(np.ceil(np.sqrt(n_features)))
            padded_features = grid_size * grid_size
            
            # Pad features if needed
            if n_features < padded_features:
                X_padded = np.zeros((n_samples, padded_features))
                X_padded[:, :n_features] = X
                X = X_padded
                # Update feature names
                for i in range(n_features, padded_features):
                    self.feature_names.append(f'padded_feature_{i}')
            
            # Reshape to 5D: (samples, time_steps, height, width, channels)
            # For now, we'll use 1 time step, grid_size x grid_size spatial dimensions, 1 channel
            X = X.reshape(n_samples, 1, grid_size, grid_size, 1)
            
        elif len(X.shape) == 5:
            # Already 5D data
            pass
        else:
            raise ValueError(f"Expected 2D or 5D input data, got {len(X.shape)}D")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if self.model_type == 'classification' else None
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp if self.model_type == 'classification' else None
        )
        
        # Store data shapes for model building
        self.input_shape = X.shape[1:]  # (time_steps, height, width, channels)
        self.n_features = X.shape[-1] * X.shape[-2] * X.shape[-3]  # total spatial features
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _reshape_for_spatiotemporal(self, X: np.ndarray) -> np.ndarray:
        """Reshape 2D data to 4D spatiotemporal format."""
        n_samples = X.shape[0]
        
        # Calculate how many spatial samples we can create
        spatial_samples = n_samples // self.time_steps
        
        # Reshape to (spatial_samples, time_steps, height, width, channels)
        X_reshaped = X[:spatial_samples * self.time_steps].reshape(
            spatial_samples, self.time_steps, self.spatial_dims[0], 
            self.spatial_dims[1], self.channels
        )
        
        return X_reshaped
    
    def _scale_spatiotemporal_data(self, X: np.ndarray) -> np.ndarray:
        """Scale spatiotemporal data while preserving structure."""
        original_shape = X.shape
        
        # Flatten for scaling
        X_flat = X.reshape(-1, X.shape[-1])
        
        # Scale
        X_scaled = self.scaler.fit_transform(X_flat)
        
        # Reshape back
        X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        hyperparameter_tuning: bool = False,
        callbacks_list: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Train the ConvLSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            callbacks_list: List of Keras callbacks
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Set up callbacks
        if callbacks_list is None:
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                ),
                callbacks.ModelCheckpoint(
                    'best_convlstm_model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False
                )
            ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            validation_split=self.validation_split if validation_data is None else 0.0,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Store training history
        self.training_history = history.history
        
        # Calculate training metrics
        train_predictions = self.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_predictions)
        
        # Calculate validation metrics if available
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_predictions)
        
        self.is_trained = True
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_history': self.training_history
        }
    
    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predictions or tuple of (predictions, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure data is in correct format
        if len(X.shape) == 2:
            X = self._reshape_for_spatiotemporal(X)
        
        # Scale data
        X_scaled = self._scale_spatiotemporal_data(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        if return_confidence and self.model_type == 'classification':
            # For classification, return predictions and probabilities
            pred_classes = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
            return pred_classes, confidence
        elif return_confidence and self.model_type == 'regression':
            # For regression, return predictions and uncertainty (approximation)
            # Use model's internal uncertainty if available
            confidence = np.std(predictions, axis=0) if len(predictions.shape) > 1 else np.zeros_like(predictions)
            return predictions.flatten(), confidence
        else:
            return predictions.flatten() if self.model_type == 'regression' else np.argmax(predictions, axis=1)
    
    def predict_with_intervals(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals using bootstrap.
        
        Args:
            X: Input features
            confidence_level: Confidence level for intervals
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure data is in correct format
        if len(X.shape) == 2:
            X = self._reshape_for_spatiotemporal(X)
        
        # Scale data
        X_scaled = self._scale_spatiotemporal_data(X)
        
        # Bootstrap predictions
        bootstrap_predictions = []
        for _ in range(n_bootstrap):
            # Add noise to input for bootstrap
            noise = np.random.normal(0, 0.01, X_scaled.shape)
            X_noisy = X_scaled + noise
            
            pred = self.model.predict(X_noisy)
            bootstrap_predictions.append(pred.flatten())
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Calculate statistics
        predictions = np.mean(bootstrap_predictions, axis=0)
        lower_bounds = np.percentile(bootstrap_predictions, (1 - confidence_level) * 50, axis=0)
        upper_bounds = np.percentile(bootstrap_predictions, (1 + confidence_level) * 50, axis=0)
        
        return predictions, lower_bounds, upper_bounds
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X_test)
        return self._calculate_metrics(y_test, predictions)
    
    def plot_training_history(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        """Plot training history."""
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting history")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Loss
        axes[0].plot(self.training_history['loss'], label='Training Loss')
        if 'val_loss' in self.training_history:
            axes[0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Metrics
        if self.model_type == 'regression':
            if 'mae' in self.training_history:
                axes[1].plot(self.training_history['mae'], label='Training MAE')
                if 'val_mae' in self.training_history:
                    axes[1].plot(self.training_history['val_mae'], label='Validation MAE')
                axes[1].set_title('Mean Absolute Error')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('MAE')
                axes[1].legend()
                axes[1].grid(True)
            
            if 'mse' in self.training_history:
                axes[2].plot(self.training_history['mse'], label='Training MSE')
                if 'val_mse' in self.training_history:
                    axes[2].plot(self.training_history['val_mse'], label='Validation MSE')
                axes[2].set_title('Mean Squared Error')
                axes[2].set_xlabel('Epoch')
                axes[2].set_ylabel('MSE')
                axes[2].legend()
                axes[2].grid(True)
        else:
            if 'accuracy' in self.training_history:
                axes[1].plot(self.training_history['accuracy'], label='Training Accuracy')
                if 'val_accuracy' in self.training_history:
                    axes[1].plot(self.training_history['val_accuracy'], label='Validation Accuracy')
                axes[1].set_title('Model Accuracy')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                axes[1].legend()
                axes[1].grid(True)
            
            axes[2].axis('off')  # Hide third subplot for classification
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and scaler
        model_data = {
            'model_weights': self.model.get_weights(),
            'model_config': self.model.get_config(),
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'input_shape': self.input_shape,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        model_data = joblib.load(filepath)
        
        # Rebuild model
        self.model = models.model_from_config(model_data['model_config'])
        self.model.set_weights(model_data['model_weights'])
        
        # Restore other attributes
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data['training_history']
        self.input_shape = model_data['input_shape']
        self.is_trained = model_data['is_trained']
        
        # Recompile model
        if self.model_type == 'regression':
            loss = 'mse'
            metrics = ['mae', 'mse']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=metrics
        )
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        if self.model_type == 'regression':
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
        else:
            # For classification, ensure predictions are classes
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred_classes = np.argmax(y_pred, axis=1)
            else:
                y_pred_classes = y_pred
            
            return {
                'accuracy': accuracy_score(y_true, y_pred_classes),
                'classification_report': classification_report(y_true, y_pred_classes, output_dict=True)
            }
