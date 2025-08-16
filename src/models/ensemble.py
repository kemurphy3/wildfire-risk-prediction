# Ensemble model implementation - combines multiple models for improved predictions

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.ensemble import VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import joblib
import os
import warnings

# Import our custom models
from .baseline_model import RandomForestFireRiskModel
from .xgboost_model import XGBoostFireRiskModel
from .convlstm_model import ConvLSTMFireRiskModel
from .lightgbm_model import LightGBMFireRiskModel


class SklearnWrapper:
    # Wrapper to make custom models compatible with sklearn
    
    def __init__(self, model):
        self.model = model
        self._is_fitted = False
    
    def fit(self, X, y):
        # Fit the model
        # For our custom models, we need to prepare data first
        if hasattr(self.model, 'prepare_data'):
            X_train, X_val, X_test, y_train, y_val, y_test = self.model.prepare_data(
                pd.DataFrame(X), y, test_size=0.0, validation_size=0.0
            )
            # Use all data for training
            X_all = np.vstack([X_train, X_val, X_test]) if len(X_val) > 0 else X_train
            y_all = np.concatenate([y_train, y_val, y_test]) if len(y_val) > 0 else y_train
            self.model.train(X_all, y_all)
        else:
            # Fallback for sklearn models
            self.model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X):
        # Generate predictions
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X):
        # Get probabilities for classification tasks
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback: create dummy probabilities
            predictions = self.predict(X)
            unique_classes = np.unique(predictions)
            proba = np.zeros((len(predictions), len(unique_classes)))
            for i, pred in enumerate(predictions):
                proba[i, np.where(unique_classes == pred)[0][0]] = 1.0
            return proba


class EnsembleFireRiskModel:
    '''
    Combines different models for fire risk prediction.
    Supports voting, stacking, and weighted averaging methods.
    '''
    
    def __init__(
        self,
        model_type: str = 'regression',
        ensemble_method: str = 'voting',
        base_models: Optional[List] = None,
        model_weights: Optional[List[float]] = None,
        random_state: int = 42
    ):
        # Initialize ensemble with specified parameters
        self.model_type = model_type.lower()
        self.ensemble_method = ensemble_method.lower()
        self.random_state = random_state
        self.is_trained = False
        self.feature_names = []
        self.base_predictions = {}
        
        # Validate ensemble method
        valid_methods = ['voting', 'stacking', 'weighted']
        if self.ensemble_method not in valid_methods:
            raise ValueError(f"ensemble_method must be one of {valid_methods}")
        
        # Initialize base models if not provided
        if base_models is None:
            self.base_models = self._initialize_default_models()
        else:
            self.base_models = base_models
        
        # Validate model types
        self._validate_base_models()
        
        # Set model weights
        if model_weights is None:
            self.model_weights = [1.0 / len(self.base_models)] * len(self.base_models)
        else:
            if len(model_weights) != len(self.base_models):
                raise ValueError("Number of weights must match number of base models")
            self.model_weights = model_weights
        
        # Initialize ensemble model
        self.ensemble_model = self._build_ensemble_model()
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
    def _initialize_default_models(self) -> List:
        """Initialize default base models."""
        models = []
        
        # Random Forest
        rf_model = RandomForestFireRiskModel(
            model_type=self.model_type,
            random_state=self.random_state
        )
        models.append(('random_forest', rf_model))
        
        # XGBoost
        xgb_model = XGBoostFireRiskModel(
            model_type=self.model_type,
            random_state=self.random_state
        )
        models.append(('xgboost', xgb_model))
        
        # ConvLSTM (only for regression due to complexity)
        if self.model_type == 'regression':
            convlstm_model = ConvLSTMFireRiskModel(
                model_type=self.model_type,
                random_state=self.random_state
            )
            models.append(('convlstm', convlstm_model))
        
        return models
    
    def _validate_base_models(self) -> None:
        # Validate that all base models have the correct type
        for i, item in enumerate(self.base_models):
            # Handle both tuple format (name, model) and direct model objects
            if isinstance(item, tuple) and len(item) == 2:
                name, model = item
            else:
                # If it's just a model object, create a name
                model = item
                name = f"model_{i}"
                # Update the base_models list to use tuple format
                self.base_models[i] = (name, model)
            
            if hasattr(model, 'model_type'):
                if model.model_type != self.model_type:
                    raise ValueError(f"Model {name} has type {model.model_type}, expected {self.model_type}")
            else:
                # If model doesn't have model_type, assume it's compatible
                pass
    
    def _build_ensemble_model(self):
        # Build ensemble model based on the chosen method
        # Create sklearn-compatible estimators from our custom models
        sklearn_estimators = []
        for name, model in self.base_models:
            # Create a wrapper that makes our custom models sklearn-compatible
            sklearn_estimators.append((name, SklearnWrapper(model)))
        
        if self.ensemble_method == 'voting':
            if self.model_type == 'regression':
                return VotingRegressor(
                    estimators=sklearn_estimators,
                    weights=self.model_weights,
                    n_jobs=-1
                )
            else:
                return VotingClassifier(
                    estimators=sklearn_estimators,
                    voting='soft',
                    weights=self.model_weights,
                    n_jobs=-1
                )
        
        elif self.ensemble_method == 'stacking':
            # Use linear models as meta-learners
            if self.model_type == 'regression':
                meta_learner = LinearRegression()
                return StackingRegressor(
                    estimators=sklearn_estimators,
                    final_estimator=meta_learner,
                    cv=5,
                    n_jobs=-1
                )
            else:
                meta_learner = LogisticRegression(random_state=self.random_state)
                return StackingClassifier(
                    estimators=sklearn_estimators,
                    final_estimator=meta_learner,
                    cv=5,
                    n_jobs=-1
                )
        
        else:  # weighted
            # For weighted averaging, we'll implement custom logic
            return None
    
    def prepare_data(
        self,
        features: pd.DataFrame,
        target: Union[pd.Series, np.ndarray],
        test_size: float = 0.2,
        validation_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training, validation, and testing.
        
        Args:
            features: Input features DataFrame
            target: Target variable
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Convert to numpy arrays
        X = features.values
        y = target.values if hasattr(target, 'values') else np.array(target)
        
        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Split train+val into train and val
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        train_base_models: bool = True,
        optimize_weights: bool = False
    ) -> Dict[str, Any]:
        """
        Train the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            train_base_models: Whether to train base models
            optimize_weights: Whether to optimize ensemble weights
            
        Returns:
            Dictionary containing training results and metrics
        """
        if train_base_models:
            # Train individual base models
            self._train_base_models(X_train, y_train, X_val, y_val)
        
        # Train ensemble model if not using weighted averaging
        if self.ensemble_method != 'weighted':
            self.ensemble_model.fit(X_train, y_train)
        
        # Optimize weights if requested
        if optimize_weights and self.ensemble_method == 'weighted':
            self._optimize_weights(X_train, y_train, X_val, y_val)
        
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
            'base_model_metrics': self._get_base_model_metrics(X_train, y_train, X_val, y_val)
        }
    
    def _train_base_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """Train individual base models."""
        for name, model in self.base_models:
            try:
                if hasattr(model, 'train'):
                    # For custom models with train method
                    model.train(X_train, y_train, X_val, y_val)
                else:
                    # For sklearn models
                    model.fit(X_train, y_train)
                print(f"Trained {name} successfully")
            except Exception as e:
                print(f"Error training {name}: {e}")
                warnings.warn(f"Failed to train {name}: {e}")
    
    def _optimize_weights(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """Optimize ensemble weights using validation data."""
        if X_val is None or y_val is None:
            print("Validation data required for weight optimization")
            return
        
        # Get base model predictions
        base_preds = {}
        for name, model in self.base_models:
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_val)
                    base_preds[name] = pred
                else:
                    warnings.warn(f"Model {name} does not have predict method")
            except Exception as e:
                warnings.warn(f"Error getting predictions from {name}: {e}")
        
        if not base_preds:
            print("No valid base model predictions for weight optimization")
            return
        
        # Convert to DataFrame for optimization
        pred_df = pd.DataFrame(base_preds)
        
        # Simple grid search for weights
        best_score = float('-inf') if self.model_type == 'regression' else 0
        best_weights = self.model_weights
        
        # Grid search over weight combinations
        weight_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for w1 in weight_options:
            for w2 in weight_options:
                if len(self.base_models) >= 2:
                    w3 = 1 - w1 - w2
                    if w3 >= 0:
                        weights = [w1, w2, w3]
                        if len(self.base_models) > 3:
                            weights.extend([0.1] * (len(self.base_models) - 3))
                        
                        # Normalize weights
                        weights = np.array(weights)
                        weights = weights / np.sum(weights)
                        
                        # Calculate ensemble prediction
                        ensemble_pred = np.zeros(len(y_val))
                        for i, (name, _) in enumerate(self.base_models):
                            if name in base_preds:
                                ensemble_pred += weights[i] * base_preds[name]
                        
                        # Calculate score
                        if self.model_type == 'regression':
                            score = r2_score(y_val, ensemble_pred)
                        else:
                            score = accuracy_score(y_val, np.round(ensemble_pred))
                        
                        # Update best weights
                        if (self.model_type == 'regression' and score > best_score) or \
                           (self.model_type == 'classification' and score > best_score):
                            best_score = score
                            best_weights = weights.copy()
        
        self.model_weights = best_weights.tolist()
        print(f"Optimized weights: {best_weights}")
        print(f"Best validation score: {best_score:.4f}")
    
    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using the ensemble model.
        
        Args:
            X: Input features
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predictions or tuple of (predictions, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        if self.ensemble_method == 'weighted':
            # Custom weighted averaging
            predictions = self._weighted_average_predict(X_scaled)
        else:
            # Use sklearn ensemble
            predictions = self.ensemble_model.predict(X_scaled)
        
        if return_confidence:
            confidence = self._calculate_ensemble_confidence(X_scaled)
            return predictions, confidence
        
        return predictions
    
    def _weighted_average_predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using weighted averaging."""
        predictions = np.zeros(len(X))
        
        for i, (name, model) in enumerate(self.base_models):
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    predictions += self.model_weights[i] * pred
                else:
                    warnings.warn(f"Model {name} does not have predict method")
            except Exception as e:
                warnings.warn(f"Error getting predictions from {name}: {e}")
        
        return predictions
    
    def _calculate_ensemble_confidence(self, X: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for ensemble predictions."""
        base_predictions = []
        
        for name, model in self.base_models:
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    base_predictions.append(pred)
                else:
                    warnings.warn(f"Model {name} does not have predict method")
            except Exception as e:
                warnings.warn(f"Error getting predictions from {name}: {e}")
        
        if not base_predictions:
            return np.zeros(len(X))
        
        # Calculate standard deviation as confidence measure
        base_preds_array = np.array(base_predictions)
        confidence = np.std(base_preds_array, axis=0)
        
        return confidence
    
    def predict_with_intervals(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Args:
            X: Input features
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get base predictions
        base_predictions = []
        for name, model in self.base_models:
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    base_predictions.append(pred)
                else:
                    warnings.warn(f"Model {name} does not have predict method")
            except Exception as e:
                warnings.warn(f"Error getting predictions from {name}: {e}")
        
        if not base_predictions:
            raise ValueError("No valid base model predictions available")
        
        # Calculate ensemble statistics
        base_preds_array = np.array(base_predictions)
        predictions = np.mean(base_preds_array, axis=0)
        
        # Calculate confidence intervals
        margin = np.percentile(base_preds_array, (1 - confidence_level) * 50, axis=0)
        lower_bounds = predictions - margin
        upper_bounds = predictions + margin
        
        return predictions, lower_bounds, upper_bounds
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the ensemble model on test data.
        
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
    
    def _get_base_model_metrics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Dict]:
        """Get metrics for individual base models."""
        metrics = {}
        
        for name, model in self.base_models:
            try:
                if hasattr(model, 'predict'):
                    # Training metrics
                    train_pred = model.predict(X_train)
                    train_metrics = self._calculate_metrics(y_train, train_pred)
                    
                    # Validation metrics
                    val_metrics = {}
                    if X_val is not None and y_val is not None:
                        val_pred = model.predict(X_val)
                        val_metrics = self._calculate_metrics(y_val, val_pred)
                    
                    metrics[name] = {
                        'train': train_metrics,
                        'validation': val_metrics
                    }
                else:
                    metrics[name] = {'error': 'No predict method'}
            except Exception as e:
                metrics[name] = {'error': str(e)}
        
        return metrics
    
    def plot_base_model_comparison(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """Plot comparison of base model performances."""
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting")
        
        # Get predictions from all models
        model_predictions = {}
        model_names = []
        
        # Base models
        for name, model in self.base_models:
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_test)
                    model_predictions[name] = pred
                    model_names.append(name)
            except Exception as e:
                print(f"Error getting predictions from {name}: {e}")
        
        # Ensemble
        ensemble_pred = self.predict(X_test)
        model_predictions['ensemble'] = ensemble_pred
        model_names.append('ensemble')
        
        # Calculate metrics
        metrics_data = []
        for name in model_names:
            pred = model_predictions[name]
            if self.model_type == 'regression':
                r2 = r2_score(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                mae = mean_absolute_error(y_test, pred)
                metrics_data.append([name, r2, rmse, mae])
            else:
                acc = accuracy_score(y_test, np.round(pred))
                metrics_data.append([name, acc, 0, 0])
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Metrics comparison
        metrics_df = pd.DataFrame(
            metrics_data,
            columns=['Model', 'R²/Accuracy', 'RMSE', 'MAE']
        )
        
        if self.model_type == 'regression':
            axes[0, 0].bar(metrics_df['Model'], metrics_df['R²/Accuracy'])
            axes[0, 0].set_title('R² Score Comparison')
            axes[0, 0].set_ylabel('R² Score')
        else:
            axes[0, 0].bar(metrics_df['Model'], metrics_df['R²/Accuracy'])
            axes[0, 0].set_title('Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
        
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        if self.model_type == 'regression':
            axes[0, 1].bar(metrics_df['Model'], metrics_df['RMSE'])
            axes[0, 1].set_title('RMSE Comparison')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            axes[1, 0].bar(metrics_df['Model'], metrics_df['MAE'])
            axes[1, 0].set_title('MAE Comparison')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Predictions vs actual
        axes[1, 1].scatter(y_test, ensemble_pred, alpha=0.6, label='Ensemble')
        axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Ensemble Predictions vs Actual')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def create_modern_ensemble(self) -> Union[VotingRegressor, VotingClassifier]:
        """
        Create a 2024 state-of-the-art ensemble model.
        
        Combines traditional and modern approaches:
        - Random Forest (baseline)
        - XGBoost (gradient boosting)
        - LightGBM (modern gradient boosting)
        
        References:
            - Prapas et al. (2023): "Deep Learning for Global Wildfire Forecasting"
            - Michael et al. (2024): "ML for High-Resolution Predictions"
        """
        # Create LightGBM model
        lgb_model = LightGBMFireRiskModel(model_type=self.model_type)
        
        # Create modern ensemble with LightGBM
        modern_models = [
            ('random_forest', RandomForestFireRiskModel(model_type=self.model_type)),
            ('xgboost', XGBoostFireRiskModel(model_type=self.model_type)),
            ('lightgbm', lgb_model)
        ]
        
        # Create sklearn-compatible estimators
        sklearn_estimators = []
        for name, model in modern_models:
            sklearn_estimators.append((name, SklearnWrapper(model)))
        
        # Create voting ensemble with optimized weights
        if self.model_type == 'regression':
            return VotingRegressor(
                estimators=sklearn_estimators,
                weights=[0.3, 0.35, 0.35],
                n_jobs=-1
            )
        else:
            return VotingClassifier(
                estimators=sklearn_estimators,
                voting='soft',
                weights=[0.3, 0.35, 0.35],
                n_jobs=-1
            )
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained ensemble model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save ensemble data
        ensemble_data = {
            'ensemble_method': self.ensemble_method,
            'model_weights': self.model_weights,
            'base_models': self.base_models,
            'ensemble_model': self.ensemble_model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(ensemble_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained ensemble model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load ensemble data
        ensemble_data = joblib.load(filepath)
        
        # Restore ensemble state
        self.ensemble_method = ensemble_data['ensemble_method']
        self.model_weights = ensemble_data['model_weights']
        self.base_models = ensemble_data['base_models']
        self.ensemble_model = ensemble_data['ensemble_model']
        self.scaler = ensemble_data['scaler']
        self.model_type = ensemble_data['model_type']
        self.feature_names = ensemble_data['feature_names']
        self.is_trained = ensemble_data['is_trained']
    
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
                y_pred_classes = np.round(y_pred)
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred_classes),
                'classification_report': classification_report(y_true, y_pred_classes, output_dict=True)
            }
            
            # Add ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
                except:
                    pass
            
            return metrics
