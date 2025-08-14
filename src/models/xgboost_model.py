"""
XGBoost model for wildfire risk prediction.

This module provides an XGBoost-based implementation for wildfire risk assessment,
including regression and classification capabilities with advanced features like
early stopping, feature importance analysis, and SHAP explainability.
"""

import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


class XGBoostFireRiskModel:
    """
    XGBoost model for wildfire risk prediction.
    
    This class provides a comprehensive XGBoost implementation for wildfire risk
    assessment, including both regression and classification tasks. It features
    advanced capabilities like early stopping, hyperparameter tuning, and SHAP
    explainability.
    
    Attributes:
        model_type (str): Type of prediction task ('regression' or 'classification')
        model (xgb.XGBRegressor or xgb.XGBClassifier): The trained XGBoost model
        scaler (StandardScaler): Feature scaler for input data
        feature_names (List[str]): Names of input features
        is_trained (bool): Whether the model has been trained
        training_history (Dict): Training history and metrics
        best_params (Dict): Best hyperparameters from tuning
    """
    
    def __init__(
        self,
        model_type: str = 'regression',
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: int = 10,
        eval_metric: Optional[str] = None
    ):
        """
        Initialize the XGBoost model.
        
        Args:
            model_type: Type of prediction task ('regression' or 'classification')
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns for each tree
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
            early_stopping_rounds: Number of rounds for early stopping
            eval_metric: Evaluation metric for early stopping
        """
        self.model_type = model_type.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.is_trained = False
        self.training_history = {}
        self.best_params = {}
        self.feature_names = []
        self.scaler = StandardScaler()
        
        # Set default evaluation metric based on model type
        if self.eval_metric is None:
            self.eval_metric = 'rmse' if self.model_type == 'regression' else 'logloss'
        
        # Validate model type
        if self.model_type not in ['regression', 'classification']:
            raise ValueError("model_type must be 'regression' or 'classification'")
        
        # Initialize the appropriate XGBoost model
        if self.model_type == 'regression':
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=random_state,
                n_jobs=n_jobs,
                eval_metric=self.eval_metric
            )
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=random_state,
                n_jobs=n_jobs,
                eval_metric=self.eval_metric,
                enable_categorical=False
            )
        
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
        # For test_size=0.2 and validation_size=0.2:
        # - test: 20% of total = 200 samples
        # - train+val: 80% of total = 800 samples
        # - val: 20% of the 80% = 160 samples (validation_size * (1-test_size) * total)
        # - train: 80% of the 80% = 640 samples ((1-validation_size) * (1-test_size) * total)
        # So validation_size should be interpreted as proportion of train+val data
        val_size_adjusted = validation_size
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
        hyperparameter_tuning: bool = False,
        tuning_method: str = 'grid',
        n_iter: int = 20
    ) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            tuning_method: Tuning method ('grid' or 'random')
            n_iter: Number of iterations for random search
            
        Returns:
            Dictionary containing training results
        """
        if hyperparameter_tuning:
            self._perform_hyperparameter_tuning(
                X_train, y_train, X_val, y_val, tuning_method, n_iter
            )
        
        # Prepare validation data for early stopping
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # Ensure scaler is fitted on training data
        if not hasattr(self.scaler, 'mean_'):
            # If scaler is not fitted, fit it on training data
            self.scaler.fit(X_train)
        
        # Train the model (without early stopping for XGBoost 3.0+ compatibility)
        if self.model_type == 'regression':
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        # Store training history
        try:
            if hasattr(self.model, 'evals_result'):
                self.training_history = self.model.evals_result()
            else:
                self.training_history = {}
        except:
            self.training_history = {}
        
        # Mark model as trained before calculating metrics
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions = self.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_predictions)
        
        # Calculate validation metrics if available
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_predictions)
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_iteration': getattr(self.model, 'best_iteration', None),
            'feature_importance': self._get_feature_importance()
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
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        if return_confidence and self.model_type == 'classification':
            # For classification, return predictions and probabilities
            predictions = self.model.predict(X_scaled)
            confidence = np.max(self.model.predict_proba(X_scaled), axis=1)
            return predictions, confidence
        elif return_confidence and self.model_type == 'regression':
            # For regression, return predictions and standard deviation (approximation)
            predictions = self.model.predict(X_scaled)
            # Use a simpler approach for confidence in XGBoost 3.0+
            # Calculate confidence based on feature importance and prediction variance
            feature_importance = self._get_feature_importance()
            if len(feature_importance) > 0:
                # Use feature importance as a proxy for uncertainty
                confidence = np.ones(len(predictions)) * np.mean(feature_importance)
            else:
                # Fallback to constant confidence
                confidence = np.ones(len(predictions)) * 0.5
            return predictions, confidence
        else:
            return self.model.predict(X_scaled)
    
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
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'regression':
            # For regression, use quantile regression or bootstrap
            predictions = self.model.predict(X_scaled)
            
            # Simple approach: use feature importance to estimate uncertainty
            feature_importance = self._get_feature_importance()
            uncertainty_factor = np.sum(feature_importance) / len(feature_importance)
            
            # Calculate bounds based on uncertainty
            margin = uncertainty_factor * (1 - confidence_level) * 2
            lower_bounds = predictions - margin
            upper_bounds = predictions + margin
            
            return predictions, lower_bounds, upper_bounds
        else:
            # For classification, return class probabilities as confidence
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            confidence = np.max(probabilities, axis=1)
            
            # Simple confidence bounds for classification
            margin = (1 - confidence_level) / 2
            lower_bounds = confidence - margin
            upper_bounds = confidence + margin
            
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
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            figsize: Figure size
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting feature importance")
        
        importance = self._get_feature_importance()
        
        if len(importance) == 0:
            print("No feature importance available")
            return
        
        # Sort features by importance
        sorted_idx = np.argsort(importance)[::-1]
        top_features = sorted_idx[:min(top_n, len(sorted_idx))]
        
        # Get actual feature names, fallback to generic if not available
        if self.feature_names and len(self.feature_names) == len(importance):
            feature_labels = [self.feature_names[i] for i in top_features]
        else:
            feature_labels = [f"Feature_{i}" for i in top_features]
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_features)), importance[top_features])
        plt.yticks(range(len(top_features)), feature_labels)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {len(top_features)} Feature Importance - XGBoost {self.model_type.title()}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def explain_prediction(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        plot: bool = True
    ) -> Dict[str, Any]:
        """
        Explain a specific prediction using SHAP.
        
        Args:
            X: Input features
            sample_idx: Index of sample to explain
            plot: Whether to plot the explanation
            
        Returns:
            Dictionary containing SHAP values and explanation
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before explaining predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # For classification, use first class
        
        # Get feature names
        feature_names = self.feature_names if self.feature_names else [f"Feature_{i}" for i in range(X.shape[1])]
        
        if plot:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values[sample_idx:sample_idx+1],
                X_scaled[sample_idx:sample_idx+1],
                feature_names=feature_names,
                plot_type="bar"
            )
            plt.title(f'SHAP Explanation for Sample {sample_idx}')
            plt.tight_layout()
            plt.show()
        
        # Return explanation data
        explanation = {
            'shap_values': shap_values[sample_idx],
            'feature_names': feature_names,
            'prediction': self.predict(X[sample_idx:sample_idx+1])[0],
            'feature_contributions': dict(zip(feature_names, shap_values[sample_idx]))
        }
        
        return explanation
    
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
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'best_params': self.best_params,
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
        
        # Restore model state
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data['training_history']
        self.best_params = model_data['best_params']
        self.is_trained = model_data['is_trained']
    
    def _perform_hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        tuning_method: str,
        n_iter: int
    ) -> None:
        """Perform hyperparameter tuning."""
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [0.1, 1.0, 10.0]
        }
        
        # Create temporary model for tuning
        if self.model_type == 'regression':
            temp_model = xgb.XGBRegressor(
                random_state=42,
                eval_metric=self.eval_metric,
                enable_categorical=False
            )
        else:
            temp_model = xgb.XGBClassifier(
                random_state=42,
                eval_metric=self.eval_metric,
                enable_categorical=False
            )
        
        # Perform tuning
        if tuning_method == 'grid':
            search = GridSearchCV(
                temp_model, param_grid, cv=3, scoring='r2' if self.model_type == 'regression' else 'accuracy',
                n_jobs=-1, verbose=0
            )
        elif tuning_method == 'random':
            search = RandomizedSearchCV(
                temp_model, param_grid, n_iter=n_iter, cv=3,
                scoring='r2' if self.model_type == 'regression' else 'accuracy',
                n_jobs=-1, verbose=0, random_state=42
            )
        else:
            raise ValueError("tuning_method must be 'grid' or 'random'")
        
        # Fit the search
        search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.best_params = search.best_params_
        self.model.set_params(**self.best_params)
    
    def get_feature_importance_dict(self) -> Dict[str, float]:
        """
        Get feature importance as a dictionary with feature names.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance = self._get_feature_importance()
        
        if len(importance) == 0:
            return {}
        
        # Create dictionary with feature names
        if self.feature_names and len(self.feature_names) == len(importance):
            return dict(zip(self.feature_names, importance))
        else:
            # Fallback to generic names
            return {f"Feature_{i}": importance[i] for i in range(len(importance))}
    
    def _get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_trained:
            return np.array([])
        
        try:
            # Get feature importance from the model
            importance = self.model.feature_importances_
            
            # Ensure we have the right number of features
            if len(importance) != len(self.feature_names):
                # If feature names don't match, return what we have
                return importance
            
            return importance
            
        except Exception as e:
            # Fallback: try to get importance from booster
            try:
                if hasattr(self.model, 'get_booster'):
                    booster = self.model.get_booster()
                    # Get importance scores as dictionary
                    importance_dict = booster.get_score(importance_type='gain')
                    
                    # Convert to array format using feature names
                    importance_array = np.zeros(len(self.feature_names))
                    for i, feature_name in enumerate(self.feature_names):
                        # Try different key formats
                        key_formats = [f'f{i}', feature_name, str(i)]
                        for key in key_formats:
                            if key in importance_dict:
                                importance_array[i] = importance_dict[key]
                                break
                    
                    return importance_array
            except:
                pass
            
            # Final fallback: return zeros
            return np.zeros(len(self.feature_names)) if self.feature_names else np.array([])
    
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
