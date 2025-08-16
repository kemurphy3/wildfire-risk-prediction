# LightGBM model for fire risk
# basically gradient boosting but faster

import os
import pickle
import warnings
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)


class LightGBMFireRiskModel:
    # lightgbm wrapper for fire predictions
    # handles both regression and classification
    
    def __init__(
        self,
        model_type: str = 'regression',
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: int = 10,
        eval_metric: Optional[str] = None
    ):
        # setup lightgbm with basic params
        self.model_type = model_type.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
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
            self.eval_metric = 'l2' if self.model_type == 'regression' else 'binary_logloss'
        
        # Initialize the appropriate LightGBM model
        if self.model_type == 'regression':
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=-1
            )
        else:
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=-1
            )
        
        logger.info(f"LightGBM {model_type} model initialized")
    
    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Names of features (optional)
        
        Returns:
            Dictionary containing training metrics and history
        """
        try:
            # Store feature names
            if feature_names is not None:
                self.feature_names = feature_names
            elif hasattr(X_train, 'columns'):
                self.feature_names = list(X_train.columns)
            else:
                self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
            
            # Prepare training data
            train_data = lgb.Dataset(X_train_scaled, label=y_train, feature_name=self.feature_names)
            
            # Training parameters
            params = {
                'objective': 'regression' if self.model_type == 'regression' else 'binary',
                'metric': self.eval_metric,
                'boosting_type': 'gbdt',
                'num_leaves': self.num_leaves,
                'learning_rate': self.learning_rate,
                'feature_fraction': self.colsample_bytree,
                'bagging_fraction': self.subsample,
                'bagging_freq': 1,
                'verbose': -1,
                'random_state': self.random_state
            }
            
            # Train with early stopping if validation data provided
            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val_scaled, label=y_val, feature_name=self.feature_names)
                
                # Train with early stopping
                self.model = lgb.train(
                    params,
                    train_set=train_data,
                    valid_sets=[train_data, val_data],
                    valid_names=['train', 'valid'],
                    num_boost_round=self.n_estimators,
                    callbacks=[lgb.early_stopping(self.early_stopping_rounds)]
                )
                
                # Extract training history - simplified for compatibility
                self.training_history = {
                    'train_loss': None,
                    'val_loss': None,
                    'best_iteration': getattr(self.model, 'best_iteration', self.n_estimators)
                }
            else:
                # Train without validation
                self.model = lgb.train(
                    params,
                    train_set=train_data,
                    num_boost_round=self.n_estimators
                )
                
                self.training_history = {
                    'train_loss': None,
                    'val_loss': None,
                    'best_iteration': getattr(self.model, 'best_iteration', self.n_estimators)
                }
            
            self.is_trained = True
            
            # Calculate training metrics
            y_pred = self.predict(X_train)
            metrics = self._calculate_metrics(y_train, y_pred, 'train')
            
            logger.info(f"LightGBM model trained successfully. Training metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training LightGBM model: {e}")
            raise
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
        
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            if self.model_type == 'regression':
                predictions = self.model.predict(X_scaled)
            else:
                # For classification, return probability of positive class
                predictions = self.model.predict_proba(X_scaled)[:, 1]
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'split', 'cover')
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        try:
            # Get feature importance
            importance_scores = self.model.feature_importance(importance_type=importance_type)
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_scores
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to show
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        try:
            importance_df = self.get_feature_importance()
            
            if importance_df.empty:
                raise ValueError("No feature importance data available")
            
            # Get top N features
            top_features = importance_df.head(top_n)
            
            # Create plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Horizontal bar plot
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_features['importance'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top {top_n} Feature Importance (LightGBM)')
            ax.invert_yaxis()  # Highest importance at top
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
            raise
    
    def explain_prediction(self, X: Union[np.ndarray, pd.DataFrame], sample_idx: int = 0) -> plt.Figure:
        """
        Explain a single prediction using SHAP.
        
        Args:
            X: Input features
            sample_idx: Index of sample to explain
        
        Returns:
            SHAP explanation plot
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before explaining predictions")
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_scaled)
            
            # Plot SHAP values for the specified sample
            fig = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values[sample_idx], X_scaled[sample_idx], 
                             feature_names=self.feature_names, show=False)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            raise
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'model_type': self.model_type,
                    'training_history': self.training_history,
                    'best_params': self.best_params
                }, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.training_history = model_data['training_history']
            self.best_params = model_data['best_params']
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            dataset_name: Name of the dataset
        
        Returns:
            Dictionary of metrics
        """
        try:
            metrics = {}
            
            if self.model_type == 'regression':
                metrics[f'{dataset_name}_mae'] = mean_absolute_error(y_true, y_pred)
                metrics[f'{dataset_name}_mse'] = mean_squared_error(y_true, y_pred)
                metrics[f'{dataset_name}_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                metrics[f'{dataset_name}_r2'] = r2_score(y_true, y_pred)
            else:
                # For classification, convert probabilities to binary predictions
                y_pred_binary = (y_pred > 0.5).astype(int)
                metrics[f'{dataset_name}_accuracy'] = accuracy_score(y_true, y_pred_binary)
                metrics[f'{dataset_name}_auc'] = roc_auc_score(y_true, y_pred)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'best_params': self.best_params,
            'model_params': self.model.params if hasattr(self.model, 'params') else {}
        }


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = LightGBMFireRiskModel(model_type='regression')
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    
    # Train model
    metrics = model.train(X, y)
    print(f"Training metrics: {metrics}")
    
    # Make predictions
    predictions = model.predict(X[:5])
    print(f"Predictions: {predictions}")
    
    # Get feature importance
    importance = model.get_feature_importance()
    print(f"Feature importance:\n{importance}")
