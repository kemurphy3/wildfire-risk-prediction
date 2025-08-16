"""
Baseline Random Forest Model for Wildfire Risk Prediction

This module implements a Random Forest model as the baseline approach
for wildfire risk assessment. Random Forests are robust, interpretable,
and provide good baseline performance for ecological prediction tasks.

References:
- Breiman (2001) - Random Forests
- Cutler et al. (2007) - Random Forests for classification in ecology
- Strobl et al. (2008) - Bias in random forest variable importance measures
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score, 
    train_test_split, 
    GridSearchCV,
    StratifiedKFold,
    KFold
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class RandomForestFireRiskModel:
    """
    Random Forest model for wildfire risk prediction.
    
    This class provides a comprehensive implementation of Random Forest
    for wildfire risk assessment, including both regression (continuous
    risk scores) and classification (risk categories) approaches.
    
    Attributes:
        model: Trained Random Forest model
        scaler: Feature scaler for preprocessing
        label_encoder: Label encoder for classification tasks
        feature_names: List of feature names
        model_type: Type of model ('regression' or 'classification')
        is_trained: Whether the model has been trained
    """
    
    def __init__(
        self,
        model_type: str = 'regression',
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the Random Forest model.
        
        Args:
            model_type: Type of model ('regression' or 'classification')
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf node
            random_state: Random seed for reproducibility
            n_jobs: Number of jobs for parallel processing
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Initialize model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        
        # Performance metrics storage
        self.training_metrics = {}
        self.validation_metrics = {}
        self.feature_importance = None
        
        # Initialize the appropriate model
        self._initialize_model()
        
        logger.info(f"Random Forest {model_type} model initialized")
    
    def _initialize_model(self) -> None:
        """Initialize the appropriate Random Forest model."""
        if self.model_type == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                oob_score=True  # Out-of-bag score for validation
            )
        elif self.model_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                oob_score=True,
                class_weight='balanced'  # Handle class imbalance
            )
        else:
            raise ValueError("Model type must be 'regression' or 'classification'")
    
    def prepare_data(
        self,
        features: pd.DataFrame,
        target: Union[pd.Series, np.ndarray],
        test_size: float = 0.2,
        validation_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training, validation, and testing.
        
        Args:
            features: Feature matrix
            target: Target variable
            test_size: Proportion of data for testing
            validation_size: Proportion of remaining data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            # Store feature names
            self.feature_names = features.columns.tolist()
            
            # Convert to numpy arrays
            X = features.values
            y = target.values if hasattr(target, 'values') else np.array(target)
            
            # Handle classification labels
            if self.model_type == 'classification':
                y = self.label_encoder.fit_transform(y)
                logger.info(f"Encoded {len(self.label_encoder.classes_)} classes: {self.label_encoder.classes_}")
            
            # Split into train/test first
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state,
                stratify=y if self.model_type == 'classification' else None
            )
            
            # Split remaining data into train/validation
            val_size_adjusted = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state,
                stratify=y_temp if self.model_type == 'classification' else None
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        hyperparameter_tuning: bool = False
    ) -> Dict[str, Any]:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            if hyperparameter_tuning:
                self._perform_hyperparameter_tuning(X_train, y_train)
            
            # Train the model
            logger.info("Training Random Forest model...")
            self.model.fit(X_train, y_train)
            
            # Store training metrics
            y_train_pred = self.model.predict(X_train)
            self.training_metrics = self._calculate_metrics(y_train, y_train_pred, 'training')
            
            # Store validation metrics if provided
            if X_val is not None and y_val is not None:
                y_val_pred = self.model.predict(X_val)
                self.validation_metrics = self._calculate_metrics(y_val, y_val_pred, 'validation')
            
            # Calculate feature importance
            self._calculate_feature_importance(X_train, y_train)
            
            # Mark as trained
            self.is_trained = True
            
            logger.info("Model training completed successfully")
            
            return {
                'training_metrics': self.training_metrics,
                'validation_metrics': self.validation_metrics,
                'feature_importance': self.feature_importance,
                'oob_score': getattr(self.model, 'oob_score_', None)
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def _perform_hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """Perform hyperparameter tuning using GridSearchCV."""
        logger.info("Performing hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Choose appropriate cross-validation strategy
        if self.model_type == 'classification':
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error' if self.model_type == 'regression' else 'f1_weighted',
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix for prediction
            return_confidence: Whether to return prediction confidence
            
        Returns:
            Predictions and optionally confidence scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            if return_confidence:
                # Get prediction probabilities/confidence
                if self.model_type == 'classification':
                    confidence = self.model.predict_proba(X_scaled).max(axis=1)
                else:
                    # For regression, use normalized confidence based on tree predictions
                    predictions_all_trees = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
                    std_dev = np.std(predictions_all_trees, axis=0)
                    
                    # Normalize confidence to be between 0 and 1
                    # Higher confidence = lower standard deviation (more agreement between trees)
                    max_std = np.max(std_dev) if np.max(std_dev) > 0 else 1.0
                    confidence = 1.0 - (std_dev / max_std)
                    # Ensure confidence is between 0 and 1
                    confidence = np.clip(confidence, 0.0, 1.0)
                
                return predictions, confidence
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_with_intervals(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Args:
            X: Feature matrix for prediction
            confidence_level: Confidence level for intervals (0.95 = 95%)
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        try:
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all trees
            predictions_all_trees = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
            
            # Calculate mean and standard deviation
            predictions = np.mean(predictions_all_trees, axis=0)
            std_dev = np.std(predictions_all_trees, axis=0)
            
            # Calculate confidence intervals
            z_score = 1.96  # 95% confidence interval
            margin_of_error = z_score * std_dev
            
            lower_bounds = predictions - margin_of_error
            upper_bounds = predictions + margin_of_error
            
            return predictions, lower_bounds, upper_bounds
            
        except Exception as e:
            logger.error(f"Error making predictions with intervals: {e}")
            raise
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        try:
            # Make predictions
            y_pred = self.predict(X_test)
            
            # Calculate metrics
            test_metrics = self._calculate_metrics(y_test, y_pred, 'test')
            
            # Store test metrics
            self.test_metrics = test_metrics
            
            logger.info("Model evaluation completed")
            return test_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        split_name: str
    ) -> Dict[str, float]:
        """Calculate performance metrics for a given split."""
        metrics = {}
        
        if self.model_type == 'regression':
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
            
        else:  # classification
            metrics['accuracy'] = np.mean(y_true == y_pred)
            
            # Detailed classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            metrics.update(report['weighted avg'])
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
                except:
                    metrics['roc_auc'] = 0.0
        
        logger.info(f"{split_name.capitalize()} metrics: {metrics}")
        return metrics
    
    def _calculate_feature_importance(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Calculate and store feature importance."""
        try:
            # Get feature importance from the model
            importance = self.model.feature_importances_
            
            # Create feature importance DataFrame
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Calculate permutation importance for more robust ranking
            perm_importance = permutation_importance(
                self.model, X_train, y_train, 
                n_repeats=10, random_state=self.random_state, n_jobs=self.n_jobs
            )
            
            self.feature_importance['permutation_importance'] = perm_importance.importances_mean
            
            # Sort by standard importance for consistency with tests
            self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)
            
            logger.info("Feature importance calculated")
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot feature importance rankings."""
        if self.feature_importance is None:
            logger.warning("Feature importance not available. Train the model first.")
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Plot standard feature importance
            top_features = self.feature_importance.head(top_n)
            ax1.barh(range(len(top_features)), top_features['importance'])
            ax1.set_yticks(range(len(top_features)))
            ax1.set_yticklabels(top_features['feature'])
            ax1.set_xlabel('Feature Importance')
            ax1.set_title('Standard Feature Importance')
            ax1.invert_yaxis()
            
            # Plot permutation importance
            ax2.barh(range(len(top_features)), top_features['permutation_importance'])
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels(top_features['feature'])
            ax2.set_xlabel('Permutation Importance')
            ax2.set_title('Permutation Feature Importance')
            ax2.invert_yaxis()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
    
    def explain_prediction(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        plot: bool = True
    ) -> Dict[str, Any]:
        """
        Explain individual predictions using SHAP values.
        
        Args:
            X: Feature matrix
            sample_idx: Index of sample to explain
            plot: Whether to plot the explanation
            
        Returns:
            Dictionary containing SHAP values and explanation
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before explaining predictions")
        
        try:
            X_scaled = self.scaler.transform(X)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_scaled)
            
            # Handle both regression and classification
            if self.model_type == 'classification':
                if len(shap_values) > 2:  # Multi-class
                    shap_values = shap_values[0]  # Use first class for simplicity
                else:  # Binary
                    shap_values = shap_values[1]  # Use positive class
            
            # Get explanation for specific sample
            sample_shap = shap_values[sample_idx]
            sample_features = X_scaled[sample_idx]
            
            explanation = {
                'shap_values': sample_shap,
                'feature_values': sample_features,
                'feature_names': self.feature_names,
                'prediction': self.model.predict(X_scaled[sample_idx:sample_idx+1])[0]
            }
            
            if plot:
                self._plot_shap_explanation(explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            raise
    
    def _plot_shap_explanation(self, explanation: Dict[str, Any]) -> None:
        """Plot SHAP explanation for a single prediction."""
        try:
            # Create feature importance plot
            feature_importance = pd.DataFrame({
                'feature': explanation['feature_names'],
                'shap_value': explanation['shap_values'],
                'feature_value': explanation['feature_values']
            }).sort_values('shap_value', key=abs, ascending=False)
            
            plt.figure(figsize=(10, 8))
            colors = ['red' if x < 0 else 'blue' for x in feature_importance['shap_value']]
            plt.barh(range(len(feature_importance)), feature_importance['shap_value'], color=colors)
            plt.yticks(range(len(feature_importance)), feature_importance['feature'])
            plt.xlabel('SHAP Value')
            plt.title(f'SHAP Explanation (Prediction: {explanation["prediction"]:.3f})')
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting SHAP explanation: {e}")
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'training_metrics': self.training_metrics,
                'validation_metrics': self.validation_metrics,
                'feature_importance': self.feature_importance
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.training_metrics = model_data['training_metrics']
            self.validation_metrics = model_data['validation_metrics']
            self.feature_importance = model_data['feature_importance']
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Generate synthetic target (regression example)
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.8 + 
         np.random.randn(n_samples) * 0.1)
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Initialize and train model
    model = RandomForestFireRiskModel(model_type='regression')
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X_df, y)
    
    # Train model
    training_results = model.train(X_train, y_train, X_val, y_val)
    print("Training completed!")
    
    # Evaluate model
    test_metrics = model.evaluate(X_test, y_test)
    print(f"Test R² Score: {test_metrics['r2']:.4f}")
    
    # Make predictions with confidence intervals
    predictions, lower, upper = model.predict_with_intervals(X_test[:5])
    print(f"Predictions with 95% CI: {predictions[:3]}")
    
    # Plot feature importance
    model.plot_feature_importance(top_n=10)
    
    print("Random Forest model demonstration completed!")
