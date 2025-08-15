"""
Crosswalk models for calibrating satellite indices with AOP ground truth.

This module implements machine learning models that learn mappings between
satellite-derived vegetation indices and high-resolution AOP measurements.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path
from typing import Dict, Union, Literal, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def fit_linear_crosswalk(X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Fit linear crosswalk model using Ridge regression.
    
    Args:
        X: Feature matrix
        y: Target variable
        
    Returns:
        Dictionary containing model, metrics, and feature importance
    """
    logger.info("Fitting linear crosswalk model")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Find optimal alpha using cross-validation
    alphas = np.logspace(-2, 3, 50)
    cv_scores = []
    
    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=42)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        cv_scores.append(scores.mean())
    
    best_alpha = alphas[np.argmax(cv_scores)]
    best_cv_score = max(cv_scores)
    
    # Fit final model
    model = Ridge(alpha=best_alpha, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    logger.info(f"Linear crosswalk - Test R²: {test_r2:.3f}, Test MAE: {test_mae:.3f}")
    
    # Feature importance (coefficients)
    feature_importance = {i: coef for i, coef in enumerate(model.coef_)}
    
    # Store metrics in model object for later access
    model.metrics_ = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'scaler': scaler
    }
    
    return {
        'model': model,
        'metrics': {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'best_alpha': best_alpha,
            'cv_r2_mean': best_cv_score,
            'cv_r2_std': np.std(cv_scores)
        },
        'feature_importance': feature_importance
    }


def fit_ensemble_crosswalk(X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
    """
    Fit ensemble crosswalk model using Gradient Boosting.
    
    Args:
        X: Feature matrix
        y: Target variable
        
    Returns:
        Trained GradientBoostingRegressor model
    """
    logger.info("Fitting ensemble crosswalk model")
    
    # Initialize model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        subsample=0.8
    )
    
    # Fit model
    model.fit(X, y)
    
    logger.info("Ensemble crosswalk model fitted successfully")
    return model


def calibrate_satellite_indices(
    satellite_df: pd.DataFrame,
    aop_df: pd.DataFrame,
    target_vars: List[str],
    model_type: Literal["linear", "ensemble"] = "linear"
) -> Dict[str, Union[Dict, GradientBoostingRegressor]]:
    """
    Calibrate satellite indices using AOP ground truth.
    
    Args:
        satellite_df: DataFrame with satellite indices
        aop_df: DataFrame with AOP ground truth
        target_vars: List of target variables to calibrate
        model_type: Type of model to use
        
    Returns:
        Dictionary of trained models for each target variable
    """
    logger.info(f"Calibrating {len(target_vars)} satellite indices with {model_type} models")
    
    # Ensure dataframes have matching indices
    if hasattr(satellite_df, 'index'):
        # DataFrame input
        common_idx = satellite_df.index.intersection(aop_df.index)
        if len(common_idx) == 0:
            raise ValueError("No common indices between satellite and AOP data")
        
        satellite_subset = satellite_df.loc[common_idx]
        aop_subset = aop_df.loc[common_idx]
        sample_count = len(common_idx)
    else:
        # Numpy array input
        if len(satellite_df) != len(aop_df):
            raise ValueError("Satellite and AOP data must have the same length")
        
        satellite_subset = satellite_df
        aop_subset = aop_df
        sample_count = len(satellite_df)
    
    logger.info(f"Using {sample_count} common samples for calibration")
    
    # Get satellite features (exclude non-numeric columns)
    if hasattr(satellite_subset, 'select_dtypes'):
        # DataFrame input
        satellite_features = satellite_subset.select_dtypes(include=[np.number])
    else:
        # Numpy array input
        satellite_features = satellite_subset
    
    # Ensure we have numeric data
    if hasattr(satellite_features, 'values'):
        X = satellite_features.values
    else:
        X = satellite_features
    
    models = {}
    
    for target_var in target_vars:
        if target_var not in aop_subset.columns:
            logger.warning(f"Target variable {target_var} not found in AOP data")
            continue
        
        logger.info(f"Calibrating {target_var}")
        
        # Prepare data
        y = aop_subset[target_var].values
        
        # Remove samples with missing values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(X_valid) < 10:
            logger.warning(f"Insufficient data for {target_var}: {len(X_valid)} samples")
            continue
        
        # Fit model
        if model_type == "linear":
            model_result = fit_linear_crosswalk(X_valid, y_valid)
            models[target_var] = model_result
        else:
            model = fit_ensemble_crosswalk(X_valid, y_valid)
            models[target_var] = model
    
    logger.info(f"Successfully calibrated {len(models)} variables")
    return models


def validate_crosswalk(
    satellite_df: pd.DataFrame,
    aop_df: pd.DataFrame,
    models: Dict,
    output_dir: Path
) -> pd.DataFrame:
    """
    Validate crosswalk models and generate metrics.
    
    Args:
        satellite_df: DataFrame with satellite indices
        aop_df: DataFrame with AOP ground truth
        models: Dictionary of trained models
        output_dir: Directory to save validation results
        
    Returns:
        DataFrame with validation metrics
    """
    logger.info("Validating crosswalk models")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validation_results = []
    
    for target_var, model_info in models.items():
        logger.info(f"Validating {target_var}")
        
        # Get satellite features
        satellite_features = satellite_df.select_dtypes(include=[np.number])
        
        # Get common samples
        common_idx = satellite_df.index.intersection(aop_df.index)
        X = satellite_features.loc[common_idx].values
        y_true = aop_df.loc[common_idx, target_var].values
        
        # Remove missing values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_true))
        X_valid = X[valid_mask]
        y_true_valid = y_true[valid_mask]
        
        if len(X_valid) == 0:
            continue
        
        # Make predictions
        if isinstance(model_info, dict) and 'model' in model_info:
            # Linear model
            model = model_info['model']
            if 'scaler' in model_info['metrics']:
                scaler = model_info['metrics']['scaler']
                X_scaled = scaler.transform(X_valid)
            else:
                # No scaler available, use raw features
                X_scaled = X_valid
            y_pred = model.predict(X_scaled)
            model_type = 'ridge'
        else:
            # Ensemble model
            model = model_info
            # For ensemble models, assume features are already scaled
            y_pred = model.predict(X_valid)
            model_type = 'gradient_boosting'
        
        # Calculate metrics
        r2 = r2_score(y_true_valid, y_pred)
        mae = mean_absolute_error(y_true_valid, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred))
        
        # Calculate bias
        bias = np.mean(y_pred - y_true_valid)
        
        # Calculate correlation
        correlation = np.corrcoef(y_true_valid, y_pred)[0, 1]
        
        # Store results
        result = {
            'target_variable': target_var,
            'model_type': model_type,
            'n_samples': len(y_true_valid),
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'bias': bias,
            'correlation': correlation
        }
        validation_results.append(result)
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true_valid, y_pred, alpha=0.6)
        plt.plot([y_true_valid.min(), y_true_valid.max()], 
                [y_true_valid.min(), y_true_valid.max()], 'r--', lw=2)
        plt.xlabel(f'AOP {target_var}')
        plt.ylabel(f'Predicted {target_var}')
        plt.title(f'Crosswalk Validation: {target_var}\nR² = {r2:.3f}, MAE = {mae:.3f}')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = output_dir / f"{target_var}_validation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"{target_var}: R² = {r2:.3f}, MAE = {mae:.3f}")
    
    # Create summary DataFrame
    results_df = pd.DataFrame(validation_results)
    
    # Save results
    results_path = output_dir / "crosswalk_validation_results.csv"
    results_df.to_csv(results_path, index=False)
    
    # Create summary plot
    if len(results_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # R² scores
        axes[0, 0].bar(results_df['target_variable'], results_df['r2'])
        axes[0, 0].set_title('R² Scores by Variable')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE scores
        axes[0, 1].bar(results_df['target_variable'], results_df['mae'])
        axes[0, 1].set_title('MAE Scores by Variable')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Model type distribution
        model_counts = results_df['model_type'].value_counts()
        axes[1, 0].pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Model Type Distribution')
        
        # Correlation vs R²
        axes[1, 1].scatter(results_df['correlation'], results_df['r2'])
        axes[1, 1].set_xlabel('Correlation')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].set_title('Correlation vs R²')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        summary_plot_path = output_dir / "crosswalk_validation_summary.png"
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Validation results saved to {output_dir}")
    return results_df


def save_crosswalk_models(
    models: Dict,
    output_dir: Path,
    model_name: str = "crosswalk_models"
) -> None:
    """
    Save crosswalk models to disk.
    
    Args:
        models: Dictionary of trained models
        output_dir: Directory to save models
        model_name: Base name for model files
    """
    logger.info(f"Saving crosswalk models to {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for target_var, model_info in models.items():
        if isinstance(model_info, dict) and 'model_type' in model_info:
            # Linear model - save as JSON
            model_data = {
                'model_type': 'ridge',
                'best_alpha': model_info['best_alpha'],
                'feature_importance': model_info['feature_importance'],
                'metrics': {
                    'train_r2': model_info['train_r2'],
                    'test_r2': model_info['test_r2'],
                    'train_mae': model_info['train_mae'],
                    'test_mae': model_info['test_mae']
                }
            }
            
            # Save model parameters as JSON
            json_path = output_dir / f"{model_name}_{target_var}_ridge.json"
            with open(json_path, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            # Save sklearn model and scaler
            model_path = output_dir / f"{model_name}_{target_var}_ridge.pkl"
            scaler_path = output_dir / f"{model_name}_{target_var}_scaler.pkl"
            
            joblib.dump(model_info['model'], model_path)
            joblib.dump(model_info['scaler'], scaler_path)
            
        else:
            # Ensemble model - save as pickle
            model_path = output_dir / f"{model_name}_{target_var}_gbm.pkl"
            joblib.dump(model_info, model_path)
    
    logger.info(f"Saved {len(models)} crosswalk models")


def load_crosswalk_models(
    model_dir: Path,
    target_vars: Optional[List[str]] = None
) -> Dict:
    """
    Load crosswalk models from disk.
    
    Args:
        model_dir: Directory containing saved models
        target_vars: List of target variables to load (if None, load all)
        
    Returns:
        Dictionary of loaded models
    """
    logger.info(f"Loading crosswalk models from {model_dir}")
    
    models = {}
    
    # Find model files
    if target_vars is None:
        # Load all available models
        model_files = list(model_dir.glob("*.pkl"))
        target_vars = list(set([f.stem.split('_')[-1] for f in model_files]))
    
    for target_var in target_vars:
        # Try to load Ridge model
        ridge_model_path = model_dir / f"crosswalk_models_{target_var}_ridge.pkl"
        ridge_scaler_path = model_dir / f"crosswalk_models_{target_var}_scaler.pkl"
        
        if ridge_model_path.exists() and ridge_scaler_path.exists():
            model = joblib.load(ridge_model_path)
            scaler = joblib.load(ridge_scaler_path)
            
            models[target_var] = {
                'model': model,
                'scaler': scaler,
                'model_type': 'ridge'
            }
            logger.info(f"Loaded Ridge model for {target_var}")
            
        else:
            # Try to load GBM model
            gbm_model_path = model_dir / f"crosswalk_models_{target_var}_gbm.pkl"
            
            if gbm_model_path.exists():
                model = joblib.load(gbm_model_path)
                models[target_var] = model
                logger.info(f"Loaded GBM model for {target_var}")
            else:
                logger.warning(f"No model found for {target_var}")
    
    logger.info(f"Loaded {len(models)} crosswalk models")
    return models


def apply_crosswalk_models(satellite_data: pd.DataFrame, 
                          crosswalk_models: Dict) -> pd.DataFrame:
    """
    Apply trained crosswalk models to satellite data.
    
    Args:
        satellite_data: DataFrame with satellite indices
        crosswalk_models: Dictionary of trained crosswalk models
        
    Returns:
        DataFrame with enhanced features
    """
    logger.info(f"Applying {len(crosswalk_models)} crosswalk models")
    
    if not crosswalk_models:
        logger.warning("No crosswalk models provided, returning original data")
        return satellite_data.copy()
    
    # Start with original data
    enhanced_data = satellite_data.copy()
    
    # Apply each model
    for target_var, model_info in crosswalk_models.items():
        try:
            if isinstance(model_info, dict) and 'model' in model_info:
                # Linear model
                model = model_info['model']
                scaler = model_info['metrics']['scaler']
                
                # Scale features
                X_scaled = scaler.transform(satellite_data.values)
                
                # Make predictions
                predictions = model.predict(X_scaled)
                
                # Add calibrated values
                enhanced_data[f"{target_var}_calibrated"] = predictions
                
            elif hasattr(model_info, 'predict'):
                # Ensemble model
                model = model_info
                
                # Make predictions (assuming features are already scaled)
                predictions = model.predict(satellite_data.values)
                
                # Add calibrated values
                enhanced_data[f"{target_var}_calibrated"] = predictions
                
            else:
                logger.warning(f"Unknown model type for {target_var}")
                
        except Exception as e:
            logger.error(f"Error applying model for {target_var}: {e}")
            # Add default values
            enhanced_data[f"{target_var}_calibrated"] = satellite_data.iloc[:, 0].values
    
    logger.info(f"Applied crosswalk models, added {len(crosswalk_models)} calibrated features")
    return enhanced_data


if __name__ == "__main__":
    # CLI interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and validate AOP crosswalk models")
    parser.add_argument("--satellite-data", required=True, help="Path to satellite data CSV")
    parser.add_argument("--aop-data", required=True, help="Path to AOP data CSV")
    parser.add_argument("--target-vars", nargs="+", required=True, help="Target variables to calibrate")
    parser.add_argument("--model-type", choices=["linear", "ensemble"], default="linear", help="Model type")
    parser.add_argument("--output-dir", required=True, help="Output directory for models and results")
    parser.add_argument("--mode", choices=["calibrate", "validate"], default="calibrate", help="Operation mode")
    
    args = parser.parse_args()
    
    # Load data
    satellite_df = pd.read_csv(args.satellite_data, index_col=0)
    aop_df = pd.read_csv(args.aop_data, index_col=0)
    
    output_dir = Path(args.output_dir)
    
    if args.mode == "calibrate":
        # Train models
        models = calibrate_satellite_indices(
            satellite_df, aop_df, args.target_vars, args.model_type
        )
        
        # Save models
        save_crosswalk_models(models, output_dir)
        
        # Validate models
        validation_results = validate_crosswalk(satellite_df, aop_df, models, output_dir)
        
        print(f"Calibration complete. Models saved to {output_dir}")
        print(f"Validation results: {validation_results}")
        
    else:
        # Load existing models and validate
        models = load_crosswalk_models(output_dir, args.target_vars)
        validation_results = validate_crosswalk(satellite_df, aop_df, models, output_dir)
        
        print(f"Validation complete. Results saved to {output_dir}")
        print(f"Validation results: {validation_results}")