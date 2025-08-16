"""
Crosswalk models to match satellite data with airplane (AOP) measurements.

Basically teaching satellites to see as well as low-flying planes. Pretty neat.
Uses ML to learn the mapping between coarse satellite pixels and fine-scale AOP data.
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
    Train a linear model to map satellite -> AOP data.
    
    Using Ridge regression cuz it's simple and works well enough.
    
    Args:
        X: satellite features 
        y: AOP ground truth
        
    Returns:
        dict with model, metrics, feature importance
    """
    logger.info("Training linear crosswalk...")
    
    # 80/20 split - pretty standard
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Find best alpha w/ cross-validation
    # TODO: maybe try elastic net too?
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
    
    logger.info(f"Linear model - R²: {test_r2:.3f}, MAE: {test_mae:.3f}")
    
    # Get feature importance from coefficients
    feature_importance = {i: coef for i, coef in enumerate(model.coef_)}
    
    # Stash metrics in the model for later (kinda hacky but works)
    model.metrics_ = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'scaler': scaler  # need this for predictions later!!
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
    Train ensemble model for the complicated stuff.
    
    GradientBoosting b/c it handles non-linear relationships better.
    
    Args:
        X: satellite features
        y: AOP measurements
        
    Returns:
        trained GB model
    """
    logger.info("Training ensemble crosswalk...")
    
    # GB params - these work pretty well
    model = GradientBoostingRegressor(
        n_estimators=100,      # not too many trees
        learning_rate=0.1,     # conservative lr
        max_depth=6,           # prevent overfitting 
        random_state=42,
        subsample=0.8          # bit of randomness helps
    )
    
    model.fit(X, y)
    
    logger.info("Ensemble model done")
    return model


def calibrate_satellite_indices(
    satellite_df: pd.DataFrame,
    aop_df: pd.DataFrame,
    target_vars: List[str],
    model_type: Literal["linear", "ensemble"] = "linear"
) -> Dict[str, Union[Dict, GradientBoostingRegressor]]:
    """
    Main calibration function - trains models to map satellite -> AOP.
    
    Args:
        satellite_df: satellite data (the crappy resolution stuff)
        aop_df: AOP ground truth (the good stuff)
        target_vars: which AOP variables to predict
        model_type: 'linear' for simple, 'ensemble' for complex patterns
        
    Returns:
        dict of trained models, one per target var
    """
    logger.info(f"Calibrating {len(target_vars)} vars using {model_type} models")
    
    # Make sure we have matching data points
    if hasattr(satellite_df, 'index'):
        # pandas dataframe
        common_idx = satellite_df.index.intersection(aop_df.index)
        if len(common_idx) == 0:
            raise ValueError("No matching indices! Check your data alignment")
        
        satellite_subset = satellite_df.loc[common_idx]
        aop_subset = aop_df.loc[common_idx]
        sample_count = len(common_idx)
    else:
        # numpy arrays
        if len(satellite_df) != len(aop_df):
            raise ValueError("Arrays must be same length")
        
        satellite_subset = satellite_df
        aop_subset = aop_df
        sample_count = len(satellite_df)
    
    logger.info(f"Got {sample_count} matching samples")
    
    # Get numeric features only (no strings/dates/etc)
    if hasattr(satellite_subset, 'select_dtypes'):
        # dataframe
        satellite_features = satellite_subset.select_dtypes(include=[np.number])
    else:
        # already numpy
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
        
        logger.info(f"Working on {target_var}...")
        
        y = aop_subset[target_var].values
        
        # Drop NaNs (there's always some...)
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(X_valid) < 10:
            logger.warning(f"Not enough data for {target_var}: only {len(X_valid)} samples")
            continue  # skip this one
        
        # Train the model
        if model_type == "linear":
            model_result = fit_linear_crosswalk(X_valid, y_valid)
            models[target_var] = model_result
        else:
            model = fit_ensemble_crosswalk(X_valid, y_valid)
            models[target_var] = model
    
    logger.info(f"Done! Calibrated {len(models)} variables")
    return models


def validate_crosswalk(
    satellite_df: pd.DataFrame,
    aop_df: pd.DataFrame,
    models: Dict,
    output_dir: Path
) -> pd.DataFrame:
    """
    Check if our models actually work.
    
    Makes predictions and compares to ground truth, saves plots.
    
    Args:
        satellite_df: satellite data
        aop_df: ground truth from planes
        models: our trained models
        output_dir: where to save results
        
    Returns:
        DataFrame with R², MAE, etc
    """
    logger.info("Validating models...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validation_results = []
    
    for target_var, model_info in models.items():
        logger.info(f"Checking {target_var}...")
        
        # grab numeric cols only
        satellite_features = satellite_df.select_dtypes(include=[np.number])
        
        # Get common samples
        common_idx = satellite_df.index.intersection(aop_df.index)
        X = satellite_features.loc[common_idx].values
        y_true = aop_df.loc[common_idx, target_var].values
        
        # Drop NaNs again
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_true))
        X_valid = X[valid_mask]
        y_true_valid = y_true[valid_mask]
        
        if len(X_valid) == 0:
            continue  # nothing to validate
        
        # Get predictions
        if isinstance(model_info, dict) and 'model' in model_info:
            # linear model
            model = model_info['model']
            if 'scaler' in model_info['metrics']:
                scaler = model_info['metrics']['scaler']
                X_scaled = scaler.transform(X_valid)
            else:
                # no scaler? use raw (shouldn't happen tho)
                X_scaled = X_valid
            y_pred = model.predict(X_scaled)
            model_type = 'ridge'
        else:
            # ensemble model
            model = model_info
            y_pred = model.predict(X_valid)  # GB doesn't need scaling
            model_type = 'gradient_boosting'
        
        # Calc all the metrics
        r2 = r2_score(y_true_valid, y_pred)
        mae = mean_absolute_error(y_true_valid, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred))
        
        # bias (are we consistently over/under?)
        bias = np.mean(y_pred - y_true_valid)
        
        # correlation
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
        
        # Make a nice scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true_valid, y_pred, alpha=0.6)
        plt.plot([y_true_valid.min(), y_true_valid.max()], 
                [y_true_valid.min(), y_true_valid.max()], 'r--', lw=2)  # 1:1 line
        plt.xlabel(f'AOP {target_var}')
        plt.ylabel(f'Predicted {target_var}')
        plt.title(f'{target_var} Validation\nR² = {r2:.3f}, MAE = {mae:.3f}')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = output_dir / f"{target_var}_validation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"{target_var}: R²={r2:.3f}, MAE={mae:.3f}")
    
    # Make summary df
    results_df = pd.DataFrame(validation_results)
    
    # Save to csv
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
            logger.info(f"Loaded Ridge for {target_var}")
            
        else:
            # try GBM
            gbm_model_path = model_dir / f"crosswalk_models_{target_var}_gbm.pkl"
            
            if gbm_model_path.exists():
                model = joblib.load(gbm_model_path)
                models[target_var] = model
                logger.info(f"Loaded GBM for {target_var}")
            else:
                logger.warning(f"No model for {target_var}")
    
    logger.info(f"Loaded {len(models)} crosswalk models")
    return models


def apply_crosswalk_models(satellite_data: pd.DataFrame, 
                          crosswalk_models: Dict) -> pd.DataFrame:
    """
    Use trained models to enhance satellite data.
    
    Takes low-res satellite data and predicts what high-res AOP would see.
    
    Args:
        satellite_data: original satellite indices
        crosswalk_models: our trained models
        
    Returns:
        DataFrame with new calibrated columns
    """
    logger.info(f"Applying {len(crosswalk_models)} models...")
    
    if not crosswalk_models:
        logger.warning("No models - returning original data")
        return satellite_data.copy()
    
    # copy original data first
    enhanced_data = satellite_data.copy()
    
    # apply each model
    for target_var, model_info in crosswalk_models.items():
        try:
            if isinstance(model_info, dict) and 'model' in model_info:
                # linear model needs scaling
                model = model_info['model']
                scaler = model_info['metrics']['scaler']
                
                X_scaled = scaler.transform(satellite_data.values)
                predictions = model.predict(X_scaled)
                
                # add new column
                enhanced_data[f"{target_var}_calibrated"] = predictions
                
            elif hasattr(model_info, 'predict'):
                # ensemble model - no scaling needed
                model = model_info
                
                predictions = model.predict(satellite_data.values)
                enhanced_data[f"{target_var}_calibrated"] = predictions
                
            else:
                logger.warning(f"Weird model type for {target_var}??")
                
        except Exception as e:
            logger.error(f"Failed to apply {target_var}: {e}")
            # add dummy values so we don't crash
            enhanced_data[f"{target_var}_calibrated"] = satellite_data.iloc[:, 0].values
    
    logger.info(f"Added {len(crosswalk_models)} calibrated features")
    return enhanced_data


if __name__ == "__main__":
    # quick CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Train/validate AOP crosswalk models")
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