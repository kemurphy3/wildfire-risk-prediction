"""Crosswalk models for calibrating satellite indices with AOP ground truth."""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path
from typing import Dict, Union, Literal, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import geopandas as gpd

logger = logging.getLogger(__name__)


class CrosswalkModel:
    """Base class for satellite-AOP crosswalk models."""
    
    def __init__(self, model_type: str = "linear"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = None
        self.metrics = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], target_name: str):
        """Fit the crosswalk model."""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply the crosswalk model."""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == "linear":
            # Save as JSON for transparency
            model_dict = {
                "model_type": self.model_type,
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "coefficients": self.model.coef_.tolist(),
                "intercept": float(self.model.intercept_),
                "scaler_mean": self.scaler.mean_.tolist(),
                "scaler_scale": self.scaler.scale_.tolist(),
                "metrics": self.metrics,
                "timestamp": datetime.now().isoformat()
            }
            with open(path.with_suffix('.json'), 'w') as f:
                json.dump(model_dict, f, indent=2)
        else:
            # Save as pickle for complex models
            model_dict = {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "metrics": self.metrics,
                "model_type": self.model_type
            }
            joblib.dump(model_dict, path.with_suffix('.pkl'))
            
    @classmethod
    def load(cls, path: Path):
        """Load model from disk."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path) as f:
                model_dict = json.load(f)
            
            instance = cls(model_type=model_dict["model_type"])
            instance.feature_names = model_dict["feature_names"]
            instance.target_name = model_dict["target_name"]
            instance.metrics = model_dict["metrics"]
            
            # Reconstruct linear model
            instance.model = LinearRegression()
            instance.model.coef_ = np.array(model_dict["coefficients"])
            instance.model.intercept_ = model_dict["intercept"]
            
            # Reconstruct scaler
            instance.scaler.mean_ = np.array(model_dict["scaler_mean"])
            instance.scaler.scale_ = np.array(model_dict["scaler_scale"])
            instance.scaler.n_features_in_ = len(instance.feature_names)
            
        else:
            model_dict = joblib.load(path)
            instance = cls(model_type=model_dict["model_type"])
            instance.model = model_dict["model"]
            instance.scaler = model_dict["scaler"]
            instance.feature_names = model_dict["feature_names"]
            instance.target_name = model_dict["target_name"]
            instance.metrics = model_dict["metrics"]
            
        return instance


class LinearCrosswalk(CrosswalkModel):
    """Linear crosswalk model with L2 regularization."""
    
    def __init__(self, alphas: Optional[np.ndarray] = None):
        super().__init__(model_type="linear")
        if alphas is None:
            alphas = np.logspace(-3, 3, 13)
        self.alphas = alphas
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], target_name: str):
        """Fit Ridge regression with cross-validation."""
        self.feature_names = feature_names
        self.target_name = target_name
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model with CV
        self.model = RidgeCV(alphas=self.alphas, cv=5)
        self.model.fit(X_scaled, y)
        
        # Calculate metrics
        y_pred = self.model.predict(X_scaled)
        self.metrics = {
            "r2": float(r2_score(y, y_pred)),
            "mae": float(mean_absolute_error(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "alpha": float(self.model.alpha_),
            "n_samples": len(y)
        }
        
        logger.info(f"Linear crosswalk R²={self.metrics['r2']:.3f}, MAE={self.metrics['mae']:.3f}")
        
        return self


class EnsembleCrosswalk(CrosswalkModel):
    """Ensemble crosswalk model using gradient boosting."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5):
        super().__init__(model_type="ensemble")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], target_name: str):
        """Fit gradient boosting model."""
        self.feature_names = feature_names
        self.target_name = target_name
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        # Calculate metrics with cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y, 
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring='r2'
        )
        
        y_pred = self.model.predict(X_scaled)
        self.metrics = {
            "r2": float(r2_score(y, y_pred)),
            "r2_cv": float(np.mean(cv_scores)),
            "r2_cv_std": float(np.std(cv_scores)),
            "mae": float(mean_absolute_error(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "n_samples": len(y),
            "feature_importance": dict(zip(feature_names, self.model.feature_importances_.tolist()))
        }
        
        logger.info(f"Ensemble crosswalk R²={self.metrics['r2']:.3f}, CV R²={self.metrics['r2_cv']:.3f}")
        
        return self


def prepare_crosswalk_data(
    satellite_df: pd.DataFrame,
    aop_df: pd.DataFrame,
    join_tolerance: float = 15.0
) -> pd.DataFrame:
    """Prepare data for crosswalk by spatially joining satellite and AOP data.
    
    Args:
        satellite_df: DataFrame with satellite-derived indices
        aop_df: DataFrame with AOP-derived features
        join_tolerance: Maximum distance (meters) for spatial join
        
    Returns:
        Joined DataFrame
    """
    # Ensure both have geometry columns
    if 'geometry' not in satellite_df.columns:
        satellite_gdf = gpd.GeoDataFrame(
            satellite_df,
            geometry=gpd.points_from_xy(satellite_df.x, satellite_df.y),
            crs='EPSG:32610'  # Assuming UTM Zone 10N
        )
    else:
        satellite_gdf = gpd.GeoDataFrame(satellite_df)
    
    if 'geometry' not in aop_df.columns:
        aop_gdf = gpd.GeoDataFrame(
            aop_df,
            geometry=gpd.points_from_xy(aop_df.x, aop_df.y),
            crs='EPSG:32610'
        )
    else:
        aop_gdf = gpd.GeoDataFrame(aop_df)
    
    # Ensure same CRS
    if satellite_gdf.crs != aop_gdf.crs:
        aop_gdf = aop_gdf.to_crs(satellite_gdf.crs)
    
    # Spatial join with tolerance
    joined = gpd.sjoin_nearest(
        satellite_gdf, 
        aop_gdf,
        max_distance=join_tolerance,
        how='inner',
        rsuffix='_aop'
    )
    
    # Remove duplicate columns
    cols_to_drop = [col for col in joined.columns if col.endswith('_aop') and 
                    col.replace('_aop', '') in joined.columns]
    joined = joined.drop(columns=cols_to_drop)
    
    logger.info(f"Joined {len(joined)} matching pixels from {len(satellite_gdf)} satellite and {len(aop_gdf)} AOP pixels")
    
    return joined


def fit_linear_crosswalk(X_sat: np.ndarray, y_aop: np.ndarray) -> Dict:
    """Fit regularized linear model for satellite-AOP mapping.
    
    Simple implementation for backward compatibility.
    """
    model = LinearCrosswalk()
    model.fit(X_sat, y_aop, [f"feat_{i}" for i in range(X_sat.shape[1])], "target")
    
    return {
        "coef": model.model.coef_.tolist(),
        "intercept": float(model.model.intercept_),
        "r2": model.metrics["r2"],
        "mae": model.metrics["mae"]
    }


def predict_linear_crosswalk(model_dict: Dict, X_sat: np.ndarray) -> np.ndarray:
    """Apply linear crosswalk model.
    
    Simple implementation for backward compatibility.
    """
    coef = np.array(model_dict["coef"])
    intercept = model_dict["intercept"]
    return X_sat @ coef + intercept


def calibrate_satellite_indices(
    satellite_df: pd.DataFrame,
    aop_df: pd.DataFrame,
    target_vars: List[str],
    satellite_features: Optional[List[str]] = None,
    model_type: Literal["linear", "ensemble"] = "linear"
) -> Dict[str, Union[CrosswalkModel, Dict]]:
    """Calibrate satellite indices using AOP ground truth.
    
    Args:
        satellite_df: DataFrame with satellite data
        aop_df: DataFrame with AOP data
        target_vars: List of AOP variables to predict
        satellite_features: List of satellite features to use (None = auto-detect)
        model_type: Type of model to fit
        
    Returns:
        Dictionary mapping target variables to fitted models
    """
    # Join datasets
    data = prepare_crosswalk_data(satellite_df, aop_df)
    
    if len(data) == 0:
        raise ValueError("No matching pixels found between satellite and AOP data")
    
    # Auto-detect satellite features if not provided
    if satellite_features is None:
        satellite_features = [col for col in data.columns if 
                            any(idx in col for idx in ['ndvi', 'nbr', 'ndwi', 'evi', 'savi']) and
                            '_aop' not in col]
    
    # Filter to valid features
    satellite_features = [f for f in satellite_features if f in data.columns]
    target_vars = [t for t in target_vars if t in data.columns]
    
    if not satellite_features:
        raise ValueError("No valid satellite features found")
    if not target_vars:
        raise ValueError("No valid target variables found")
    
    logger.info(f"Using {len(satellite_features)} satellite features to predict {len(target_vars)} AOP targets")
    
    models = {}
    
    for target in target_vars:
        logger.info(f"Calibrating model for {target}")
        
        # Prepare data
        valid_mask = data[satellite_features + [target]].notna().all(axis=1)
        X = data.loc[valid_mask, satellite_features].values
        y = data.loc[valid_mask, target].values
        
        if len(X) < 50:
            logger.warning(f"Only {len(X)} samples for {target}, skipping")
            continue
        
        # Fit model
        if model_type == "linear":
            model = LinearCrosswalk()
        else:
            model = EnsembleCrosswalk()
        
        model.fit(X, y, satellite_features, target)
        models[target] = model
    
    return models


def validate_crosswalk(
    satellite_df: pd.DataFrame,
    aop_df: pd.DataFrame,
    models: Dict[str, CrosswalkModel],
    output_dir: Path,
    satellite_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """Validate crosswalk models and generate metrics.
    
    Args:
        satellite_df: Satellite data for validation
        aop_df: AOP ground truth data
        models: Dictionary of fitted crosswalk models
        output_dir: Directory to save plots and metrics
        satellite_features: List of satellite features (None = auto-detect)
        
    Returns:
        DataFrame with validation metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Join datasets
    data = prepare_crosswalk_data(satellite_df, aop_df)
    
    # Auto-detect features if needed
    if satellite_features is None:
        # Use features from first model
        first_model = next(iter(models.values()))
        satellite_features = first_model.feature_names
    
    metrics_list = []
    
    for target, model in models.items():
        if target not in data.columns:
            logger.warning(f"Target {target} not in validation data")
            continue
        
        # Prepare data
        valid_mask = data[satellite_features + [target]].notna().all(axis=1)
        X = data.loc[valid_mask, satellite_features].values
        y_true = data.loc[valid_mask, target].values
        
        if len(X) == 0:
            continue
        
        # Predict
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = {
            "target": target,
            "model_type": model.model_type,
            "n_samples": len(y_true),
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "bias": np.mean(y_pred - y_true),
            "correlation": np.corrcoef(y_true, y_pred)[0, 1]
        }
        metrics_list.append(metrics)
        
        # Generate validation plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Scatter plot
        ax = axes[0]
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax.set_xlabel(f"AOP {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"R² = {metrics['r2']:.3f}, MAE = {metrics['mae']:.3f}")
        
        # Residual plot
        ax = axes[1]
        residuals = y_pred - y_true
        ax.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel(f"Predicted {target}")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        
        # Histogram of residuals
        ax = axes[2]
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Bias = {metrics['bias']:.3f}")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"validation_{target}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance plot for ensemble models
        if hasattr(model.model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame({
                'feature': model.feature_names,
                'importance': model.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance for {target}')
            plt.tight_layout()
            plt.savefig(output_dir / f"feature_importance_{target}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(output_dir / "validation_metrics.csv", index=False)
    
    # Summary plot
    if len(metrics_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df.set_index('target')[['r2', 'correlation']].plot(kind='bar', ax=ax)
        ax.set_ylabel('Score')
        ax.set_title('Crosswalk Model Performance')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "performance_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return metrics_df


def apply_crosswalk_to_satellite(
    satellite_df: pd.DataFrame,
    models: Dict[str, CrosswalkModel],
    satellite_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """Apply crosswalk models to satellite data to generate synthetic AOP features.
    
    Args:
        satellite_df: Satellite data
        models: Fitted crosswalk models
        satellite_features: List of satellite features to use
        
    Returns:
        DataFrame with original + synthetic AOP features
    """
    result_df = satellite_df.copy()
    
    # Auto-detect features if needed
    if satellite_features is None and models:
        first_model = next(iter(models.values()))
        satellite_features = first_model.feature_names
    
    # Apply each model
    for target, model in models.items():
        # Check if we have required features
        missing_features = set(model.feature_names) - set(satellite_df.columns)
        if missing_features:
            logger.warning(f"Missing features for {target}: {missing_features}")
            continue
        
        # Prepare data
        X = satellite_df[model.feature_names].values
        
        # Handle missing values
        valid_mask = ~np.isnan(X).any(axis=1)
        predictions = np.full(len(satellite_df), np.nan)
        
        if valid_mask.any():
            predictions[valid_mask] = model.predict(X[valid_mask])
        
        # Add to dataframe
        result_df[f"{target}_synthetic"] = predictions
    
    return result_df


def main():
    """CLI interface for crosswalk calibration and validation."""
    import argparse
    import yaml
    from ..data_collection import satellite_client, neon_client
    
    parser = argparse.ArgumentParser(description="Calibrate satellite-AOP crosswalk models")
    parser.add_argument('--sites', required=True, help='Comma-separated list of NEON sites')
    parser.add_argument('--years', required=True, help='Comma-separated list of years')
    parser.add_argument('--mode', choices=['calibrate', 'validate', 'apply'], default='calibrate')
    parser.add_argument('--model-type', choices=['linear', 'ensemble'], default='linear')
    parser.add_argument('--output', default='data/models/aop_crosswalk/', help='Output directory')
    parser.add_argument('--config', default='configs/aop_sites.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    sites = args.sites.split(',')
    years = [int(y) for y in args.years.split(',')]
    output_dir = Path(args.output)
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.mode == 'calibrate':
        # Calibrate models for each site
        for site in sites:
            logger.info(f"Calibrating crosswalk for {site}")
            
            # TODO: Load actual satellite and AOP data
            # This would integrate with existing data loading infrastructure
            
            # Placeholder for demonstration
            logger.warning("Using placeholder data - implement actual data loading")
            
    elif args.mode == 'validate':
        # Validate existing models
        for site in sites:
            model_path = output_dir / f"crosswalk_{site}.json"
            if model_path.exists():
                model = CrosswalkModel.load(model_path)
                logger.info(f"Loaded model for {site}: {model.metrics}")
            else:
                logger.warning(f"No model found for {site}")
    
    else:  # apply
        logger.info("Applying crosswalk models to generate synthetic AOP features")
        # TODO: Implement application mode


if __name__ == "__main__":
    main()