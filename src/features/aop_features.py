"""
Extract features from NEON AOP data for fire risk modeling.

This module provides comprehensive feature extraction from NEON Airborne Observation
Platform data including canopy height models, hyperspectral reflectance, and
derived vegetation indices.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import rasterio as rio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import geopandas as gpd
from skimage.feature import graycomatrix, graycoprops
import logging

logger = logging.getLogger(__name__)


@dataclass
class AOPBundle:
    """Container for AOP data paths and metadata."""
    site: str
    year: int
    chm_path: Optional[str] = None
    hsi_path: Optional[str] = None
    crs: Optional[str] = None
    bounds: Optional[Tuple] = None


def extract_chm_features(chm_arr: np.ndarray, heights: List[float] = [2, 5]) -> Dict[str, float]:
    """
    Extract canopy height model statistics.
    
    Args:
        chm_arr: Canopy height model array
        heights: Height thresholds for canopy cover calculation
        
    Returns:
        Dictionary of CHM features
    """
    logger.info("Extracting CHM features")
    
    # Remove invalid values
    valid_heights = chm_arr[~np.isnan(chm_arr) & (chm_arr > 0)]
    
    if len(valid_heights) == 0:
        logger.warning("No valid heights found in CHM data")
        return {
            'chm_mean': 0.0,
            'chm_std': 0.0,
            'chm_p10': 0.0,
            'chm_p25': 0.0,
            'chm_p50': 0.0,
            'chm_p75': 0.0,
            'chm_p90': 0.0,
            'chm_max': 0.0,
            'canopy_cover_gt2m': 0.0,
            'canopy_cover_gt5m': 0.0,
            'rumple_index': 0.0,
            'height_entropy': 0.0
        }
    
    # Basic statistics
    features = {
        'chm_mean': float(np.mean(valid_heights)),
        'chm_std': float(np.std(valid_heights)),
        'chm_p10': float(np.percentile(valid_heights, 10)),
        'chm_p25': float(np.percentile(valid_heights, 25)),
        'chm_p50': float(np.percentile(valid_heights, 50)),
        'chm_p75': float(np.percentile(valid_heights, 75)),
        'chm_p90': float(np.percentile(valid_heights, 90)),
        'chm_max': float(np.max(valid_heights))
    }
    
    # Canopy cover at different heights
    total_pixels = len(valid_heights)
    for height in heights:
        cover = np.sum(valid_heights >= height) / total_pixels
        features[f'canopy_cover_gt{int(height)}m'] = float(cover)
    
    # Complexity metrics
    features['rumple_index'] = float(np.sum(np.abs(np.diff(valid_heights))) / len(valid_heights))
    
    # Height entropy (measure of height diversity)
    hist, _ = np.histogram(valid_heights, bins=20)
    hist = hist[hist > 0] / hist.sum()
    features['height_entropy'] = float(-np.sum(hist * np.log2(hist)))
    
    logger.info(f"Extracted {len(features)} CHM features")
    return features


def extract_spectral_features(hsi_arr: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Extract spectral features from hyperspectral imagery.
    
    Args:
        hsi_arr: Hyperspectral image array (height, width, bands)
        wavelengths: Wavelength values for each band (optional)
        
    Returns:
        Dictionary of spectral features
    """
    logger.info("Extracting spectral features from hyperspectral imagery")
    
    # If no wavelengths provided, use default band indices
    if wavelengths is None:
        wavelengths = np.arange(hsi_arr.shape[2])
    
    # Calculate vegetation indices
    features = {}
    
    # NDVI (using bands around 800nm and 680nm if available)
    if hsi_arr.shape[2] >= 2:
        # Use first and last bands as approximation
        red_band = hsi_arr[:, :, 0]
        nir_band = hsi_arr[:, :, -1]
        
        ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
        features['ndvi_aop'] = float(np.nanmean(ndvi))
    
    # EVI
    if hsi_arr.shape[2] >= 3:
        blue_band = hsi_arr[:, :, 1]
        red_band = hsi_arr[:, :, 0]
        nir_band = hsi_arr[:, :, -1]
        
        # Ensure bands are in valid range and calculate EVI with bounds checking
        evi = 2.5 * (nir_band - red_band) / (nir_band + 6 * red_band - 7.5 * blue_band + 1e-8)
        # Clip EVI to valid range [-1, 1]
        evi = np.clip(evi, -1, 1)
        features['evi_aop'] = float(np.nanmean(evi))
    
    # NBR
    if hsi_arr.shape[2] >= 2:
        nir_band = hsi_arr[:, :, -1]
        swir_band = hsi_arr[:, :, hsi_arr.shape[2] // 2]  # Use middle band as SWIR approximation
        
        nbr = (nir_band - swir_band) / (nir_band + swir_band + 1e-8)
        features['nbr_aop'] = float(np.nanmean(nbr))
    
    # NDWI
    if hsi_arr.shape[2] >= 2:
        green_band = hsi_arr[:, :, 1]
        nir_band = hsi_arr[:, :, -1]
        
        ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-8)
        features['ndwi_aop'] = float(np.nanmean(ndwi))
    
    # Add basic statistics
    features['spectral_mean'] = float(np.nanmean(hsi_arr))
    features['spectral_std'] = float(np.nanstd(hsi_arr))
    features['spectral_min'] = float(np.nanmin(hsi_arr))
    features['spectral_max'] = float(np.nanmax(hsi_arr))
    
    logger.info(f"Extracted {len(features)} spectral features")
    return features


def process_aop_to_grid(
    aop_bundle: AOPBundle,
    grid_spec: str,
    output_dir: Path
) -> gpd.GeoDataFrame:
    """
    Process AOP data to target grid and export features.
    
    Args:
        aop_bundle: AOP data bundle
        grid_spec: Target grid specification
        output_dir: Output directory for processed features
        
    Returns:
        GeoDataFrame with extracted features
    """
    logger.info(f"Processing AOP data for {aop_bundle.site} {aop_bundle.year}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_list = []
    
    # Process CHM data if available
    if aop_bundle.chm_path and Path(aop_bundle.chm_path).exists():
        logger.info(f"Processing CHM data: {aop_bundle.chm_path}")
        
        with rio.open(aop_bundle.chm_path) as src:
            chm_data = src.read(1)  # Read first band
            
            # Extract CHM features
            chm_features = extract_chm_features(chm_data)
            
            # Add metadata
            chm_features.update({
                'site': aop_bundle.site,
                'year': aop_bundle.year,
                'data_type': 'chm',
                'source_file': aop_bundle.chm_path
            })
            
            features_list.append(chm_features)
    
    # Process hyperspectral data if available
    if aop_bundle.hsi_path and Path(aop_bundle.hsi_path).exists():
        logger.info(f"Processing hyperspectral data: {aop_bundle.hsi_path}")
        
        with rio.open(aop_bundle.hsi_path) as src:
            hsi_data = src.read()  # Read all bands
            
            # Create wavelength array (placeholder - should come from metadata)
            # For NEON AOP, typically 426 bands from 380-2510nm
            wavelengths = np.linspace(380, 2510, hsi_data.shape[0])
            
            # Extract spectral features
            spectral_features = extract_spectral_features(hsi_data, wavelengths)
            
            # Add metadata
            spectral_features.update({
                'site': aop_bundle.site,
                'year': aop_bundle.year,
                'data_type': 'hyperspectral',
                'source_file': aop_bundle.hsi_path
            })
            
            features_list.append(spectral_features)
    
    if not features_list:
        logger.warning(f"No valid AOP data found for {aop_bundle.site} {aop_bundle.year}")
        return gpd.GeoDataFrame()
    
    # Combine features
    combined_features = {}
    for feature_dict in features_list:
        combined_features.update(feature_dict)
    
    # Create GeoDataFrame
    # For now, create a simple point geometry at the center of the AOI
    if aop_bundle.bounds:
        center_lon = (aop_bundle.bounds[0] + aop_bundle.bounds[2]) / 2
        center_lat = (aop_bundle.bounds[1] + aop_bundle.bounds[3]) / 2
        geometry = gpd.points_from_xy([center_lon], [center_lat])
    else:
        # Default to site center if bounds not provided
        geometry = gpd.points_from_xy([0], [0])  # Placeholder
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame([combined_features], geometry=geometry, crs="EPSG:4326")
    
    # Save to file
    output_file = output_dir / f"{aop_bundle.site}_{aop_bundle.year}_features.gpkg"
    gdf.to_file(output_file, driver="GPKG")
    
    logger.info(f"Saved features to {output_file}")
    return gdf


def create_aop_bundle(
    site: str,
    year: int,
    data_dir: Path,
    chm_filename: Optional[str] = None,
    hsi_filename: Optional[str] = None
) -> AOPBundle:
    """
    Create an AOP bundle from data directory.
    
    Args:
        site: NEON site code
        year: Data year
        data_dir: Directory containing AOP data
        chm_filename: Optional CHM filename
        hsi_filename: Optional hyperspectral filename
        
    Returns:
        AOPBundle object
    """
    logger.info(f"Creating AOP bundle for {site} {year}")
    
    # Look for data files if not specified
    if chm_filename is None:
        chm_pattern = f"*{site}*{year}*CHM*.tif"
        chm_files = list(data_dir.glob(chm_pattern))
        chm_path = str(chm_files[0]) if chm_files else None
    else:
        chm_path = str(data_dir / chm_filename)
    
    if hsi_filename is None:
        hsi_pattern = f"*{site}*{year}*HSI*.tif"
        hsi_files = list(data_dir.glob(hsi_pattern))
        hsi_path = str(hsi_files[0]) if hsi_files else None
    else:
        hsi_path = str(data_dir / hsi_filename)
    
    # Get bounds from first available file
    bounds = None
    crs = None
    
    if chm_path and Path(chm_path).exists():
        with rio.open(chm_path) as src:
            bounds = src.bounds
            crs = src.crs
    elif hsi_path and Path(hsi_path).exists():
        with rio.open(hsi_path) as src:
            bounds = src.bounds
            crs = src.crs
    
    bundle = AOPBundle(
        site=site,
        year=year,
        chm_path=chm_path,
        hsi_path=hsi_path,
        crs=crs,
        bounds=bounds
    )
    
    logger.info(f"Created AOP bundle: CHM={chm_path is not None}, HSI={hsi_path is not None}")
    return bundle


if __name__ == "__main__":
    # CLI interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features from NEON AOP data")
    parser.add_argument("--site", required=True, help="NEON site code")
    parser.add_argument("--year", type=int, required=True, help="Data year")
    parser.add_argument("--data-dir", required=True, help="Directory containing AOP data")
    parser.add_argument("--output-dir", required=True, help="Output directory for features")
    parser.add_argument("--chm-file", help="CHM filename (optional)")
    parser.add_argument("--hsi-file", help="Hyperspectral filename (optional)")
    
    args = parser.parse_args()
    
    # Create bundle
    bundle = create_aop_bundle(
        site=args.site,
        year=args.year,
        data_dir=Path(args.data_dir),
        chm_filename=args.chm_file,
        hsi_filename=args.hsi_file
    )
    
    # Process data
    output_dir = Path(args.output_dir)
    features_df = process_aop_to_grid(bundle, "sentinel2_10m", output_dir)
    
    print(f"Processed {len(features_df)} feature sets")
    print(f"Features saved to: {output_dir}")