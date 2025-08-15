"""Extract features from NEON AOP data for fire risk modeling."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import rasterio as rio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import geopandas as gpd
from shapely.geometry import box, Point
from skimage.feature import graycomatrix, graycoprops
import logging
from datetime import datetime
import json

from ..utils.geoalign import warp_to_grid, infer_dst_grid

logger = logging.getLogger(__name__)


@dataclass
class AOPBundle:
    """Container for AOP data paths and metadata."""
    site: str
    year: int
    chm_path: Optional[str] = None
    hsi_path: Optional[str] = None
    rgb_path: Optional[str] = None
    crs: Optional[str] = None
    bounds: Optional[Tuple] = None


def extract_chm_features(chm_arr: np.ndarray, heights: List[float] = [2, 5]) -> Dict[str, float]:
    """Extract canopy height model statistics.
    
    Args:
        chm_arr: Canopy height model array (meters)
        heights: Height thresholds for canopy cover calculation
        
    Returns:
        Dictionary of CHM-derived features
    """
    # Handle invalid values
    valid_mask = ~np.isnan(chm_arr) & (chm_arr >= 0)
    valid_chm = chm_arr[valid_mask]
    
    if len(valid_chm) == 0:
        logger.warning("No valid CHM values found")
        return {f"chm_{k}": np.nan for k in ["p10", "p50", "p90", "mean", "std"] + 
                [f"cover_gt{h}m" for h in heights]}
    
    # Basic statistics
    features = {
        "chm_p10": float(np.percentile(valid_chm, 10)),
        "chm_p25": float(np.percentile(valid_chm, 25)),
        "chm_p50": float(np.percentile(valid_chm, 50)),
        "chm_p75": float(np.percentile(valid_chm, 75)),
        "chm_p90": float(np.percentile(valid_chm, 90)),
        "chm_mean": float(np.mean(valid_chm)),
        "chm_std": float(np.std(valid_chm)),
        "chm_max": float(np.max(valid_chm)),
        "chm_cv": float(np.std(valid_chm) / np.mean(valid_chm)) if np.mean(valid_chm) > 0 else 0
    }
    
    # Canopy cover at different heights
    for h in heights:
        cover_frac = float(np.sum(valid_chm > h) / len(valid_chm))
        features[f"canopy_cover_gt{h}m"] = cover_frac
    
    # Canopy complexity metrics
    features["chm_rumple"] = calculate_rumple_index(chm_arr)
    features["chm_entropy"] = calculate_entropy(valid_chm)
    
    return features


def calculate_rumple_index(chm_arr: np.ndarray) -> float:
    """Calculate rumple index (surface roughness) from CHM.
    
    Rumple index = 3D surface area / 2D projected area
    """
    if chm_arr.size == 0:
        return np.nan
        
    # Calculate surface area using neighboring pixels
    dy, dx = np.gradient(chm_arr)
    
    # Surface area element
    surface_area = np.sqrt(1 + dx**2 + dy**2)
    
    # Handle NaN values
    valid_mask = ~np.isnan(surface_area)
    if not valid_mask.any():
        return 1.0
    
    # Rumple index
    rumple = float(np.nanmean(surface_area))
    
    return rumple


def calculate_entropy(arr: np.ndarray, bins: int = 50) -> float:
    """Calculate Shannon entropy of array values."""
    if len(arr) == 0:
        return np.nan
        
    # Create histogram
    counts, _ = np.histogram(arr, bins=bins)
    
    # Calculate probabilities
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Remove zeros
    
    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))
    
    return float(entropy)


def extract_spectral_features(
    hsi_arr: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    band_mapping: Optional[Dict[str, int]] = None
) -> Dict[str, float]:
    """Extract spectral indices and texture features from hyperspectral data.
    
    Args:
        hsi_arr: Hyperspectral array (bands, height, width)
        wavelengths: Array of wavelengths for each band (nm)
        band_mapping: Dictionary mapping band names to indices
        
    Returns:
        Dictionary of spectral features
    """
    features = {}
    
    # If band mapping not provided, try to infer from wavelengths
    if band_mapping is None and wavelengths is not None:
        band_mapping = find_spectral_bands(wavelengths)
    
    if band_mapping is None:
        logger.warning("No band mapping available for spectral indices")
        return features
    
    # Extract key bands
    try:
        red = hsi_arr[band_mapping.get('red', 50)]  # ~660nm
        nir = hsi_arr[band_mapping.get('nir', 90)]  # ~860nm
        swir1 = hsi_arr[band_mapping.get('swir1', 200)]  # ~1600nm
        swir2 = hsi_arr[band_mapping.get('swir2', 240)]  # ~2200nm
        blue = hsi_arr[band_mapping.get('blue', 20)]  # ~480nm
        green = hsi_arr[band_mapping.get('green', 35)]  # ~560nm
        
        # Vegetation indices
        features['ndvi_aop'] = calculate_index(nir, red, 'normalized_difference')
        features['evi_aop'] = calculate_evi(nir, red, blue)
        features['savi_aop'] = calculate_savi(nir, red)
        features['msavi_aop'] = calculate_msavi(nir, red)
        
        # Water indices
        features['ndwi_aop'] = calculate_index(green, nir, 'normalized_difference')
        features['mndwi_aop'] = calculate_index(green, swir1, 'normalized_difference')
        
        # Burn indices
        features['nbr_aop'] = calculate_index(nir, swir2, 'normalized_difference')
        features['nbr2_aop'] = calculate_index(swir1, swir2, 'normalized_difference')
        
        # Soil/mineral indices
        features['ndsi_aop'] = calculate_index(swir1, green, 'normalized_difference')
        
        # Texture features for key indices
        ndvi = features['ndvi_aop']
        if not np.isnan(ndvi).all():
            texture_feats = extract_texture_features(
                normalize_to_uint8(hsi_arr[band_mapping.get('nir', 90)])
            )
            features.update({f"nir_texture_{k}": v for k, v in texture_feats.items()})
        
    except Exception as e:
        logger.error(f"Error calculating spectral features: {e}")
    
    return features


def find_spectral_bands(wavelengths: np.ndarray) -> Dict[str, int]:
    """Find band indices for common spectral regions.
    
    Args:
        wavelengths: Array of wavelengths in nm
        
    Returns:
        Dictionary mapping band names to indices
    """
    band_centers = {
        'blue': 480,
        'green': 560,
        'red': 665,
        'red_edge': 705,
        'nir': 860,
        'swir1': 1610,
        'swir2': 2190
    }
    
    band_mapping = {}
    for name, target_wl in band_centers.items():
        # Find closest wavelength
        idx = np.argmin(np.abs(wavelengths - target_wl))
        if np.abs(wavelengths[idx] - target_wl) < 50:  # Within 50nm
            band_mapping[name] = int(idx)
    
    return band_mapping


def calculate_index(band1: np.ndarray, band2: np.ndarray, method: str = 'normalized_difference') -> float:
    """Calculate spectral index from two bands."""
    valid_mask = ~np.isnan(band1) & ~np.isnan(band2) & (band1 > 0) & (band2 > 0)
    
    if not valid_mask.any():
        return np.nan
    
    b1 = band1[valid_mask]
    b2 = band2[valid_mask]
    
    if method == 'normalized_difference':
        index = (b1 - b2) / (b1 + b2 + 1e-8)
    elif method == 'ratio':
        index = b1 / (b2 + 1e-8)
    else:
        raise ValueError(f"Unknown index method: {method}")
    
    return float(np.median(index))


def calculate_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> float:
    """Calculate Enhanced Vegetation Index."""
    valid_mask = ~np.isnan(nir) & ~np.isnan(red) & ~np.isnan(blue)
    
    if not valid_mask.any():
        return np.nan
    
    G = 2.5
    C1 = 6.0
    C2 = 7.5
    L = 1.0
    
    evi = G * (nir - red) / (nir + C1 * red - C2 * blue + L + 1e-8)
    
    return float(np.median(evi[valid_mask]))


def calculate_savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> float:
    """Calculate Soil Adjusted Vegetation Index."""
    valid_mask = ~np.isnan(nir) & ~np.isnan(red)
    
    if not valid_mask.any():
        return np.nan
    
    savi = (1 + L) * (nir - red) / (nir + red + L + 1e-8)
    
    return float(np.median(savi[valid_mask]))


def calculate_msavi(nir: np.ndarray, red: np.ndarray) -> float:
    """Calculate Modified SAVI."""
    valid_mask = ~np.isnan(nir) & ~np.isnan(red)
    
    if not valid_mask.any():
        return np.nan
    
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
    
    return float(np.median(msavi[valid_mask]))


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize array to uint8 range for texture analysis."""
    arr_clean = arr[~np.isnan(arr)]
    
    if len(arr_clean) == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    
    # Clip to percentiles to handle outliers
    vmin = np.percentile(arr_clean, 2)
    vmax = np.percentile(arr_clean, 98)
    
    # Normalize
    arr_norm = (arr - vmin) / (vmax - vmin + 1e-8)
    arr_norm = np.clip(arr_norm, 0, 1)
    
    return (arr_norm * 255).astype(np.uint8)


def extract_texture_features(
    arr: np.ndarray,
    distances: List[int] = [1, 3, 5],
    angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]
) -> Dict[str, float]:
    """Extract GLCM texture features."""
    features = {}
    
    # Convert to uint8 if needed
    if arr.dtype != np.uint8:
        arr = normalize_to_uint8(arr)
    
    # Calculate GLCM
    glcm = graycomatrix(
        arr, 
        distances=distances, 
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )
    
    # Extract properties
    props = ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity']
    
    for prop in props:
        vals = graycoprops(glcm, prop).flatten()
        features[f"{prop}_mean"] = float(np.mean(vals))
        features[f"{prop}_std"] = float(np.std(vals))
    
    return features


def process_aop_to_grid(
    aop_bundle: AOPBundle,
    grid_spec: str,
    output_dir: Path,
    aoi_bounds: Optional[Tuple[float, float, float, float]] = None
) -> gpd.GeoDataFrame:
    """Process AOP data to target grid and export features.
    
    Args:
        aop_bundle: Container with AOP data paths
        grid_spec: Target grid specification
        output_dir: Output directory
        aoi_bounds: Optional bounds to clip to (minx, miny, maxx, maxy)
        
    Returns:
        GeoDataFrame with extracted features
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine AOI
    if aoi_bounds:
        aoi = box(*aoi_bounds)
    else:
        # Use bounds from first available raster
        raster_path = aop_bundle.chm_path or aop_bundle.hsi_path
        if not raster_path:
            raise ValueError("No raster data available in AOPBundle")
        
        with rio.open(raster_path) as src:
            aoi = box(*src.bounds)
            aop_bundle.crs = src.crs
            aop_bundle.bounds = src.bounds
    
    # Get target grid parameters
    dst_crs, dst_transform, width, height, res = infer_dst_grid(grid_spec, aoi)
    
    features_list = []
    
    # Process CHM if available
    if aop_bundle.chm_path and Path(aop_bundle.chm_path).exists():
        logger.info(f"Processing CHM data from {aop_bundle.chm_path}")
        
        # Warp to grid
        chm_aligned_path = output_dir / f"chm_{grid_spec}.tif"
        warp_to_grid(
            aop_bundle.chm_path,
            str(chm_aligned_path),
            dst_crs, dst_transform,
            width, height,
            resampling="bilinear"  # Better for continuous elevation data
        )
        
        # Extract features per pixel
        with rio.open(chm_aligned_path) as src:
            chm_arr = src.read(1)
            
            # Process in tiles for memory efficiency
            tile_size = 256
            for i in range(0, height, tile_size):
                for j in range(0, width, tile_size):
                    # Extract tile
                    i_end = min(i + tile_size, height)
                    j_end = min(j + tile_size, width)
                    tile = chm_arr[i:i_end, j:j_end]
                    
                    if tile.size > 0 and not np.isnan(tile).all():
                        # Calculate features for tile
                        chm_feats = extract_chm_features(tile)
                        
                        # Get center coordinates
                        row_center = i + (i_end - i) // 2
                        col_center = j + (j_end - j) // 2
                        x, y = rio.transform.xy(dst_transform, row_center, col_center)
                        
                        # Create feature record
                        feature_rec = {
                            'site': aop_bundle.site,
                            'year': aop_bundle.year,
                            'x': x,
                            'y': y,
                            'geometry': Point(x, y),
                            **chm_feats
                        }
                        features_list.append(feature_rec)
    
    # Process hyperspectral if available
    if aop_bundle.hsi_path and Path(aop_bundle.hsi_path).exists():
        logger.info(f"Processing hyperspectral data from {aop_bundle.hsi_path}")
        
        # Warp to grid
        hsi_aligned_path = output_dir / f"hsi_{grid_spec}.tif"
        warp_to_grid(
            aop_bundle.hsi_path,
            str(hsi_aligned_path),
            dst_crs, dst_transform,
            width, height,
            resampling="average"
        )
        
        # Extract spectral features
        # TODO: Read wavelength information from NEON metadata
        with rio.open(hsi_aligned_path) as src:
            # For now, assume standard NEON hyperspectral band configuration
            # This would need to be updated based on actual NEON products
            logger.warning("Using placeholder band mapping - update with actual NEON wavelengths")
            
            # Process a subset of bands for indices
            # Would need actual wavelength mapping here
            pass
    
    # Convert to GeoDataFrame
    if features_list:
        gdf = gpd.GeoDataFrame(features_list, crs=dst_crs)
        
        # Save outputs
        gdf.to_parquet(output_dir / f"aop_features_{aop_bundle.site}_{aop_bundle.year}.parquet")
        
        # Also save as GeoTIFF rasters for each feature
        feature_cols = [col for col in gdf.columns if col not in ['site', 'year', 'x', 'y', 'geometry']]
        
        for feat_name in feature_cols[:5]:  # Save top 5 features as rasters
            if feat_name in gdf.columns:
                save_feature_as_raster(
                    gdf, feat_name,
                    output_dir / f"{feat_name}_{aop_bundle.site}_{aop_bundle.year}.tif",
                    dst_transform, width, height, dst_crs
                )
        
        logger.info(f"Saved {len(gdf)} feature records for {aop_bundle.site} {aop_bundle.year}")
        
        return gdf
    
    else:
        logger.warning("No features extracted")
        return gpd.GeoDataFrame()


def save_feature_as_raster(
    gdf: gpd.GeoDataFrame,
    feature_name: str,
    output_path: Path,
    transform: rio.Affine,
    width: int,
    height: int,
    crs: str
) -> None:
    """Save a single feature as a GeoTIFF raster."""
    # Create empty raster
    raster = np.full((height, width), np.nan, dtype=np.float32)
    
    # Fill with feature values
    for _, row in gdf.iterrows():
        # Get pixel coordinates
        col, row_idx = ~transform * (row.x, row.y)
        col, row_idx = int(col), int(row_idx)
        
        if 0 <= col < width and 0 <= row_idx < height:
            raster[row_idx, col] = row[feature_name]
    
    # Write raster
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',
        'nodata': np.nan
    }
    
    with rio.open(output_path, 'w', **profile) as dst:
        dst.write(raster, 1)


def main():
    """CLI interface for AOP feature extraction."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Extract features from NEON AOP data")
    parser.add_argument('--site', required=True, help='NEON site code')
    parser.add_argument('--year', type=int, required=True, help='Year of AOP data')
    parser.add_argument('--config', default='configs/aop_sites.yaml', help='Site configuration file')
    parser.add_argument('--grid', help='Grid specification (overrides config)')
    parser.add_argument('--output', default='data/processed/aop', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Find site config
    site_config = None
    for site in config['sites']:
        if site['code'] == args.site:
            site_config = site
            break
    
    if not site_config:
        raise ValueError(f"Site {args.site} not found in configuration")
    
    # TODO: Implement actual data loading based on configuration
    # For now, this is a placeholder
    logger.info(f"Processing {args.site} for year {args.year}")
    

if __name__ == "__main__":
    main()