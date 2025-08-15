"""Geospatial alignment utilities for co-registering rasters to common grids."""

import rasterio as rio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import Polygon, box
import numpy as np
from typing import Tuple, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


def infer_dst_grid(grid_spec: str, aoi: Polygon) -> Tuple[str, rio.Affine, int, int, float]:
    """Infer destination grid parameters from specification.
    
    Args:
        grid_spec: Grid specification ('sentinel2_10m' or 'landsat_30m')
        aoi: Area of interest polygon
        
    Returns:
        Tuple of (crs, transform, width, height, resolution)
    """
    # Standard grid specifications
    grid_configs = {
        'sentinel2_10m': {
            'crs': 'EPSG:32610',  # UTM Zone 10N (works for CA sites)
            'resolution': 10.0
        },
        'landsat_30m': {
            'crs': 'EPSG:32610',
            'resolution': 30.0
        }
    }
    
    if grid_spec not in grid_configs:
        raise ValueError(f"Unknown grid specification: {grid_spec}")
    
    config = grid_configs[grid_spec]
    bounds = aoi.bounds  # (minx, miny, maxx, maxy)
    res = config['resolution']
    
    # Snap bounds to grid
    minx = np.floor(bounds[0] / res) * res
    miny = np.floor(bounds[1] / res) * res
    maxx = np.ceil(bounds[2] / res) * res
    maxy = np.ceil(bounds[3] / res) * res
    
    # Calculate dimensions
    width = int((maxx - minx) / res)
    height = int((maxy - miny) / res)
    
    # Create transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    logger.info(f"Grid {grid_spec}: {width}x{height} pixels at {res}m resolution")
    
    return config['crs'], transform, width, height, res


def warp_to_grid(
    src_path: str,
    dst_path: str,
    dst_crs: str,
    dst_transform: rio.Affine,
    width: int,
    height: int,
    resampling: str = "average"
) -> None:
    """Warp source raster to destination grid.
    
    Args:
        src_path: Path to source raster
        dst_path: Path to output raster
        dst_crs: Destination CRS
        dst_transform: Destination transform
        width: Output width in pixels
        height: Output height in pixels
        resampling: Resampling method name
    """
    resamp = getattr(Resampling, resampling)
    
    with rio.open(src_path) as src:
        # Update profile for output
        profile = src.profile.copy()
        profile.update({
            "crs": dst_crs,
            "transform": dst_transform,
            "width": width,
            "height": height,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256
        })
        
        with rio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                logger.debug(f"Reprojecting band {i}/{src.count}")
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resamp
                )
    
    logger.info(f"Warped {src_path} -> {dst_path}")


def rasterize_mask(
    shapes: List[Union[Polygon, dict]],
    out_shape: Tuple[int, int],
    transform: rio.Affine,
    all_touched: bool = True
) -> np.ndarray:
    """Rasterize vector shapes to match a raster template.
    
    Args:
        shapes: List of shapes to rasterize
        out_shape: Output shape (height, width)
        transform: Raster transform
        all_touched: Include all pixels touched by geometries
        
    Returns:
        Binary mask array
    """
    # Convert shapes to (geometry, value) pairs if needed
    if shapes and isinstance(shapes[0], Polygon):
        shapes = [(geom, 1) for geom in shapes]
    
    mask = rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        all_touched=all_touched,
        dtype=np.uint8
    )
    
    return mask


def align_rasters(
    src_paths: List[str],
    dst_dir: str,
    grid_spec: str,
    aoi: Optional[Polygon] = None,
    resampling: str = "average"
) -> List[str]:
    """Align multiple rasters to a common grid.
    
    Args:
        src_paths: List of source raster paths
        dst_dir: Output directory
        grid_spec: Target grid specification
        aoi: Optional area of interest to clip to
        resampling: Resampling method
        
    Returns:
        List of output paths
    """
    import os
    from pathlib import Path
    
    # Create output directory
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    
    # If no AOI provided, compute from first raster
    if aoi is None:
        with rio.open(src_paths[0]) as src:
            aoi = box(*src.bounds)
    
    # Get target grid parameters
    dst_crs, dst_transform, width, height, res = infer_dst_grid(grid_spec, aoi)
    
    output_paths = []
    for src_path in src_paths:
        # Generate output filename
        basename = os.path.basename(src_path)
        name, ext = os.path.splitext(basename)
        dst_path = os.path.join(dst_dir, f"{name}_{grid_spec}{ext}")
        
        # Warp to grid
        warp_to_grid(
            src_path, dst_path,
            dst_crs, dst_transform,
            width, height,
            resampling
        )
        
        output_paths.append(dst_path)
    
    return output_paths


def extract_window_from_bounds(
    raster_path: str,
    bounds: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, dict]:
    """Extract a window from a raster based on geographic bounds.
    
    Args:
        raster_path: Path to raster file
        bounds: Geographic bounds (minx, miny, maxx, maxy)
        
    Returns:
        Tuple of (data array, metadata dict)
    """
    with rio.open(raster_path) as src:
        # Convert bounds to pixel coordinates
        window = rio.windows.from_bounds(*bounds, src.transform)
        
        # Read the window
        data = src.read(window=window)
        
        # Get transform for the window
        win_transform = src.window_transform(window)
        
        # Create metadata
        meta = src.meta.copy()
        meta.update({
            'transform': win_transform,
            'width': window.width,
            'height': window.height
        })
        
    return data, meta


def compute_overlap_stats(
    raster1_path: str,
    raster2_path: str,
    band1: int = 1,
    band2: int = 1
) -> dict:
    """Compute statistics for overlapping regions of two rasters.
    
    Args:
        raster1_path: Path to first raster
        raster2_path: Path to second raster
        band1: Band index for first raster
        band2: Band index for second raster
        
    Returns:
        Dictionary of statistics
    """
    with rio.open(raster1_path) as src1, rio.open(raster2_path) as src2:
        # Find overlap bounds
        bounds1 = box(*src1.bounds)
        bounds2 = box(*src2.bounds)
        overlap = bounds1.intersection(bounds2)
        
        if overlap.is_empty:
            return {'overlap': False}
        
        # Extract overlapping windows
        data1, _ = extract_window_from_bounds(raster1_path, overlap.bounds)
        data2, _ = extract_window_from_bounds(raster2_path, overlap.bounds)
        
        # Ensure same shape (might differ by 1 pixel due to rounding)
        min_height = min(data1.shape[1], data2.shape[1])
        min_width = min(data1.shape[2], data2.shape[2])
        data1 = data1[:, :min_height, :min_width]
        data2 = data2[:, :min_height, :min_width]
        
        # Get specific bands
        arr1 = data1[band1 - 1]
        arr2 = data2[band2 - 1]
        
        # Compute statistics on valid pixels
        valid_mask = (~np.isnan(arr1)) & (~np.isnan(arr2))
        
        if not valid_mask.any():
            return {'overlap': True, 'valid_pixels': 0}
        
        valid1 = arr1[valid_mask]
        valid2 = arr2[valid_mask]
        
        stats = {
            'overlap': True,
            'valid_pixels': int(valid_mask.sum()),
            'correlation': float(np.corrcoef(valid1, valid2)[0, 1]),
            'rmse': float(np.sqrt(np.mean((valid1 - valid2) ** 2))),
            'mae': float(np.mean(np.abs(valid1 - valid2))),
            'r2': float(np.corrcoef(valid1, valid2)[0, 1] ** 2)
        }
        
    return stats