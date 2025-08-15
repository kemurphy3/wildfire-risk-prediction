"""
Geospatial alignment utilities for co-registering rasters to common grids.

This module provides utilities for aligning NEON AOP data with satellite grids
for crosswalk analysis and validation.
"""

import rasterio as rio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import Polygon
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union, List
import logging

logger = logging.getLogger(__name__)


def infer_dst_grid(grid_spec: Union[str, List[float]], aoi: Union[str, Polygon]) -> Dict[str, Any]:
    """
    Infer destination grid parameters from specification.
    
    Args:
        grid_spec: Grid specification ('sentinel2_10m' or 'landsat_30m') or bounds list
        aoi: Area of interest polygon or grid specification string
        
    Returns:
        Tuple of (crs, transform, width, height, resolution)
    """
    # Handle different input types
    if isinstance(grid_spec, str):
        # grid_spec is the grid type, aoi should be bounds
        grid_type = grid_spec
        bounds = aoi
    else:
        # grid_spec is bounds, aoi is grid type
        grid_type = aoi
        bounds = grid_spec
    
    # Get bounds
    if isinstance(bounds, str):
        # Parse bounds string
        bounds = [float(x) for x in bounds.strip('[]').split(',')]
    
    minx, miny, maxx, maxy = bounds
    
    # Set grid parameters based on type
    if grid_type == 'sentinel2_10m':
        resolution = 10.0
        crs = 'EPSG:32632'  # UTM Zone 32N for Sentinel-2
    elif grid_type == 'landsat_30m':
        resolution = 30.0
        crs = 'EPSG:32632'  # UTM Zone 32N for Landsat
    else:
        resolution = 10.0  # Default
        crs = 'EPSG:32632'
    
    # Calculate grid dimensions
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    
    # Create transform
    transform = rio.Affine.translation(minx, miny) * rio.Affine.scale(resolution, resolution)
    
    logger.info(f"Inferred grid: {grid_type}, {width}x{height}, {resolution}m resolution")
    
    return {
        'crs': crs,
        'transform': transform,
        'width': width,
        'height': height,
        'resolution': resolution
    }


def warp_to_grid(
    src_path: str,
    dst_path: str,
    dst_crs: str,
    dst_transform: rio.Affine,
    width: int,
    height: int,
    resampling: str = "average"
) -> None:
    """
    Warp source raster to destination grid.
    
    Args:
        src_path: Path to source raster
        dst_path: Path to destination raster
        dst_crs: Destination CRS
        dst_transform: Destination transform
        width: Destination width in pixels
        height: Destination height in pixels
        resampling: Resampling method
    """
    logger.info(f"Warping {src_path} to {dst_path}")
    
    with rio.open(src_path) as src:
        # Determine resampling method
        if resampling == "average":
            resampling_method = Resampling.average
        elif resampling == "bilinear":
            resampling_method = Resampling.bilinear
        elif resampling == "cubic":
            resampling_method = Resampling.cubic
        elif resampling == "nearest":
            resampling_method = Resampling.nearest
        else:
            resampling_method = Resampling.average
        
        # Prepare destination metadata
        dst_meta = src.meta.copy()
        dst_meta.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': width,
            'height': height,
            'driver': 'GTiff'
        })
        
        # Perform warping
        with rio.open(dst_path, 'w', **dst_meta) as dst:
            reproject(
                source=rio.band(src, 1),
                destination=rio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling_method
            )
    
    logger.info(f"Successfully warped to {dst_path}")


def rasterize_mask(shapes, out_meta_like) -> np.ndarray:
    """
    Rasterize vector shapes to match a raster template.
    
    Args:
        shapes: Vector shapes to rasterize
        out_meta_like: Raster metadata to match
        
    Returns:
        Rasterized mask as numpy array
    """
    logger.info("Rasterizing shapes to mask")
    
    # Create output array
    out_shape = (out_meta_like['height'], out_meta_like['width'])
    mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=out_meta_like['transform'],
        fill=0,
        default_value=1,
        dtype=np.uint8
    )
    
    return mask


def calculate_overlap_stats(src_path: str, dst_path: str) -> Dict[str, float]:
    """
    Calculate overlap statistics between source and destination rasters.
    
    Args:
        src_path: Path to source raster
        dst_path: Path to destination raster
        
    Returns:
        Dictionary of overlap statistics
    """
    logger.info(f"Calculating overlap between {src_path} and {dst_path}")
    
    with rio.open(src_path) as src, rio.open(dst_path) as dst:
        # Get bounds
        src_bounds = src.bounds
        dst_bounds = dst.bounds
        
        # Calculate intersection
        intersection = (
            max(src_bounds.left, dst_bounds.left),
            max(src_bounds.bottom, dst_bounds.bottom),
            min(src_bounds.right, dst_bounds.right),
            min(src_bounds.top, dst_bounds.top)
        )
        
        # Calculate areas
        src_area = (src_bounds.right - src_bounds.left) * (src_bounds.top - src_bounds.bottom)
        dst_area = (dst_bounds.right - dst_bounds.left) * (dst_bounds.top - dst_bounds.bottom)
        
        if intersection[2] > intersection[0] and intersection[3] > intersection[1]:
            intersection_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
            overlap_ratio = intersection_area / min(src_area, dst_area)
        else:
            intersection_area = 0.0
            overlap_ratio = 0.0
        
        stats = {
            'src_area': src_area,
            'dst_area': dst_area,
            'intersection_area': intersection_area,
            'overlap_ratio': overlap_ratio,
            'src_resolution': src.res[0],
            'dst_resolution': dst.res[0]
        }
        
        logger.info(f"Overlap ratio: {overlap_ratio:.3f}")
        return stats


def extract_window_from_bounds(
    raster_path: str,
    bounds: Tuple[float, float, float, float],
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Extract a window from a raster based on geographic bounds.
    
    Args:
        raster_path: Path to input raster
        bounds: Geographic bounds (minx, miny, maxx, maxy)
        output_path: Optional path to save extracted window
        
    Returns:
        Extracted window as numpy array
    """
    logger.info(f"Extracting window from {raster_path}")
    
    with rio.open(raster_path) as src:
        # Convert bounds to pixel coordinates
        window = rio.windows.from_bounds(bounds, src.transform)
        
        # Read the window
        data = src.read(window=window)
        
        if output_path:
            # Save extracted window
            meta = src.meta.copy()
            meta.update({
                'height': window.height,
                'width': window.width,
                'transform': rio.windows.transform(window, src.transform)
            })
            
            with rio.open(output_path, 'w', **meta) as dst:
                dst.write(data)
            
            logger.info(f"Saved extracted window to {output_path}")
        
        return data


def validate_alignment(
    src_path: str,
    dst_path: str,
    tolerance_pixels: float = 0.5
) -> Dict[str, Any]:
    """
    Validate alignment between source and destination rasters.
    
    Args:
        src_path: Path to source raster
        dst_path: Path to destination raster
        tolerance_pixels: Tolerance for alignment in pixels
        
    Returns:
        Validation results
    """
    logger.info(f"Validating alignment between {src_path} and {dst_path}")
    
    with rio.open(src_path) as src, rio.open(dst_path) as dst:
        # Check CRS
        crs_match = src.crs == dst.crs
        
        # Check resolution
        src_res = src.res[0]
        dst_res = dst.res[0]
        resolution_match = abs(src_res - dst_res) < tolerance_pixels * min(src_res, dst_res)
        
        # Check bounds alignment
        src_bounds = src.bounds
        dst_bounds = dst.bounds
        
        # Calculate alignment error
        alignment_error = max(
            abs(src_bounds.left - dst_bounds.left),
            abs(src_bounds.bottom - dst_bounds.bottom),
            abs(src_bounds.right - dst_bounds.right),
            abs(src_bounds.top - dst_bounds.top)
        )
        
        alignment_ok = alignment_error < tolerance_pixels * min(src_res, dst_res)
        
        validation_results = {
            'crs_match': crs_match,
            'resolution_match': resolution_match,
            'alignment_error': alignment_error,
            'alignment_ok': alignment_ok,
            'src_resolution': src_res,
            'dst_resolution': dst_res,
            'tolerance_pixels': tolerance_pixels
        }
        
        logger.info(f"Alignment validation: {'PASS' if alignment_ok else 'FAIL'}")
        logger.info(f"Alignment error: {alignment_error:.3f} meters")
        
        return validation_results


if __name__ == "__main__":
    # Example usage
    print("Geospatial Alignment Utilities")
    print("=" * 40)
    print("This module provides utilities for aligning NEON AOP data with satellite grids.")
    print("Use the functions to:")
    print("- Infer destination grid parameters")
    print("- Warp rasters to common grids")
    print("- Validate alignment accuracy")
    print("- Extract windows from rasters")