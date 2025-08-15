# 📋 Cursor Prompt — NEON AOP Crosswalk Implementation

## Context
The wildfire-risk-prediction repository models wildfire spread/risk using satellite data (Sentinel-2/Landsat/VIIRS/MODIS), weather, and topography. We need to enhance it by integrating fine-scale structure/condition signals from NEON Airborne Observation Platform (AOP) data (hyperspectral + LiDAR-derived rasters) and crosswalk them to 10–30m grids to:
a) Calibrate satellite-derived vegetation indices using ground-truth AOP data
b) Validate satellite products at NEON sites with known fire history
c) Improve fire risk predictions by incorporating fine-scale canopy structure

### Existing Infrastructure to Leverage:
- `src/data_collection/neon_client.py` has basic AOP functionality and placeholder crosswalk methods
- `src/data_collection/satellite_client.py` has Google Earth Engine integration for Sentinel-2/Landsat/MODIS
- `src/features/fire_features.py` has comprehensive feature engineering pipeline
- Strong configuration system in `src/config.py`
- Existing notebook `notebooks/02_aop_satellite_crosswalk.ipynb` (needs implementation)

### Target NEON Sites with Fire History:
1. **GRSM** (Great Smoky Mountains) - 2016 Chimney Tops 2 Fire
   - Bounds: [-83.50, 35.55, -83.45, 35.60]  # TODO: Verify exact AOI
2. **SOAP** (Soaproot Saddle) - Creek Fire 2020, Blue Fire 2021
   - Bounds: [-119.26, 37.03, -119.21, 37.08]  # TODO: Verify exact AOI
3. **SJER** (San Joaquin Experimental Range) - Fire-prone, multiple small fires
   - Bounds: [-119.74, 37.10, -119.69, 37.15]  # TODO: Verify exact AOI
4. **SYCA** (Sycamore Creek) - Sand Stone Fire 2024
   - Bounds: [-111.51, 33.74, -111.46, 33.79]  # TODO: Verify exact AOI

## 🎯 Implementation Goals

1. **Enhance existing NEON client** to properly download and stage AOP rasters
2. **Create spatial alignment utilities** for co-registering AOP with satellite grids
3. **Extract AOP-derived features** at 10-30m resolution (canopy metrics, spectral indices)
4. **Build crosswalk models** that learn mappings between satellite and AOP-derived indices
5. **Integrate with existing pipeline** to improve fire risk predictions
6. **Maintain privacy separation** between raw AOP data (private) and processed features (public)

## 📁 Files to Create/Modify

### 1. Create `src/utils/geoalign.py`
```python
"""Geospatial alignment utilities for co-registering rasters to common grids."""

import rasterio as rio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from shapely.geometry import Polygon
import numpy as np
from typing import Tuple, Optional

def infer_dst_grid(grid_spec: str, aoi: Polygon) -> Tuple[str, rio.Affine, int, int, float]:
    """Infer destination grid parameters from specification.
    
    Args:
        grid_spec: Grid specification ('sentinel2_10m' or 'landsat_30m')
        aoi: Area of interest polygon
        
    Returns:
        Tuple of (crs, transform, width, height, resolution)
    """
    # TODO: Implement based on grid_spec
    
def warp_to_grid(
    src_path: str,
    dst_path: str,
    dst_crs: str,
    dst_transform: rio.Affine,
    width: int,
    height: int,
    resampling: str = "average"
) -> None:
    """Warp source raster to destination grid."""
    # Implementation provided in original prompt
    
def rasterize_mask(shapes, out_meta_like) -> np.ndarray:
    """Rasterize vector shapes to match a raster template."""
    # TODO: Implement using rasterio.features.rasterize
```

### 2. Create `src/features/aop_features.py`
```python
"""Extract features from NEON AOP data for fire risk modeling."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import rasterio as rio
from pathlib import Path
from typing import Dict, List, Optional
import geopandas as gpd
from skimage.feature import graycomatrix, graycoprops

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
    """Extract canopy height model statistics."""
    # Implementation provided in original prompt
    
def extract_spectral_features(hsi_arr: np.ndarray, wavelengths: np.ndarray) -> Dict[str, float]:
    """Extract spectral indices and texture features from hyperspectral data."""
    # TODO: Implement NDVI, NBR, NDWI calculation using NEON wavelengths
    # TODO: Add GLCM texture features
    
def process_aop_to_grid(
    aop_bundle: AOPBundle,
    grid_spec: str,
    output_dir: Path
) -> gpd.GeoDataFrame:
    """Process AOP data to target grid and export features."""
    # TODO: Main processing pipeline
    
if __name__ == "__main__":
    # CLI interface
    pass
```

### 3. Create `src/features/aop_crosswalk.py`
```python
"""Crosswalk models for calibrating satellite indices with AOP ground truth."""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
from pathlib import Path
from typing import Dict, Union, Literal

def fit_linear_crosswalk(X_sat: np.ndarray, y_aop: np.ndarray) -> Dict:
    """Fit regularized linear model for satellite-AOP mapping."""
    # Implementation provided in original prompt
    
def fit_ensemble_crosswalk(X_sat: np.ndarray, y_aop: np.ndarray) -> GradientBoostingRegressor:
    """Fit gradient boosting model for non-linear satellite-AOP mapping."""
    # TODO: Implement with cross-validation
    
def calibrate_satellite_indices(
    satellite_df: pd.DataFrame,
    aop_df: pd.DataFrame,
    target_vars: List[str],
    model_type: Literal["linear", "ensemble"] = "linear"
) -> Dict[str, Union[Dict, GradientBoostingRegressor]]:
    """Calibrate satellite indices using AOP ground truth."""
    # TODO: Main calibration pipeline
    
def validate_crosswalk(
    satellite_df: pd.DataFrame,
    aop_df: pd.DataFrame,
    models: Dict,
    output_dir: Path
) -> pd.DataFrame:
    """Validate crosswalk models and generate metrics."""
    # TODO: Compute R², MAE, generate plots
    
if __name__ == "__main__":
    # CLI interface
    pass
```

### 4. Update `src/data_collection/neon_client.py`
Enhance the existing placeholder `create_satellite_crosswalk` method:
- Integrate with new aop_features.py module
- Use proper NEON AOP product IDs
- Handle temporal matching with satellite data
- Add methods for downloading specific AOP tiles

### 5. Create `configs/aop_sites.yaml`
```yaml
# NEON AOP site configuration for wildfire crosswalk analysis
sites:
  GRSM:
    name: "Great Smoky Mountains"
    fire_events:
      - name: "Chimney Tops 2"
        year: 2016
    aoi: 
      minx: -83.50
      miny: 35.55
      maxx: -83.45
      maxy: 35.60
    years: [2015, 2017]  # Pre/post fire
    grid: "sentinel2_10m"
    
  SOAP:
    name: "Soaproot Saddle"
    fire_events:
      - name: "Creek Fire"
        year: 2020
      - name: "Blue Fire"
        year: 2021
    aoi:
      minx: -119.26
      miny: 37.03
      maxx: -119.21
      maxy: 37.08
    years: [2019, 2021]
    grid: "landsat_30m"
    
  SJER:
    name: "San Joaquin Experimental Range"
    fire_events: "Multiple small fires"
    aoi:
      minx: -119.74
      miny: 37.10
      maxx: -119.69
      maxy: 37.15
    years: [2018, 2020, 2022]
    grid: "sentinel2_10m"
    
  SYCA:
    name: "Sycamore Creek"
    fire_events:
      - name: "Sand Stone Fire"
        year: 2024
    aoi:
      minx: -111.51
      miny: 33.74
      maxx: -111.46
      maxy: 33.79
    years: [2023, 2024]
    grid: "sentinel2_10m"
```

### 6. Create `configs/aop_products.yaml`
```yaml
# NEON AOP product configuration
# TODO: Verify exact product IDs with NEON catalog
aop_products:
  canopy_height_model:
    id: "DP3.30015.001"
    name: "Ecosystem structure"
    type: "raster"
    
  hyperspectral_reflectance:
    id: "DP3.30006.001"  
    name: "Spectrometer orthorectified surface directional reflectance"
    type: "raster"
    bands: 426  # Full hyperspectral cube
    
  canopy_cover:
    id: "DP3.30014.001"  # TODO: Verify if this exists as separate product
    name: "Canopy cover and LAI"
    type: "raster"
    
  vegetation_indices:
    id: "DP3.30026.001"  # TODO: Verify
    name: "Vegetation indices"
    type: "raster"
    
# Derived products we'll compute
derived_features:
  - chm_p10
  - chm_p50
  - chm_p90
  - chm_std
  - canopy_cover_gt2m
  - canopy_cover_gt5m
  - ndvi_aop
  - nbr_aop
  - ndwi_aop
  - texture_contrast
  - texture_homogeneity
```

### 7. Create `Makefile`
```makefile
# Makefile for NEON AOP wildfire crosswalk pipeline

.PHONY: help aop_fetch aop_features aop_crosswalk aop_eval clean

help:
	@echo "NEON AOP Wildfire Crosswalk Pipeline"
	@echo "====================================="
	@echo "aop_fetch     - Download NEON AOP data for target sites"
	@echo "aop_features  - Extract features from AOP data"
	@echo "aop_crosswalk - Train/validate satellite-AOP crosswalk models"
	@echo "aop_eval      - Run evaluation notebook and generate report"
	@echo "clean         - Remove intermediate files"

aop_fetch:
	python -m src.data_collection.neon_client download-aop \
		--sites GRSM SOAP SJER SYCA \
		--years 2015,2017,2019,2021,2023,2024 \
		--products chm,hyperspectral

aop_features: aop_fetch
	@for site in GRSM SOAP SJER SYCA; do \
		echo "Processing $$site..."; \
		python -m src.features.aop_features \
			--site $$site \
			--config configs/aop_sites.yaml; \
	done

aop_crosswalk: aop_features
	python -m src.features.aop_crosswalk \
		--sites GRSM,SOAP,SJER,SYCA \
		--mode calibrate \
		--output data/models/aop_crosswalk/
	python -m src.features.aop_crosswalk \
		--sites GRSM,SOAP,SJER,SYCA \
		--mode validate \
		--output reports/aop/

aop_eval: aop_crosswalk
	jupyter nbconvert --execute notebooks/04_aop_crosswalk_demo.ipynb \
		--to html \
		--output ../reports/aop/crosswalk_demo.html

clean:
	rm -rf data/intermediate/aop/
	rm -rf reports/aop/*.png
```

### 8. Update `.env.example`
Add these lines:
```bash
# NEON AOP Configuration
AOP_DATA_ROOT=/path/to/private/aop/data  # Raw AOP data (not tracked)
AOP_OUT_ROOT=./data/processed/aop        # Processed features (can be tracked)
SAT_GRID=sentinel2_10m                   # Default grid specification

# NEON API (already exists, but verify)
NEON_API_TOKEN=your_token_here
```

### 9. Create `notebooks/04_aop_crosswalk_demo.ipynb`
Structure:
1. Load processed AOP features and satellite data for all 4 sites
2. Visualize pre/post fire changes where applicable
3. Show scatter plots of satellite vs AOP-derived indices
4. Load and apply crosswalk models
5. Compare model performance (satellite-only vs satellite+crosswalk)
6. Generate summary table of improvements

### 10. Update `README.md`
Add section:
```markdown
## NEON AOP Integration

This project integrates high-resolution airborne data from NEON's Airborne Observation Platform (AOP) to calibrate and validate satellite-derived vegetation indices. The crosswalk models improve fire risk predictions by incorporating fine-scale canopy structure and hyperspectral signatures.

### Target Sites
- GRSM: Great Smoky Mountains (2016 Chimney Tops 2 Fire)
- SOAP: Soaproot Saddle (2020 Creek Fire, 2021 Blue Fire)  
- SJER: San Joaquin Experimental Range (fire-prone ecosystem)
- SYCA: Sycamore Creek (2024 Sand Stone Fire)

### Privacy Note
Raw NEON AOP data is stored in a private repository. This public repository contains only:
- Configuration files and processing code
- Aggregated 10-30m features
- Trained crosswalk models
- Sample tiles for demonstration

See PRIVATE_DATA.md for information about accessing raw AOP data.
```

## 🔧 Implementation Priority

1. **Phase 1**: Core infrastructure
   - geoalign.py utilities
   - Update neon_client.py to properly download AOP data
   - Basic feature extraction from CHM

2. **Phase 2**: Feature engineering
   - Complete aop_features.py with spectral indices
   - Add texture features
   - Export to standardized format

3. **Phase 3**: Crosswalk modeling
   - Implement calibration models
   - Add validation metrics
   - Create visualization functions

4. **Phase 4**: Integration
   - Connect to existing fire_features.py pipeline
   - Update model training to use crosswalk features
   - Create comprehensive demo notebook

## 🧪 Testing Strategy

1. Unit tests for:
   - Geospatial alignment functions
   - Feature extraction algorithms
   - Model serialization/deserialization

2. Integration tests for:
   - End-to-end processing pipeline
   - Temporal matching with satellite data
   - Model prediction consistency

3. Validation checks:
   - Ensure co-registration accuracy < 0.5 pixels
   - Verify feature values are within expected ranges
   - Check model performance metrics improve with crosswalk

## 🔐 Data Management

- Keep raw AOP files in `AOP_DATA_ROOT` (private, not tracked)
- Export processed features to `AOP_OUT_ROOT/site/year/`
- Use Git LFS for sample tiles in `data/samples/aop/`
- Store models in `data/models/aop_crosswalk/`

## 📊 Success Metrics

- [ ] All 4 sites processed with pre/post fire data where available
- [ ] Crosswalk models show R² > 0.7 for key indices
- [ ] Fire risk model AUROC improves by >5% with AOP features
- [ ] Processing pipeline runs end-to-end via Makefile
- [ ] Demo notebook renders without errors using only public data