# Private Data Management for NEON AOP Integration

## Overview

This document describes the separation between private raw NEON AOP data and public processed features. The strategy ensures that sensitive or large-scale raw data remains private while sharing valuable processed features and models publicly.

## Data Separation Strategy

### Private Repository (`wildfire-aop-private`)
Contains:
- Raw NEON AOP downloads (HDF5, full resolution GeoTIFFs)
- Bulk hyperspectral data cubes
- Full-resolution LiDAR point clouds
- Pre-processing scripts for raw data
- Data staging and organization tools
- API keys and authentication credentials

Location: `AOP_DATA_ROOT` environment variable (e.g., `/data/private/neon-aop/`)

### Public Repository (this repo)
Contains:
- Processing code and algorithms
- Configuration files (sites, products, parameters)
- Aggregated features at 10-30m resolution
- Trained crosswalk models (small JSON/PKL files)
- Sample tiles for demonstration (256x256 pixels)
- Documentation and notebooks

Location: `AOP_OUT_ROOT` environment variable (default: `./data/processed/aop/`)

## Data Flow

```
Private Repository                    Public Repository
─────────────────                    ─────────────────
                                    
Raw AOP Data (TB)                    Processing Code
     │                                    │
     ├─> Pre-processing ─────────────────┤
     │                                    │
     └─> Staging Area                     │
              │                           │
              └──> Feature Extraction <───┘
                          │
                          ├─> 10-30m Features (MB)
                          ├─> Crosswalk Models (KB)
                          └─> Sample Tiles (MB)
```

## Access Instructions

### For Collaborators with Private Access

1. Request access to the private repository
2. Clone both repositories:
   ```bash
   git clone <private-repo-url> ~/wildfire-aop-private
   git clone <public-repo-url> ~/wildfire-risk-prediction
   ```

3. Set environment variables:
   ```bash
   export AOP_DATA_ROOT=~/wildfire-aop-private/data
   export AOP_OUT_ROOT=~/wildfire-risk-prediction/data/processed/aop
   ```

4. Run the full pipeline:
   ```bash
   cd ~/wildfire-risk-prediction
   make aop_all
   ```

### For Public Users

1. Use pre-processed features available in this repository
2. Download sample data:
   ```bash
   make download-samples
   ```

3. Run crosswalk models on your own satellite data:
   ```python
   from src.features.aop_crosswalk import CrosswalkModel
   
   # Load pre-trained model
   model = CrosswalkModel.load('data/models/aop_crosswalk/grsm_chm_p50.json')
   
   # Apply to satellite data
   synthetic_chm = model.predict(satellite_features)
   ```

## Data Products

### Available in Public Repo

1. **Aggregated Features** (`data/processed/aop/`)
   - Site-year combinations as Parquet files
   - Key metrics: CHM statistics, spectral indices
   - Resolution: 10m (Sentinel-2) or 30m (Landsat)

2. **Crosswalk Models** (`data/models/aop_crosswalk/`)
   - Linear models as JSON (transparent, small)
   - Ensemble models as PKL (more complex)
   - Model performance metrics

3. **Sample Tiles** (`data/samples/aop/`)
   - Small subsets for testing (256x256 pixels)
   - Includes both raw and processed examples
   - One tile per site for demonstration

### Available Only in Private Repo

1. **Raw Hyperspectral Cubes**
   - Full 426-band reflectance data
   - ~10-50 GB per site-year
   - HDF5 format with metadata

2. **LiDAR Products**
   - Point clouds (LAS/LAZ format)
   - Full-resolution CHM/DTM/DSM
   - 1m resolution rasters

3. **Processing Logs**
   - Download histories
   - Processing parameters
   - Quality control reports

## Security Considerations

- Never commit raw AOP data to the public repository
- Use `.gitignore` to exclude `AOP_DATA_ROOT` paths
- Store API credentials only in private repository
- Review all commits for accidental data inclusion

## Commercial Use

The separation allows for:
- **Open Source**: Share algorithms and methods publicly
- **Commercial**: Monetize processed features and models
- **Research**: Provide sample data for reproducibility
- **Privacy**: Protect sensitive ecological locations

## Contact

For access to private data or commercial licensing:
- Email: [your-email]
- Subject: "NEON AOP Data Access Request"

Include:
1. Your affiliation
2. Intended use case
3. Required sites and years
4. Data sharing agreements you can sign