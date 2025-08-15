# Public NEON AOP Data Access Guide

This guide explains how to access and use the publicly available NEON AOP (Airborne Observation Platform) data that powers the crosswalk system.

## 🌐 NEON Data Portal

All NEON AOP data is publicly available through the **NEON Data Portal**:
- **URL**: https://data.neonscience.org/
- **Access**: No registration required for public data
- **Cost**: Completely free
- **License**: Open data with citation requirements

## 📊 Available Data Products

### 1. Canopy Height Model (CHM) - DP3.30024.001
- **Resolution**: 1 meter
- **Format**: GeoTIFF
- **Coverage**: Full site coverage
- **Use Case**: Vegetation structure analysis, biomass estimation

### 2. Hyperspectral Reflectance - DP3.30026.001
- **Resolution**: 1 meter
- **Format**: HDF5
- **Bands**: 426 spectral bands (380-2510 nm)
- **Use Case**: Detailed spectral analysis, vegetation health assessment

### 3. RGB Camera Imagery - DP3.30010.001
- **Resolution**: 10 centimeters
- **Format**: GeoTIFF
- **Bands**: 3 (Red, Green, Blue)
- **Use Case**: Visual interpretation, cloud detection, ground truth validation

### 4. Discrete Return LiDAR - DP1.30003.001
- **Resolution**: Point cloud
- **Format**: LAS
- **Density**: 8-15 points per square meter
- **Use Case**: 3D structure analysis, ground elevation, canopy modeling

## 🗺️ Supported Sites

### SRER - Santa Rita Experimental Range
- **Location**: Arizona (31.8214°N, 110.8661°W)
- **Ecosystem**: Desert grassland and shrubland
- **Available Years**: 2021, 2022
- **Research Focus**: Vegetation structure, biomass estimation, ecosystem monitoring

### JORN - Jornada Experimental Range
- **Location**: New Mexico (32.6147°N, 106.7400°W)
- **Ecosystem**: Chihuahuan Desert
- **Available Years**: 2021
- **Research Focus**: Desert ecology, vegetation change, climate adaptation

### ONAQ - Onaqui Airstrip
- **Location**: Utah (40.4567°N, 112.4567°W)
- **Ecosystem**: Sagebrush steppe
- **Available Years**: 2021
- **Research Focus**: Rangeland management, sagebrush ecology, wildlife habitat

### SJER - San Joaquin Experimental Range
- **Location**: California (37.1089°N, 119.7328°W)
- **Ecosystem**: Oak woodland
- **Available Years**: 2021
- **Research Focus**: Oak woodland ecology, Mediterranean ecosystems, biodiversity monitoring

## 📥 How to Download Data

### Method 1: NEON Data Portal (Web Interface)
1. Visit https://data.neonscience.org/
2. Navigate to "Data" → "Airborne Remote Sensing"
3. Select your site of interest
4. Choose the data product and year
5. Download files directly to your computer

### Method 2: NEON API (Programmatic)
```python
import requests
import json

# NEON API base URL
base_url = "https://data.neonscience.org/api/v0"

# Get available data for a site
site_code = "SRER"
year = "2021"
product_code = "DP3.30024.001"  # CHM

# Query available data
url = f"{base_url}/data/{site_code}/{year}/{product_code}"
response = requests.get(url)
data = response.json()

# Download files
for item in data['data']['files']:
    if item['name'].endswith('.tif'):  # GeoTIFF files
        download_url = item['url']
        filename = item['name']
        # Download logic here
```

### Method 3: Using the Makefile (Automated)
```bash
# Download data for a specific site
make download-SRER

# Download data for all sites
make download
```

## 🔧 Data Processing

### Local Processing
The crosswalk system processes data locally on your machine:
1. **Download**: Raw data from NEON Data Portal
2. **Process**: Extract features using the provided tools
3. **Train**: Crosswalk models using your processed data
4. **Validate**: Assess model performance and accuracy

### Processing Commands
```bash
# Process downloaded AOP data
make process

# Train crosswalk models
make calibrate

# Validate models
make validate

# Run complete pipeline
make all
```

## 📁 Data Organization

### Recommended Directory Structure
```
data/
├── raw/
│   └── aop/
│       ├── SRER/
│       │   ├── 2021/
│       │   │   ├── DP3.30024.001/  # CHM
│       │   │   ├── DP3.30026.001/  # Hyperspectral
│       │   │   ├── DP3.30010.001/  # RGB
│       │   │   └── DP1.30003.001/  # LiDAR
│       │   └── 2022/
│       └── JORN/
│           └── 2021/
├── processed/
│   └── aop/
│       ├── SRER/
│       │   ├── 2021/
│       │   └── 2022/
│       └── JORN/
│           └── 2021/
└── models/
    └── aop_crosswalk/
```

## 📊 Data Requirements

### Minimum Data for Crosswalk
- **CHM Data**: Required for canopy structure analysis
- **Hyperspectral Data**: Required for spectral calibration
- **RGB Data**: Recommended for quality assessment
- **LiDAR Data**: Optional but enhances accuracy

### Quality Thresholds
- **Cloud Cover**: Maximum 20%
- **Data Coverage**: Minimum 80% of study area
- **Temporal Gap**: Maximum 30 days between AOP and satellite data

## 🚀 Getting Started

### 1. Choose Your Site
- Start with **SRER** for comprehensive data availability
- Consider **JORN** for desert ecosystem studies
- Use **ONAQ** for rangeland applications
- Select **SJER** for Mediterranean climate research

### 2. Download Data
```bash
# Download data for your chosen site
make download-SRER  # Replace SRER with your site
```

### 3. Process and Analyze
```bash
# Process the downloaded data
make process

# Train crosswalk models
make calibrate
```

### 4. Integrate with Your Workflow
```python
from src.integration.aop_integration import AOPIntegrationManager

# Initialize with your configuration
manager = AOPIntegrationManager('configs/aop_sites.yaml')

# Generate enhanced features
enhanced_data = manager.get_enhanced_features(
    satellite_data, 'SRER', 2021
)
```

## 📚 Additional Resources

### NEON Documentation
- **Data Portal Guide**: https://data.neonscience.org/data-products/explore
- **API Documentation**: https://data.neonscience.org/api
- **Data Quality**: https://data.neonscience.org/data-products/DP3.30024.001

### Scientific Literature
- **NEON AOP Overview**: Kampe et al. (2010) - "NEON: the first continental-scale ecological observatory"
- **AOP Applications**: Schimel et al. (2019) - "NEON science: enabling understanding of continental-scale ecology"

### Community Support
- **NEON Forums**: https://community.neonscience.org/
- **GitHub Issues**: Report bugs or request features
- **Documentation**: Comprehensive guides in this repository

## ⚠️ Important Notes

### Data Usage
- **Citation Required**: Always cite NEON when using their data
- **License**: Data is open but subject to NEON's terms of use
- **Attribution**: Include NEON acknowledgment in publications

### Technical Requirements
- **Storage**: AOP data can be large (several GB per site)
- **Processing**: Feature extraction requires significant computational resources
- **Memory**: Hyperspectral data processing needs adequate RAM

### Best Practices
- **Start Small**: Begin with a single site and year
- **Validate Data**: Check data quality before processing
- **Backup**: Keep raw data backups
- **Document**: Record your processing steps and parameters

---

**Need Help?** Check the main README, open a GitHub issue, or consult the NEON documentation for additional support.
