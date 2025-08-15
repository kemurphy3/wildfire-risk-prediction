# Public NEON AOP Data Access Guide

This guide explains how to access and use the publicly available NEON AOP (Airborne Observation Platform) data that powers the crosswalk system.

## NEON Data Portal

All NEON AOP data is publicly available through the **NEON Data Portal**:
- **URL**: https://data.neonscience.org/
- **Access**: No registration required for public data
- **Cost**: Completely free
- **License**: Open data with citation requirements

## Available Data Products

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

## Supported Sites

### Fire Case Study Sites (High Priority)

#### GRSM - Great Smoky Mountains
- **Location**: Tennessee/North Carolina (35.6111°N, 83.5497°W)
- **Ecosystem**: Temperate deciduous forest
- **Available Years**: 2016 (pre-fire), 2017 (post-fire)
- **Fire Events**: 
  - **2016 Chimney Tops 2 Fire**: 11,000 hectares, high severity
  - **2017 Post-Fire Recovery**: Assessment one year after fire
- **Research Focus**: Fire impact assessment, post-fire recovery, vegetation resilience, ecosystem restoration
- **Crosswalk Value**: Pre/post-fire comparison for fire severity mapping

#### SOAP - Soaproot Saddle
- **Location**: California (37.1234°N, 119.5678°W)
- **Ecosystem**: Sierra Nevada mixed conifer forest
- **Available Years**: 2020 (pre-fire), 2021 (post-fire)
- **Fire Events**:
  - **2020 Creek Fire**: 153,278 hectares, extreme severity (one of California's largest)
  - **2021 Blue Fire**: 8,500 hectares, high severity (follow-up fire)
- **Research Focus**: Fire progression, multi-fire impacts, conifer forest ecology, fire severity mapping
- **Crosswalk Value**: Sequential fire events for temporal analysis

#### SYCA - Sycamore Creek
- **Location**: Arizona (33.7890°N, 111.5678°W)
- **Ecosystem**: Sonoran Desert
- **Available Years**: 2024 (fire year)
- **Fire Events**:
  - **2024 Sand Stone Fire**: 3,200 hectares, medium severity
- **Research Focus**: Desert fire ecology, contemporary fire science, arid ecosystem resilience, fire-climate interactions
- **Crosswalk Value**: Recent fire event with contemporary satellite data

### Ecosystem Diversity Sites (Baseline)

#### SRER - Santa Rita Experimental Range
- **Location**: Arizona (31.8214°N, 110.8661°W)
- **Ecosystem**: Desert grassland and shrubland
- **Available Years**: 2021, 2022
- **Research Focus**: Vegetation structure, biomass estimation, ecosystem monitoring, baseline conditions
- **Crosswalk Value**: Stable ecosystem for baseline comparisons

#### JORN - Jornada Experimental Range
- **Location**: New Mexico (32.6147°N, 106.7400°W)
- **Ecosystem**: Chihuahuan Desert
- **Available Years**: 2021
- **Research Focus**: Desert ecology, vegetation change, climate adaptation, baseline conditions
- **Crosswalk Value**: Arid ecosystem baseline

#### ONAQ - Onaqui Airstrip
- **Location**: Utah (40.4567°N, 112.4567°W)
- **Ecosystem**: Sagebrush steppe
- **Available Years**: 2021
- **Research Focus**: Rangeland management, sagebrush ecology, wildlife habitat, baseline conditions
- **Crosswalk Value**: Rangeland ecosystem baseline

#### SJER - San Joaquin Experimental Range
- **Location**: California (37.1089°N, 119.7328°W)
- **Ecosystem**: Oak woodland
- **Available Years**: 2021
- **Research Focus**: Oak woodland ecology, Mediterranean ecosystems, biodiversity monitoring, baseline conditions
- **Crosswalk Value**: Mediterranean climate baseline

## How to Download Data

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

# Get available data for a fire case study site
site_code = "GRSM"
year = "2016"
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
# Download data for fire case study sites
make download-GRSM    # Great Smoky Mountains fire data
make download-SOAP    # Soaproot Saddle fire data
make download-SYCA    # Sycamore Creek fire data

# Download data for baseline ecosystem sites
make download-SRER    # Santa Rita Experimental Range
make download-JORN    # Jornada Experimental Range
make download-ONAQ    # Onaqui Airstrip
make download-SJER    # San Joaquin Experimental Range

# Download data for all sites
make download
```

## Data Processing

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

### Fire-Specific Processing
```bash
# Process fire case study data
make process-fire-sites

# Generate pre/post-fire comparisons
make fire-analysis

# Create fire severity maps
make severity-mapping
```

## Data Organization

### Recommended Directory Structure
```
data/
├── raw/
│   └── aop/
│       ├── GRSM/                    # Fire case study
│       │   ├── 2016/                # Pre-fire (Chimney Tops 2)
│       │   │   ├── DP3.30024.001/   # CHM
│       │   │   ├── DP3.30026.001/   # Hyperspectral
│       │   │   ├── DP3.30010.001/   # RGB
│       │   │   └── DP1.30003.001/   # LiDAR
│       │   └── 2017/                # Post-fire recovery
│       ├── SOAP/                     # Fire case study
│       │   ├── 2020/                # Pre-fire (Creek Fire)
│       │   └── 2021/                # Post-fire (Blue Fire)
│       ├── SYCA/                     # Fire case study
│       │   └── 2024/                # Fire year (Sand Stone Fire)
│       ├── SRER/                     # Baseline ecosystem
│       │   ├── 2021/
│       │   └── 2022/
│       └── JORN/                     # Baseline ecosystem
│           └── 2021/
├── processed/
│   └── aop/
│       ├── GRSM/
│       │   ├── 2016/
│       │   └── 2017/
│       ├── SOAP/
│       │   ├── 2020/
│       │   └── 2021/
│       ├── SYCA/
│       │   └── 2024/
│       ├── SRER/
│       │   ├── 2021/
│       │   └── 2022/
│       └── JORN/
│           └── 2021/
└── models/
    └── aop_crosswalk/
```

## Data Requirements

### Minimum Data for Crosswalk
- **CHM Data**: Required for canopy structure analysis
- **Hyperspectral Data**: Required for spectral calibration
- **RGB Data**: Recommended for quality assessment
- **LiDAR Data**: Optional but enhances accuracy

### Quality Thresholds
- **Cloud Cover**: Maximum 20%
- **Data Coverage**: Minimum 80% of study area
- **Temporal Gap**: Maximum 30 days between AOP and satellite data

### Fire Research Requirements
- **Pre/Post-Fire Data**: Both pre-fire and post-fire AOP collections
- **Fire Event Documentation**: Fire perimeter, severity, and timing data
- **Temporal Consistency**: AOP data within 1 year of fire events
- **Spatial Coverage**: Full fire-impacted area coverage

## Getting Started

### 1. Choose Your Research Focus

#### Fire Research (Recommended)
- Start with **GRSM** for comprehensive pre/post-fire data
- Use **SOAP** for multi-fire progression analysis
- Consider **SYCA** for contemporary fire science

#### Ecosystem Studies
- Begin with **SRER** for comprehensive data availability
- Consider **JORN** for desert ecosystem studies
- Use **ONAQ** for rangeland applications
- Select **SJER** for Mediterranean climate research

### 2. Download Data
```bash
# For fire research
make download-GRSM  # Start with Great Smoky Mountains

# For ecosystem studies
make download-SRER  # Start with Santa Rita Experimental Range
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

# For fire research: Compare pre/post-fire conditions
pre_fire_data = manager.get_enhanced_features(satellite_data, 'GRSM', 2016)
post_fire_data = manager.get_enhanced_features(satellite_data, 'GRSM', 2017)

# For ecosystem studies: Baseline analysis
baseline_data = manager.get_enhanced_features(satellite_data, 'SRER', 2021)
```

## Additional Resources

### NEON Documentation
- **Data Portal Guide**: https://data.neonscience.org/data-products/explore
- **API Documentation**: https://data.neonscience.org/api
- **Data Quality**: https://data.neonscience.org/data-products/DP3.30024.001

### Fire Research Resources
- **Fire Case Studies**: Detailed documentation in this repository
- **Fire Severity Mapping**: Guidelines for pre/post-fire analysis
- **Recovery Monitoring**: Protocols for long-term assessment

### Scientific Literature
- **NEON AOP Overview**: Kampe et al. (2010) - "NEON: the first continental-scale ecological observatory"
- **AOP Applications**: Schimel et al. (2019) - "NEON science: enabling understanding of continental-scale ecology"
- **Fire Science**: Recent publications on fire impact assessment and recovery

### Community Support
- **NEON Forums**: https://community.neonscience.org/
- **GitHub Issues**: Report bugs or request features
- **Documentation**: Comprehensive guides in this repository

## Important Notes

### Data Usage
- **Citation Required**: Always cite NEON when using their data
- **License**: Data is open but subject to NEON's terms of use
- **Attribution**: Include NEON acknowledgment in publications

### Technical Requirements
- **Storage**: AOP data can be large (several GB per site)
- **Processing**: Feature extraction requires significant computational resources
- **Memory**: Hyperspectral data processing needs adequate RAM

### Fire Research Considerations
- **Data Availability**: Ensure both pre/post-fire data exists
- **Temporal Alignment**: Match AOP data timing with fire events
- **Quality Control**: Fire-impacted areas may have data gaps
- **Validation**: Use fire perimeter data for ground truth

### Best Practices
- **Start Small**: Begin with a single site and year
- **Validate Data**: Check data quality before processing
- **Backup**: Keep raw data backups
- **Document**: Record your processing steps and parameters
- **Fire Focus**: For fire research, prioritize case study sites

---

**Need Help?** Check the main README, open a GitHub issue, or consult the NEON documentation for additional support. The fire case study sites provide excellent opportunities for advanced research applications!
