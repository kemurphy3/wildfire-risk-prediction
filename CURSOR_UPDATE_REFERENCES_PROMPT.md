# Cursor AI Prompt: Update Wildfire Prediction System with Modern Research

## Context
This wildfire risk prediction system currently uses outdated scientific references from 1968-2000. We need to update it with modern research (2020-2024) while maintaining backward compatibility.

## Task Overview
Update the wildfire risk prediction system to incorporate modern research, algorithms, and data sources from 2020-2024, while keeping the existing functionality as a baseline for comparison.

## Specific Changes Required

### 1. Update Scientific References in `src/features/fire_features.py`

**Current outdated references to update:**
- Van Wagner (1987) → Keep as historical baseline, add modern alternatives
- Nelson (2000) → Update with modern ML-based moisture prediction
- Keetch & Byram (1968) → Supplement with Hot-Dry-Windy Index (2023)
- Rothermel (1972) → Add modern fire spread models

**Add these modern references in docstrings and comments:**
```python
"""
Modern Fire Risk Feature Engineering

Primary References (2020-2024):
- Jain et al. (2020): "A review of machine learning applications in wildfire science and management"
- Huot et al. (2022): "Next Day Wildfire Spread: A Machine Learning Dataset"
- Prapas et al. (2023): "Deep Learning for Global Wildfire Forecasting"
- Sayad et al. (2023): "Predictive modeling of wildfire using ML/DL algorithms"
- Michael et al. (2024): "ML for High-Resolution Predictions of Wildfire Probability"

Historical References (retained for baseline comparison):
- Van Wagner (1987): Canadian Forest Fire Weather Index System
- Nelson (2000): Prediction of dead fuel moisture content
"""
```

### 2. Add Modern Fire Indices to `FireRiskFeatureEngine` class

Add these methods to the existing class:

```python
def calculate_vapor_pressure_deficit(self, temperature: float, relative_humidity: float) -> float:
    """
    Calculate Vapor Pressure Deficit (VPD) - critical for California wildfires.
    
    VPD is a key driver of plant water stress and wildfire risk, especially
    in Mediterranean climates. Higher VPD = higher fire risk.
    
    References:
        - Williams et al. (2023): "Growing impact of wildfire on western US water supply"
        - Abatzoglou et al. (2021): "Projected increases in western US forest fire"
    
    Args:
        temperature: Air temperature in Celsius
        relative_humidity: Relative humidity (0-100%)
    
    Returns:
        VPD in kPa
    """
    # Implementation here

def calculate_hot_dry_windy_index(self, temperature: float, relative_humidity: float, 
                                  wind_speed: float) -> float:
    """
    Calculate Hot-Dry-Windy Index (HDW) - 2023 fire weather standard.
    
    HDW combines atmospheric drivers of fire spread into a single index,
    outperforming traditional indices for extreme fire weather.
    
    References:
        - Srock et al. (2023): "The Hot-Dry-Windy Index: A New Fire Weather Index"
    
    Args:
        temperature: Air temperature in Celsius
        relative_humidity: Relative humidity (0-100%)
        wind_speed: Wind speed in km/h
    
    Returns:
        HDW index value
    """
    # Implementation here

def calculate_fire_potential_index_ml(self, features: Dict[str, float]) -> float:
    """
    Calculate Fire Potential Index using ML approach (2024 method).
    
    Modern ML-based fire potential that outperforms traditional indices
    by learning complex nonlinear relationships.
    
    References:
        - Gholamnia et al. (2024): "ML approaches for wildfire susceptibility"
    
    Args:
        features: Dictionary of environmental features
    
    Returns:
        FPI value (0-100 scale)
    """
    # Implementation here
```

### 3. Update `engineer_all_features` method

Add modern features to the comprehensive feature engineering:

```python
# In engineer_all_features method, add:

# Modern indices (2020-2024 research)
features['vapor_pressure_deficit'] = self.calculate_vapor_pressure_deficit(
    weather_data['temperature'], 
    weather_data['relative_humidity']
)

features['hot_dry_windy_index'] = self.calculate_hot_dry_windy_index(
    weather_data['temperature'],
    weather_data['relative_humidity'], 
    weather_data['wind_speed']
)

# Add modern satellite-derived features
features['viirs_fire_detections'] = self.get_viirs_active_fires(location)  # New method needed
features['sentinel5p_co_levels'] = self.get_air_quality_data(location)  # New method needed
features['ecostress_water_stress'] = self.get_water_stress_index(location)  # New method needed

# Social vulnerability and WUI
features['wui_distance'] = self.calculate_wui_proximity(location)  # New method needed
features['social_vulnerability_index'] = self.get_svi_score(location)  # New method needed

# Lightning data (important for natural ignitions)
features['lightning_strike_density'] = self.get_lightning_density(location)  # New method needed
```

### 4. Add LightGBM model in `src/models/`

Create new file `src/models/lightgbm_model.py`:

```python
"""
LightGBM implementation for wildfire risk prediction.

LightGBM is a state-of-the-art gradient boosting framework that uses
tree-based learning algorithms. It's designed for efficiency and high
performance, particularly well-suited for wildfire prediction due to
its ability to handle categorical features and missing values.

References:
    - Ke et al. (2017): "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
    - Sayad et al. (2023): "Predictive modeling of wildfire using ML/DL algorithms"
"""

import lightgbm as lgb
from typing import Dict, Tuple, Optional, Union, List
import numpy as np

class LightGBMFireRiskModel:
    """Modern gradient boosting using LightGBM for wildfire risk prediction."""
    
    def __init__(self):
        # Implementation following same pattern as XGBoostFireRiskModel
        pass
```

### 5. Update `src/models/ensemble.py`

Add modern ensemble method that includes LightGBM:

```python
def create_modern_ensemble(self) -> Union[VotingRegressor, VotingClassifier]:
    """
    Create a 2024 state-of-the-art ensemble model.
    
    Combines traditional and modern approaches:
    - Random Forest (baseline)
    - XGBoost (gradient boosting)
    - LightGBM (modern gradient boosting)
    - Simple Neural Network (captures non-linear patterns)
    
    References:
        - Prapas et al. (2023): "Deep Learning for Global Wildfire Forecasting"
        - Michael et al. (2024): "ML for High-Resolution Predictions"
    """
    # Import LightGBM model
    from .lightgbm_model import LightGBMFireRiskModel
    
    # Create models
    lgb_model = LightGBMFireRiskModel(model_type=self.model_type)
    
    # Add to ensemble
    models = [
        ('rf', self.rf_model),
        ('xgb', self.xgb_model),
        ('lgb', lgb_model),
    ]
    
    # Create voting ensemble with optimized weights
    if self.model_type == 'regression':
        return VotingRegressor(models, weights=[0.3, 0.35, 0.35])
    else:
        return VotingClassifier(models, voting='soft', weights=[0.3, 0.35, 0.35])
```

### 6. Update README.md

Add section on modern research foundation:

```markdown
## 🔬 Scientific Foundation

### Modern Research (2020-2024)
This system implements cutting-edge wildfire prediction research:

- **Machine Learning Applications**: Based on Jain et al. (2020) comprehensive review
- **Deep Learning**: Implements approaches from Prapas et al. (2023) and Huot et al. (2022)
- **Satellite Integration**: Following Chuvieco et al. (2023) and Ban et al. (2020)
- **Climate Adaptation**: Incorporates Abatzoglou et al. (2021) projections

### Key Innovations
- **Vapor Pressure Deficit (VPD)**: Critical for Mediterranean climate fire risk
- **Hot-Dry-Windy Index**: 2023 standard replacing traditional indices
- **ML-based Fire Potential**: Learns complex nonlinear relationships
- **Multi-source Data Fusion**: Combines satellite, weather, and social data

### Historical Baselines
We maintain implementations of classical models for comparison:
- Canadian FWI (Van Wagner, 1987)
- KBDI drought index (Keetch & Byram, 1968)
- Rothermel spread model (1972)
```

### 7. Update imports in requirements.txt

Add modern ML library:
```
lightgbm>=3.3.0                  # State-of-the-art gradient boosting
```

## Implementation Notes

1. **Backward Compatibility**: Keep all existing methods functional
2. **Comparison Features**: Add flags to compare old vs new methods
3. **Documentation**: Clearly mark which methods are modern (2020+) vs classical
4. **Testing**: Add tests comparing classical vs modern predictions
5. **Performance**: Modern methods should show improved accuracy

## Expected Outcomes

After implementation:
- System uses 2024 research and methods
- Improved prediction accuracy with modern indices
- Portfolio demonstrates current knowledge
- Code shows evolution from classical to modern approaches

## Additional Considerations

- Add logging to track which methods (classical vs modern) are being used
- Create visualization comparing old vs new predictions
- Document performance improvements in README
- Add citations in APA format for academic credibility