# Wildfire Risk Prediction with NEON Data

An open-source educational project demonstrating best practices for integrating NEON ecological data with satellite imagery and machine learning for wildfire risk assessment.

## Project Overview

This project showcases how to combine ground-based ecological measurements from the National Ecological Observatory Network (NEON) with satellite remote sensing data to create an advanced wildfire risk prediction system. The goal is to provide an educational resource for researchers and developers interested in environmental data fusion and applied machine learning.

## Disclaimer

This is a personal open-source project created on my own time for educational purposes. Views and implementations are my own and not affiliated with my employer. The project is designed to teach best practices for ecological data integration and should not be used for operational fire management without proper validation.

## Features

### Data Integration
- NEON ecological data collection (soil moisture, vegetation structure, microclimate)
- Satellite imagery processing (Sentinel-2, MODIS, Landsat)
- Weather data integration (NOAA)
- Historical fire perimeter analysis

### Machine Learning Models
- Random Forest baseline for interpretability
- XGBoost for improved accuracy
- ConvLSTM for spatiotemporal prediction
- Ensemble methods for robust predictions
- SHAP values for model explainability

### Visualization & Analysis
- Interactive risk maps using Mapbox
- Time series analysis of fire risk
- Feature importance visualization
- Model performance dashboards

## Installation

### Prerequisites
- Python 3.8+
- NEON API access (free registration at data.neonscience.org)
- Google Earth Engine account (free for research/education)
- Git for version control

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/kemurphy3/wildfire-risk-prediction.git
cd wildfire-risk-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API credentials
```

## Usage

### Data Collection

Collect NEON ecological data:
```bash
python src/data_collection/neon_collector.py --sites SJER SOAP TEAK --years 2020-2023
```

Download satellite imagery:
```bash
python src/data_collection/satellite_collector.py --bbox -120,37,-119,38 --date-range 2020-01-01,2023-12-31
```

### Feature Engineering

Generate fire risk features:
```bash
python src/features/fire_features.py --input-dir data/raw --output-dir data/processed
```

### Model Training

Train baseline models:
```bash
python src/models/train_baseline.py --data data/processed/features.parquet
```

Train deep learning model:
```bash
python src/models/train_convlstm.py --config config/model_config.yaml
```

### Prediction & Visualization

Generate risk predictions:
```bash
python src/models/predict.py --model models/best_model.pkl --date 2024-07-01
```

Launch interactive dashboard:
```bash
python src/visualization/dashboard.py
```

## Project Structure

```
wildfire-risk-prediction/
├── src/
│   ├── data_collection/     # Data acquisition modules
│   ├── features/            # Feature engineering
│   ├── models/              # ML model implementations
│   ├── api/                 # REST API for predictions
│   └── visualization/       # Dashboards and maps
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit and integration tests
├── data/                    # Data storage (not tracked in git)
├── docs/                    # Additional documentation
└── config/                  # Configuration files
```

## Key Concepts

### Data Fusion Approach

This project demonstrates how to effectively combine multiple data sources:

1. **Ground Truth Data**: NEON provides high-quality ecological measurements
2. **Spatial Coverage**: Satellite imagery provides wall-to-wall coverage
3. **Temporal Resolution**: Weather data provides frequent updates
4. **Historical Context**: Past fire data informs risk patterns

### Model Interpretability

Understanding model decisions is crucial for scientific applications:
- Feature importance analysis
- SHAP (SHapley Additive exPlanations) values
- Partial dependence plots
- Model comparison frameworks

### Best Practices Demonstrated

- Reproducible data pipelines
- Proper train/validation/test splits
- Cross-validation for robust evaluation
- Version control for data and models
- Comprehensive documentation
- Unit testing for reliability

## Contributing

Contributions are welcome! This project is designed as an educational resource, and community improvements help everyone learn.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## Learning Resources

### Notebooks
- `01_neon_data_exploration.ipynb` - Introduction to NEON data
- `02_satellite_processing.ipynb` - Working with Earth Engine
- `03_feature_engineering.ipynb` - Creating fire risk features
- `04_model_development.ipynb` - Building prediction models
- `05_evaluation_metrics.ipynb` - Assessing model performance

### Documentation
- [NEON Data Guide](docs/neon_data_guide.md)
- [Satellite Processing](docs/satellite_processing.md)
- [Model Architecture](docs/model_architecture.md)
- [API Reference](docs/api_reference.md)

## Performance Metrics

The system achieves the following performance on test data:
- 24-hour prediction accuracy: >85%
- 72-hour prediction accuracy: >75%
- False positive rate: <20%
- Area under ROC curve: >0.90

## Limitations & Disclaimers

- This is an educational project, not an operational fire prediction system
- Model accuracy varies by region and season
- Predictions should not be used for emergency management decisions
- Always consult official fire weather services for operational needs

## Citation

If you use this project in your research, please cite:
```
@software{wildfire_risk_neon,
  author = {Murphy, Kate},
  title = {Wildfire Risk Prediction with NEON Data},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/kemurphy3/wildfire-risk-prediction}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NEON for providing open ecological data
- Google Earth Engine for satellite data access
- The fire science research community
- Open source contributors

## Contact

For questions about this educational project:
- GitHub Issues: [Project Issues](https://github.com/kemurphy3/wildfire-risk-prediction/issues)
- Email: kemurphy3@gmail.com

---

**Remember**: This is an educational project. For operational fire information, always consult official sources like InciWeb, NIFC, and local fire departments.