# Quick Start Guide

Get your wildfire risk prediction system running in **5 minutes**!

## What You'll Have After This Guide

- **Working dashboard** with real-time risk assessment
- **Live satellite data** from Google Earth Engine
- **Weather integration** from OpenWeather
- **Machine learning models** making predictions
- **Professional API** for programmatic access

## Step 1: Setup (2 minutes)

```bash
# Clone and setup
git clone <your-repo-url>
cd wildfire-risk-prediction

# Choose your installation level:
# Full system (recommended): pip install -r requirements.txt
# Minimal setup: pip install -r requirements-minimal.txt
# Development: pip install -r requirements.txt && pip install -r requirements-dev.txt

pip install -r requirements.txt
```

## Step 2: Get API Keys (2 minutes)

### Google Earth Engine (FREE)
1. Go to [https://earthengine.google.com/](https://earthengine.google.com/)
2. Click "Sign Up" and request access
3. Wait for approval (24-48 hours)
4. Run: `earthengine authenticate`

### OpenWeather (FREE)
1. Go to [https://openweathermap.org/api](https://openweathermap.org/api)
2. Sign up for free account
3. Get your API key

### NEON (FREE)
1. Go to [https://data.neonscience.org/](https://data.neonscience.org/)
2. Request data access
3. Get your API token

## Step 3: Configure (1 minute)

```bash
python create_env_file.py
```

Enter your API keys when prompted. They'll be saved securely in a `.env` file.

## Step 4: Launch (1 minute)

```bash
# Start the dashboard
python run_dashboard.py

# Open in browser: http://localhost:8050
```

## What You Can Do Now

### Dashboard Features
- **Risk Assessment** - Interactive maps with real-time fire risk
- **Environmental Monitoring** - Live weather and satellite data
- **Predictions** - Custom risk assessments for any location
- **Model Analysis** - Feature importance and performance metrics

### API Access
```bash
# Start API server
python -m src.api.main

# API docs: http://localhost:8000/docs
```

## Testing Your Setup

```bash
# Test environment configuration
python test_env.py

# Run all tests
pytest tests/ -v
```

## Troubleshooting

### Dashboard won't start?
- Check if port 8050 is available
- Ensure all dependencies are installed
- Check console for error messages

### API keys not working?
- Verify keys are correct
- Check `.env` file exists
- Run `python test_env.py` to diagnose

### Models not loading?
- Check console for initialization messages
- Ensure TensorFlow is installed correctly
- Verify model files exist in `src/models/`

## You're Ready!

Your wildfire risk prediction system is now:
- **Fully operational** with real data
- **Production-ready** for deployment
- **Portfolio-worthy** for demonstrations
- **Scalable** for real-world use

**Next steps**: Explore the dashboard, test the API, and customize for your needs!

---

*Need help? Check the main README.md for detailed documentation.*
