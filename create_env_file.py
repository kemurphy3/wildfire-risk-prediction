#!/usr/bin/env python3
"""
Secure .env File Creator for Wildfire Risk Prediction System

This script safely creates a .env file with your API keys without exposing them in the code.
Your API keys will be stored locally and never committed to Git.
"""

import os
import getpass
from pathlib import Path

def create_env_file():
    """Create a .env file with user-provided API keys."""
    
    print("🔐 Secure .env File Creator")
    print("=" * 50)
    print("This script will create a .env file with your API keys.")
    print("Your keys will be stored locally and never committed to Git.")
    print("You can skip any API by pressing Enter.")
    print()
    
    # Collect API keys securely
    print("🌤️  OpenWeather API Setup (FREE)")
    print("- Get your key from: https://openweathermap.org/api")
    openweather_key = getpass.getpass("Enter your OpenWeather API key (input will be hidden): ").strip()
    
    earth_engine_creds = setup_google_earth_engine()
    
    print("\n🌿 NEON Data Access Setup (Optional)")
    print("- Get your token from: https://data.neonscience.org/")
    neon_token = getpass.getpass("Enter your NEON API token (input will be hidden): ").strip()
    
    # Create .env content
    env_content = f"""# Environment Variables for Wildfire Risk Prediction System
# ⚠️  IMPORTANT: This file contains sensitive information - never commit it to Git!
# This file is automatically ignored by .gitignore

# Weather Data API (FREE)
# Get your key from: https://openweathermap.org/api
OPENWEATHER_API_KEY={openweather_key if openweather_key else ''}

# NEON Data Access (Optional)
# Get your token from: https://data.neonscience.org/
NEON_API_TOKEN={neon_token if neon_token else ''}

# Google Earth Engine (FREE for Research)
# Get access from: https://earthengine.google.com/
# After approval, run: earthengine authenticate
# For personal auth: leave empty (run 'earthengine authenticate' in terminal)
# For service account: provide path to JSON file
GOOGLE_EARTH_ENGINE_CREDENTIALS={earth_engine_creds if earth_engine_creds and earth_engine_creds != 'personal' else ''}

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_DEBUG=false

# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050
DASHBOARD_DEBUG=true

# Cache Configuration
CACHE_TTL=3600
CACHE_MAX_SIZE=10000

# Logging
LOG_LEVEL=INFO
LOG_FILE=wildfire_risk.log
"""
    
    # Write .env file
    env_file = Path('.env')
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ .env file created successfully: {env_file.absolute()}")
    print("🔒 Your API keys are now stored securely and will never be committed to Git.")
    
    # Set environment variables for current session
    if openweather_key:
        os.environ['OPENWEATHER_API_KEY'] = openweather_key
        print("✅ OpenWeather API key set for current session")
    
    if neon_token:
        os.environ['NEON_API_TOKEN'] = neon_token
        print("✅ NEON API token set for current session")
    
    if earth_engine_creds and earth_engine_creds != 'personal':
        os.environ['GOOGLE_EARTH_ENGINE_CREDENTIALS'] = earth_engine_creds
        print("✅ Earth Engine credentials path set for current session")
    elif earth_engine_creds == 'personal':
        print("✅ Earth Engine personal authentication selected")
        print("   Run 'earthengine authenticate' in your terminal to complete setup")
    
    return bool(openweather_key or neon_token or earth_engine_creds)

def setup_google_earth_engine():
    """Set up Google Earth Engine credentials."""
    print("\n🛰️  Google Earth Engine Setup (FREE for Research)")
    print("=" * 50)
    print("1. Go to: https://earthengine.google.com/")
    print("2. Click 'Sign Up' and request access")
    print("3. Wait for approval (24-48 hours)")
    print("4. After approval, choose authentication method:")
    print()
    print("Authentication Options:")
    print("A) Personal Authentication (Recommended for development)")
    print("   - Run: earthengine authenticate")
    print("   - Opens browser for Google login")
    print("   - No files needed")
    print()
    print("B) Service Account Key (.json file)")
    print("   - For server deployments")
    print("   - Requires service account setup")
    print()
    
    auth_choice = input("Choose authentication method (A/B, or press Enter to skip): ").strip().upper()
    
    if auth_choice == 'A':
        print("\n✅ Using Personal Authentication")
        print("After setup, run: earthengine authenticate")
        print("This will open your browser to complete authentication.")
        return "personal"
    elif auth_choice == 'B':
        print("\n📁 Service Account Key Setup")
        print("1. Go to Google Cloud Console > IAM & Admin > Service Accounts")
        print("2. Create a new service account with Earth Engine permissions")
        print("3. Download the JSON key file")
        json_path = input("Enter path to your service account JSON file: ").strip()
        if json_path and Path(json_path).exists():
            return json_path
        else:
            print("⚠️  Invalid file path. Using personal authentication instead.")
            return "personal"
    else:
        print("⚠️  Google Earth Engine not configured. Satellite data will use demo data.")
        return ""

def test_configuration():
    """Test the configuration to ensure everything works."""
    print("\n🧪 Testing Configuration")
    print("=" * 50)
    
    # Test OpenWeather API
    if os.environ.get('OPENWEATHER_API_KEY'):
        print("✅ OpenWeather API key: Configured")
    else:
        print("❌ OpenWeather API key: Not configured")
    
    # Test Google Earth Engine
    try:
        import ee
        # Check if authenticated
        try:
            ee.Initialize()
            print("✅ Google Earth Engine: Available and authenticated")
        except Exception:
            print("✅ Google Earth Engine: Available but not authenticated")
            print("   Run: earthengine authenticate")
    except ImportError:
        print("❌ Google Earth Engine: Not available")
        print("   Install with: pip install earthengine-api")
    
    # Test NEON API
    if os.environ.get('NEON_API_TOKEN'):
        print("✅ NEON API token: Configured")
    else:
        print("❌ NEON API token: Not configured")
    
    # Test data integration
    try:
        from src.dashboard.data_integration import RealDataIntegration
        from config import get_config
        
        config = get_config()
        di = RealDataIntegration(config)
        
        # Test a simple risk calculation
        risk_data = di.calculate_comprehensive_risk(37.7749, -122.4194)
        print(f"✅ Data Integration: Working (Test risk: {risk_data['total_risk']:.1f})")
        
    except Exception as e:
        print(f"❌ Data Integration: Error - {e}")

def main():
    """Main function."""
    print("🚀 Wildfire Risk Prediction System - Secure API Key Setup")
    print("=" * 60)
    
    # Check if .env already exists
    env_file = Path('.env')
    if env_file.exists():
        overwrite = input(f"\n⚠️  .env file already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Setup cancelled.")
            return
    
    # Create .env file
    has_real_data = create_env_file()
    
    # Test configuration
    test_configuration()
    
    # Summary
    print("\n📋 Setup Summary")
    print("=" * 50)
    print(f"OpenWeather API: {'✅ Configured' if os.environ.get('OPENWEATHER_API_KEY') else '❌ Not configured'}")
    
    # Check Earth Engine status
    try:
        import ee
        try:
            ee.Initialize()
            print("✅ Google Earth Engine: Available and authenticated")
        except Exception:
            print("✅ Google Earth Engine: Available but needs authentication")
            print("   Run: earthengine authenticate")
    except ImportError:
        print("❌ Google Earth Engine: Not available")
    
    print(f"NEON API: {'✅ Configured' if os.environ.get('NEON_API_TOKEN') else '❌ Not configured'}")
    
    if has_real_data:
        print("\n🎉 Great! You have real data sources configured.")
        print("Your dashboard will now use real environmental data!")
    else:
        print("\n⚠️  No real data sources configured.")
        print("Your dashboard will use realistic demo data based on California patterns.")
    
    print("\n🚀 Next steps:")
    print("1. Start the dashboard: python run_dashboard.py")
    print("2. Open: http://localhost:8050")
    print("3. Explore the real-time environmental monitoring!")
    
    # Earth Engine specific instructions
    if earth_engine_creds == 'personal':
        print("\n🛰️  Earth Engine Personal Authentication:")
        print("1. Run: earthengine authenticate")
        print("2. Follow the browser prompts to log in")
        print("3. Grant Earth Engine permissions")
        print("4. Your credentials will be stored locally")
    
    print("\n🔒 Security Notes:")
    print("- Your .env file is automatically ignored by Git")
    print("- API keys are stored locally only")
    print("- Never share your .env file or commit it to version control")
    
    print("\n💡 For permanent setup (Windows):")
    print("1. Open System Properties > Environment Variables")
    print("2. Add your API keys as user variables")
    print("3. Restart your terminal/IDE")
    
    print("\n💡 For permanent setup (Linux/Mac):")
    print("1. Add to ~/.bashrc or ~/.zshrc:")
    print("   export OPENWEATHER_API_KEY='your_key_here'")
    print("2. Run: source ~/.bashrc")

if __name__ == "__main__":
    main()
