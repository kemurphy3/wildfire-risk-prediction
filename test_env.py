#!/usr/bin/env python3
"""
Test script to verify .env file configuration

This script tests that your API keys are properly loaded from the .env file.
"""

import os
from pathlib import Path

def test_env_file():
    """Test that .env file exists and contains expected variables."""
    print("🔍 Testing .env File Configuration")
    print("=" * 50)
    
    env_file = Path('.env')
    
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Run: python create_env_file.py")
        return False
    
    print(f"✅ .env file found: {env_file.absolute()}")
    
    # Check if .env is in .gitignore
    gitignore = Path('.gitignore')
    if gitignore.exists():
        with open(gitignore, 'r') as f:
            gitignore_content = f.read()
            if '.env' in gitignore_content:
                print("✅ .env file is properly ignored by Git")
            else:
                print("⚠️  .env file is NOT in .gitignore!")
    
    return True

def test_environment_variables():
    """Test that environment variables are loaded."""
    print("\n🌍 Testing Environment Variables")
    print("=" * 50)
    
    # Import config to get the loaded environment variables
    from config import OPENWEATHER_API_KEY, NEON_API_TOKEN, GOOGLE_EARTH_ENGINE_CREDENTIALS
    
    # Test each API key
    apis = {
        'OPENWEATHER_API_KEY': ('OpenWeather API', OPENWEATHER_API_KEY),
        'NEON_API_TOKEN': ('NEON API', NEON_API_TOKEN),
        'GOOGLE_EARTH_ENGINE_CREDENTIALS': ('Earth Engine', GOOGLE_EARTH_ENGINE_CREDENTIALS)
    }
    
    configured_apis = []
    
    for var, (name, value) in apis.items():
        if value and value != 'your_api_key_here' and value != '':
            print(f"✅ {name}: Configured")
            configured_apis.append(name)
        else:
            print(f"❌ {name}: Not configured")
    
    return configured_apis

def test_data_integration():
    """Test that data integration works with current configuration."""
    print("\n🧪 Testing Data Integration")
    print("=" * 50)
    
    try:
        from src.dashboard.data_integration import RealDataIntegration
        from config import get_config
        
        config = get_config()
        di = RealDataIntegration(config)
        
        # Test a simple risk calculation
        risk_data = di.calculate_comprehensive_risk(37.7749, -122.4194)
        print(f"✅ Data Integration: Working")
        print(f"   Test risk score: {risk_data['total_risk']:.1f}")
        print(f"   Risk category: {risk_data['risk_category']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data Integration: Error - {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Wildfire Risk Prediction System - Environment Test")
    print("=" * 60)
    
    # Test .env file
    env_ok = test_env_file()
    if not env_ok:
        return
    
    # Test environment variables
    configured_apis = test_environment_variables()
    
    # Test data integration
    integration_ok = test_data_integration()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    
    if configured_apis:
        print(f"✅ Configured APIs: {', '.join(configured_apis)}")
        print("🎉 Your system is ready for real data!")
    else:
        print("⚠️  No APIs configured - using demo data")
        print("💡 Run: python create_env_file.py to configure APIs")
    
    if integration_ok:
        print("✅ Data integration is working correctly")
    else:
        print("❌ Data integration has issues")
    
    print("\n🚀 Ready to start the dashboard!")
    print("Run: python run_dashboard.py")

if __name__ == "__main__":
    main()
