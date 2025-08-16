#!/usr/bin/env python3
# starts the dashboard

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    try:
        from dashboard.app import app
        
        print("Starting Wildfire Risk Prediction Dashboard...")
        print("Initializing models and demo data...")
        
        # Use the existing app
        
        print("Dashboard ready!")
        print("Opening at: http://localhost:8050")
        print("Press Ctrl+C to stop the server")
        
        app.run(debug=True, host='0.0.0.0', port=8050)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
