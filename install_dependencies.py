#!/usr/bin/env python3
"""
Dependency Installation Script for Wildfire Risk Prediction System

This script helps you choose and install the right dependencies for your needs.
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print a welcome banner."""
    print("=" * 70)
    print("🔥 Wildfire Risk Prediction System - Dependency Installer")
    print("=" * 70)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        print("Please upgrade Python and try again.")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_pip():
    """Check if pip is available."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ pip is not available")
        print("Please install pip and try again.")
        return False

def get_installation_choice():
    """Get user's installation preference."""
    print("Choose your installation level:")
    print("1. Full System (Recommended) - All features and dependencies")
    print("2. Minimal Setup - Core functionality only")
    print("3. Development - Full system + development tools")
    print("4. Custom - Choose specific packages")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            else:
                print("Please enter a number between 1 and 4.")
        except KeyboardInterrupt:
            print("\nInstallation cancelled.")
            sys.exit(0)

def install_requirements(requirements_file):
    """Install requirements from a specific file."""
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file {requirements_file} not found!")
        return False
    
    print(f"📦 Installing dependencies from {requirements_file}...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], 
                      check=True)
        print(f"✅ Successfully installed dependencies from {requirements_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def install_custom_packages():
    """Install custom package selection."""
    print("\n🔧 Custom Package Installation")
    print("Enter package names separated by spaces (e.g., numpy pandas tensorflow):")
    
    try:
        packages = input("Packages: ").strip().split()
        if not packages:
            print("No packages specified.")
            return False
        
        print(f"📦 Installing: {' '.join(packages)}")
        subprocess.run([sys.executable, "-m", "pip", "install"] + packages, check=True)
        print("✅ Custom packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install custom packages: {e}")
        return False
    except KeyboardInterrupt:
        print("\nCustom installation cancelled.")
        return False

def main():
    """Main installation function."""
    print_banner()
    
    # Check prerequisites
    if not check_python_version() or not check_pip():
        sys.exit(1)
    
    print()
    
    # Get user choice
    choice = get_installation_choice()
    
    # Install based on choice
    success = False
    
    if choice == '1':
        # Full system
        print("\n🚀 Installing Full System...")
        success = install_requirements("requirements.txt")
        
    elif choice == '2':
        # Minimal setup
        print("\n⚡ Installing Minimal Setup...")
        success = install_requirements("requirements-minimal.txt")
        
    elif choice == '3':
        # Development
        print("\n🛠️ Installing Development Environment...")
        if install_requirements("requirements.txt"):
            success = install_requirements("requirements-dev.txt")
        else:
            success = False
            
    elif choice == '4':
        # Custom
        success = install_custom_packages()
    
    # Final status
    print("\n" + "=" * 70)
    if success:
        print("🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Set up your API keys: python create_env_file.py")
        print("2. Launch the dashboard: python run_dashboard.py")
        print("3. Test the API: python -m src.api.main")
    else:
        print("❌ Installation failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("- Ensure you have sufficient disk space")
        print("- Try updating pip: python -m pip install --upgrade pip")
        print("- Check your internet connection")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
