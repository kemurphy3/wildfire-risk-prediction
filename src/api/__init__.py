"""
API Module for Wildfire Risk Prediction

This module provides a FastAPI-based REST API for wildfire risk
assessment, including prediction endpoints, model management, and
comprehensive documentation.
"""

from .main import app

__all__ = ['app']
