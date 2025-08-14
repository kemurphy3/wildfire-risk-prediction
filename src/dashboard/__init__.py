"""
Dashboard package for wildfire risk prediction visualization.

This package provides interactive web-based visualizations for the
wildfire risk prediction system, including risk maps, time series,
feature importance, and model comparison tools.
"""

from .app import create_app

__all__ = ['create_app']
