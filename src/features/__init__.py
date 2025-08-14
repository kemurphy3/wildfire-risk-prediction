"""
Feature Engineering Module for Wildfire Risk Prediction

This module provides comprehensive feature engineering capabilities for
wildfire risk assessment, including fuel moisture calculations, fire
weather indices, and topographical features.
"""

from .fire_features import FireRiskFeatureEngine

__all__ = ['FireRiskFeatureEngine']
