"""
Main dashboard application for wildfire risk prediction.

This module creates a comprehensive Plotly Dash application that provides
interactive visualizations for wildfire risk assessment, including:
- Risk maps and spatial analysis
- Time series analysis
- Feature importance visualization
- Model comparison and evaluation
- Real-time prediction interface
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import base64
import io

# Import our working models
from src.models.baseline_model import RandomForestFireRiskModel
from src.models.xgboost_model import XGBoostFireRiskModel

# Configure Dash app
app = dash.Dash(__name__, 
                title="Wildfire Risk Prediction Dashboard",
                suppress_callback_exceptions=True)

# Import and register callbacks
from .callbacks import register_callbacks

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Wildfire Risk Prediction Dashboard", 
                style={'textAlign': 'center', 'color': '#2E8B57', 'marginBottom': '20px'}),
        html.P("Interactive visualization and analysis platform for wildfire risk assessment",
               style={'textAlign': 'center', 'color': '#666', 'marginBottom': '30px'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'}),
    
    # Navigation Tabs
    dcc.Tabs([
        # Risk Assessment Tab
        dcc.Tab(label='Risk Assessment', children=[
            html.Div([
                html.H3("Spatial Risk Assessment", style={'color': '#2E8B57'}),
                
                # Controls
                html.Div([
                    html.Div([
                        html.Label("Select Model:"),
                        dcc.Dropdown(
                            id='model-selector',
                            options=[
                                {'label': 'Random Forest', 'value': 'rf'},
                                {'label': 'XGBoost', 'value': 'xgb'}
                            ],
                            value='rf',
                            style={'width': '200px'}
                        )
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Label("Risk Threshold:"),
                        dcc.Slider(
                            id='risk-threshold',
                            min=0,
                            max=100,
                            step=5,
                            value=50,
                            marks={i: str(i) for i in range(0, 101, 20)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'display': 'inline-block', 'width': '300px'}),
                    
                    # Hidden input for map bounds (used by real data integration)
                    dcc.Store(id='map-bounds', data={
                        'min_lat': 32.5, 'max_lat': 42.0,
                        'min_lon': -124.5, 'max_lon': -114.0
                    })
                ], style={'marginBottom': '20px'}),
                
                # Risk Map
                dcc.Graph(
                    id='risk-map',
                    style={'height': '600px'}
                ),
                
                # Risk Statistics
                html.Div([
                    html.Div([
                        html.H4("Risk Statistics"),
                        html.Div(id='risk-stats')
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        html.H4("Risk Distribution"),
                        dcc.Graph(id='risk-distribution', style={'height': '300px'})
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                ])
            ])
        ]),
        
        # Time Series Analysis Tab
        dcc.Tab(label='Time Series', children=[
            html.Div([
                html.H3("Temporal Risk Analysis", style={'color': '#2E8B57'}),
                
                # Time Controls
                html.Div([
                    html.Div([
                        html.Label("Date Range:"),
                        dcc.DatePickerRange(
                            id='date-range',
                            start_date=(datetime.now() - timedelta(days=30)).date(),
                            end_date=datetime.now().date(),
                            display_format='MMM DD, YYYY'
                        )
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Label("Aggregation:"),
                        dcc.Dropdown(
                            id='time-aggregation',
                            options=[
                                {'label': 'Daily', 'value': 'D'},
                                {'label': 'Weekly', 'value': 'W'},
                                {'label': 'Monthly', 'value': 'M'}
                            ],
                            value='D',
                            style={'width': '150px'}
                        )
                    ], style={'display': 'inline-block'})
                ], style={'marginBottom': '20px'}),
                
                # Time Series Plot
                dcc.Graph(
                    id='time-series-plot',
                    style={'height': '500px'}
                ),
                
                # Seasonal Analysis
                html.Div([
                    html.H4("Seasonal Patterns"),
                    dcc.Graph(id='seasonal-plot', style={'height': '400px'})
                ])
            ])
        ]),
        
        # Feature Analysis Tab
        dcc.Tab(label='Feature Analysis', children=[
            html.Div([
                html.H3("Feature Importance & Analysis", style={'color': '#2E8B57'}),
                
                # Model Selection for Features
                html.Div([
                    html.Label("Select Model for Feature Analysis:"),
                    dcc.Dropdown(
                        id='feature-model-selector',
                        options=[
                            {'label': 'Random Forest', 'value': 'rf'},
                            {'label': 'XGBoost', 'value': 'xgb'}
                        ],
                        value='rf',
                        style={'width': '200px', 'marginBottom': '20px'}
                    )
                ]),
                
                # Feature Importance Plot
                dcc.Graph(
                    id='feature-importance-plot',
                    style={'height': '500px'}
                ),
                
                # Feature Correlation Matrix
                html.Div([
                    html.H4("Feature Correlations"),
                    dcc.Graph(id='correlation-matrix', style={'height': '500px'})
                ])
            ])
        ]),
        
        # Model Comparison Tab
        dcc.Tab(label='Model Comparison', children=[
            html.Div([
                html.H3("Model Performance Comparison", style={'color': '#2E8B57'}),
                
                # Performance Metrics
                html.Div([
                    html.Div([
                        html.H4("Regression Metrics"),
                        html.Div(id='regression-metrics')
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        html.H4("Classification Metrics"),
                        html.Div(id='classification-metrics')
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                ]),
                
                # Model Comparison Plot
                dcc.Graph(
                    id='model-comparison-plot',
                    style={'height': '500px'}
                ),
                
                # Prediction Comparison
                html.Div([
                    html.H4("Prediction Comparison"),
                    dcc.Graph(id='prediction-comparison', style={'height': '400px'})
                ])
            ])
        ]),
        
        # Make Predictions Tab
        dcc.Tab(label='Make Predictions', children=[
            html.Div([
                html.H3("Real-time Risk Prediction", style={'color': '#2E8B57'}),
                
                # Prediction Interface
                html.Div([
                    html.H4("Manual Risk Prediction", style={'color': '#2E8B57', 'marginBottom': '20px'}),
                    
                    # Input fields
                    html.Div([
                        html.Div([
                            html.Label("Temperature (°C):", style={'fontWeight': 'bold'}),
                            dcc.Input(id='temp-input', type='number', value=25, style={'width': '100%', 'marginBottom': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.Label("Humidity (%):", style={'fontWeight': 'bold'}),
                            dcc.Input(id='humidity-input', type='number', value=60, style={'width': '100%', 'marginBottom': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block'})
                    ]),
                    
                    html.Div([
                        html.Div([
                            html.Label("Wind Speed (km/h):", style={'fontWeight': 'bold'}),
                            dcc.Input(id='wind-input', type='number', value=15, style={'width': '100%', 'marginBottom': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.Label("Precipitation (mm):", style={'fontWeight': 'bold'}),
                            dcc.Input(id='precip-input', type='number', value=0, style={'width': '100%', 'marginBottom': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block'})
                    ]),
                    
                    html.Div([
                        html.Div([
                            html.Label("Elevation (m):", style={'fontWeight': 'bold'}),
                            dcc.Input(id='elevation-input', type='number', value=100, style={'width': '100%', 'marginBottom': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.Label("Slope (%):", style={'fontWeight': 'bold'}),
                            dcc.Input(id='slope-input', type='number', value=5, style={'width': '100%', 'marginBottom': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block'})
                    ]),
                    
                    html.Div([
                        html.Div([
                            html.Label("Vegetation Density:", style={'fontWeight': 'bold'}),
                            dcc.Input(id='veg-density-input', type='number', value=0.7, step=0.1, min=0, max=1, style={'width': '100%', 'marginBottom': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.Label("Fuel Moisture (%):", style={'fontWeight': 'bold'}),
                            dcc.Input(id='fuel-moisture-input', type='number', value=30, style={'width': '100%', 'marginBottom': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block'})
                    ]),
                    
                    html.Button('Predict Risk', id='predict-button', n_clicks=0, 
                              style={'backgroundColor': '#2E8B57', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'marginTop': '10px'}),
                    
                    html.Div(id='manual-prediction-results', style={'marginTop': '20px'})
                ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'marginBottom': '20px'}),
                
                # Location-based Prediction
                html.Div([
                    html.H4("Location-Based Risk Prediction", style={'color': '#2E8B57', 'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Div([
                            html.Label("Latitude:", style={'fontWeight': 'bold'}),
                            dcc.Input(id='pred-lat', type='number', placeholder='e.g., 37.7749', style={'width': '100%', 'marginBottom': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.Label("Longitude:", style={'fontWeight': 'bold'}),
                            dcc.Input(id='pred-lon', type='number', placeholder='e.g., -122.4194', style={'width': '100%', 'marginBottom': '10px'})
                        ], style={'width': '48%', 'display': 'inline-block'})
                    ]),
                    
                    html.Button('Calculate Risk', id='calculate-risk-btn', n_clicks=0,
                              style={'backgroundColor': '#FF6B35', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'marginTop': '10px'}),
                    
                    html.Div(id='location-prediction-results', style={'marginTop': '20px'})
                ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
            ])
        ]),
        
        # Environmental Monitoring Tab
        dcc.Tab(label='Environmental Monitoring', children=[
            html.Div([
                html.H3("Real-time Environmental Data", style={'color': '#2E8B57'}),
                
                # Location selector
                html.Div([
                    html.Div([
                        html.Label("Monitor Location:"),
                        dcc.Dropdown(
                            id='monitor-location',
                            options=[
                                {'label': 'San Francisco', 'value': 'sf'},
                                {'label': 'Los Angeles', 'value': 'la'},
                                {'label': 'San Diego', 'value': 'sd'},
                                {'label': 'Sacramento', 'value': 'sac'},
                                {'label': 'Custom Location', 'value': 'custom'}
                            ],
                            value='sf',
                            style={'width': '200px'}
                        )
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Label("Custom Lat:"),
                        dcc.Input(
                            id='custom-lat',
                            type='number',
                            placeholder='37.7749',
                            style={'width': '120px'}
                        )
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Label("Custom Lon:"),
                        dcc.Input(
                            id='custom-lon',
                            type='number',
                            placeholder='-122.4194',
                            style={'width': '120px'}
                        )
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    
                    html.Button('Update Data', id='update-env-data-btn', n_clicks=0,
                              style={'backgroundColor': '#FF6B35', 'color': 'white', 'border': 'none', 'padding': '10px 20px'})
                ], style={'marginBottom': '20px'}),
                
                # Environmental data display
                html.Div([
                    # Weather data
                    html.Div([
                        html.H4("Weather Conditions", style={'color': '#2E8B57'}),
                        html.Div(id='weather-display')
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                    
                    # Satellite data
                    html.Div([
                        html.H4("Satellite Data", style={'color': '#2E8B57'}),
                        html.Div(id='satellite-display')
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                ], style={'marginBottom': '20px'}),
                
                # Topography and fire history
                html.Div([
                    html.Div([
                        html.H4("Topographical Information", style={'color': '#2E8B57'}),
                        html.Div(id='topography-display')
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                    
                    html.Div([
                        html.H4("Fire History", style={'color': '#2E8B57'}),
                        html.Div(id='fire-history-display')
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                ])
            ])
        ])
    ], style={'marginTop': '20px'}),
    
    # Footer
    html.Div([
        html.Hr(),
        html.P("Wildfire Risk Prediction System - Built with Dash & Python",
               style={'textAlign': 'center', 'color': '#666', 'fontSize': '12px'})
    ], style={'marginTop': '40px'})
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

# Register callbacks
register_callbacks(app)

def create_app():
    """Create and return the configured Dash application."""
    # Import and register callbacks
    from .callbacks import register_callbacks
    register_callbacks(app)
    return app

if __name__ == '__main__':
    # Create and run the app
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=8050)
