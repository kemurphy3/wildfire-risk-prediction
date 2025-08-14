"""
Callback functions for the wildfire risk prediction dashboard.

This module contains all the interactive functionality for the dashboard,
including data processing, model predictions, and visualization updates.
"""

import dash
from dash import Input, Output, State, html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import our working models
from src.models.baseline_model import RandomForestFireRiskModel
from src.models.xgboost_model import XGBoostFireRiskModel

# Import real data integration
from .data_integration import RealDataIntegration
from config import get_config

# Global variables for data integration and models
DATA_INTEGRATION = None
DEMO_MODELS = {}
DEMO_DATA = None

def initialize_data_integration():
    """Initialize the real data integration system."""
    global DATA_INTEGRATION
    
    try:
        config = get_config()
        DATA_INTEGRATION = RealDataIntegration(config)
        print("Real data integration initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing data integration: {e}")
        print("Falling back to demo data only")
        return False

def initialize_demo_data():
    """Initialize demo data and models for the dashboard (fallback)."""
    global DEMO_DATA, DEMO_MODELS
    
    # Generate synthetic demo data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    # Create realistic feature names
    feature_names = [
        'temperature', 'humidity', 'wind_speed', 'precipitation',
        'fuel_moisture', 'vegetation_density', 'slope_angle', 'elevation',
        'distance_to_water', 'fire_history', 'soil_moisture', 'canopy_cover',
        'aspect', 'distance_to_road', 'population_density'
    ]
    
    # Generate synthetic features with realistic ranges
    X = np.random.randn(n_samples, n_features)
    
    # Make features more realistic
    X[:, 0] = X[:, 0] * 10 + 25  # Temperature: 15-35°C
    X[:, 1] = np.clip(X[:, 1] * 20 + 60, 20, 100)  # Humidity: 20-100%
    X[:, 2] = np.clip(X[:, 2] * 10 + 15, 0, 50)  # Wind: 0-50 km/h
    X[:, 3] = np.clip(X[:, 3] * 5 + 2, 0, 20)  # Precipitation: 0-20 mm
    X[:, 4] = np.clip(X[:, 4] * 15 + 25, 5, 50)  # Fuel moisture: 5-50%
    X[:, 5] = np.clip(X[:, 5] * 30 + 50, 10, 90)  # Vegetation: 10-90%
    X[:, 6] = np.clip(X[:, 6] * 15 + 10, 0, 45)  # Slope: 0-45°
    X[:, 7] = X[:, 7] * 500 + 1000  # Elevation: 500-1500m
    X[:, 8] = np.clip(X[:, 8] * 2000 + 5000, 100, 10000)  # Distance to water
    X[:, 9] = np.clip(X[:, 9] * 0.5 + 0.3, 0, 1)  # Fire history: 0-1
    X[:, 10] = np.clip(X[:, 10] * 20 + 40, 10, 70)  # Soil moisture: 10-70%
    X[:, 11] = np.clip(X[:, 11] * 25 + 50, 10, 90)  # Canopy cover: 10-90%
    X[:, 12] = np.clip(X[:, 12] * 90 + 180, 0, 360)  # Aspect: 0-360°
    X[:, 13] = np.clip(X[:, 12] * 5000 + 10000, 100, 20000)  # Distance to road
    X[:, 14] = np.clip(X[:, 12] * 50 + 100, 10, 200)  # Population density
    
    # Generate target variable (fire risk score 0-100)
    # Make it depend on key features
    y = (X[:, 0] * 0.3 +  # Temperature
         X[:, 1] * -0.2 +  # Humidity (negative)
         X[:, 2] * 0.4 +   # Wind speed
         X[:, 4] * -0.3 +  # Fuel moisture (negative)
         X[:, 5] * 0.2 +   # Vegetation density
         X[:, 9] * 0.5 +   # Fire history
         np.random.randn(n_samples) * 5)  # Noise
    
    # Normalize to 0-100 range
    y = np.clip((y - y.min()) / (y.max() - y.min()) * 100, 0, 100)
    
    # Convert to DataFrame
    DEMO_DATA = pd.DataFrame(X, columns=feature_names)
    DEMO_DATA['fire_risk'] = y
    
    # Add spatial coordinates for mapping
    DEMO_DATA['latitude'] = np.random.uniform(35, 45, n_samples)  # California-like
    DEMO_DATA['longitude'] = np.random.uniform(-120, -110, n_samples)
    
    # Add timestamps for time series
    base_date = datetime.now() - timedelta(days=365)
    DEMO_DATA['timestamp'] = [base_date + timedelta(days=i) for i in range(n_samples)]
    
    # Initialize and train models
    print("Initializing demo models...")
    
    # Random Forest
    rf_model = RandomForestFireRiskModel(model_type='regression', n_estimators=100)
    X_train, X_val, X_test, y_train, y_val, y_test = rf_model.prepare_data(DEMO_DATA.drop(['fire_risk', 'latitude', 'longitude', 'timestamp'], axis=1), y)
    rf_model.train(X_train, y_train, X_val, y_val)
    DEMO_MODELS['rf'] = rf_model
    
    # XGBoost
    xgb_model = XGBoostFireRiskModel(model_type='regression', n_estimators=100)
    X_train, X_val, X_test, y_train, y_val, y_test = xgb_model.prepare_data(DEMO_DATA.drop(['fire_risk', 'latitude', 'longitude', 'timestamp'], axis=1), y)
    xgb_model.train(X_train, y_train, X_val, y_val)
    DEMO_MODELS['xgb'] = xgb_model
    
    print("Demo models initialized successfully!")

def register_callbacks(app):
    """Register all callback functions with the Dash app."""
    
    # Initialize data integration (with fallback to demo)
    if not initialize_data_integration():
        initialize_demo_data()
    
    # Risk Map Callback
    @app.callback(
        [Output('risk-map', 'figure'),
         Output('risk-stats', 'children'),
         Output('risk-distribution', 'figure')],
        [Input('model-selector', 'value'),
         Input('risk-threshold', 'value'),
         Input('map-bounds', 'value')]
    )
    def update_risk_map(model_type, threshold, map_bounds):
        """Update the risk map based on model selection and threshold."""
        if DATA_INTEGRATION:
            return update_real_risk_map(threshold, map_bounds)
        else:
            return update_demo_risk_map(model_type, threshold)
    
    def update_real_risk_map(threshold, map_bounds):
        """Update risk map using real data integration."""
        try:
            # Get California bounding box
            config = get_config()
            bounds = config['geo']['bounding_box']
            
            # Get grid risk assessment
            grid_data = DATA_INTEGRATION.get_grid_risk_assessment(
                (bounds['min_lat'], bounds['max_lat'], bounds['min_lon'], bounds['max_lon']),
                grid_size=50
            )
            
            if grid_data.empty:
                return go.Figure(), "No risk data available", go.Figure()
            
            # Create risk heatmap
            fig_map = go.Figure()
            
            # Create heatmap using grid data
            risk_matrix = grid_data.pivot(index='grid_y', columns='grid_x', values='risk_score')
            
            fig_map.add_trace(go.Heatmap(
                z=risk_matrix.values,
                colorscale='RdYlGn_r',  # Red (high risk) to Green (low risk)
                zmin=0,
                zmax=100,
                colorbar=dict(title="Risk Score", x=1.1)
            ))
            
            # Update layout for California
            fig_map.update_layout(
                title="California Wildfire Risk Assessment",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=600,
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            # Risk statistics
            high_risk = np.sum(grid_data['risk_score'] >= threshold)
            total = len(grid_data)
            avg_risk = np.mean(grid_data['risk_score'])
            
            stats_html = html.Div([
                html.P(f"High Risk Areas: {high_risk}/{total} ({high_risk/total*100:.1f}%)"),
                html.P(f"Average Risk: {avg_risk:.1f}"),
                html.P(f"Max Risk: {np.max(grid_data['risk_score']):.1f}"),
                html.P(f"Min Risk: {np.min(grid_data['risk_score']):.1f}"),
                html.P(f"Data Source: Real Environmental Data", style={'color': 'green', 'fontWeight': 'bold'})
            ])
            
            # Risk distribution
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=grid_data['risk_score'],
                nbinsx=20,
                marker_color='lightcoral',
                opacity=0.7
            ))
            
            fig_dist.add_vline(x=threshold, line_dash="dash", line_color="red", 
                              annotation_text=f"Threshold: {threshold}")
            
            fig_dist.update_layout(
                title="Risk Score Distribution",
                xaxis_title="Risk Score",
                yaxis_title="Frequency",
                height=300
            )
            
            return fig_map, stats_html, fig_dist
            
        except Exception as e:
            print(f"Error updating real risk map: {e}")
            return go.Figure(), f"Error: {str(e)}", go.Figure()
    
    def update_demo_risk_map(model_type, threshold):
        """Update risk map using demo data (fallback)."""
        if model_type not in DEMO_MODELS or DEMO_DATA is None:
            return go.Figure(), "No data available", go.Figure()
        
        # Get predictions from selected model
        model = DEMO_MODELS[model_type]
        features = DEMO_DATA.drop(['fire_risk', 'latitude', 'longitude', 'timestamp'], axis=1)
        predictions = model.predict(features.values)
        
        # Create risk map
        fig_map = go.Figure()
        
        # Color by risk level
        colors = ['green' if p < threshold else 'orange' if p < threshold + 20 else 'red' for p in predictions]
        
        fig_map.add_trace(go.Scattermapbox(
            lat=DEMO_DATA['latitude'],
            lon=DEMO_DATA['longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=8,
                color=predictions,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Risk Score")
            ),
            text=[f"Risk: {p:.1f}" for p in predictions],
            hoverinfo='text'
        ))
        
        fig_map.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=40, lon=-115),
                zoom=5
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=600
        )
        
        # Risk statistics
        high_risk = np.sum(predictions >= threshold)
        total = len(predictions)
        avg_risk = np.mean(predictions)
        
        stats_html = html.Div([
            html.P(f"High Risk Areas: {high_risk}/{total} ({high_risk/total*100:.1f}%)"),
            html.P(f"Average Risk: {avg_risk:.1f}"),
            html.P(f"Max Risk: {np.max(predictions):.1f}"),
            html.P(f"Min Risk: {np.min(predictions):.1f}"),
            html.P(f"Data Source: Demo Data", style={'color': 'orange', 'fontWeight': 'bold'})
        ])
        
        # Risk distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=predictions,
            nbinsx=20,
            marker_color='lightcoral',
            opacity=0.7
        ))
        
        fig_dist.add_vline(x=threshold, line_dash="dash", line_color="red", 
                          annotation_text=f"Threshold: {threshold}")
        
        fig_dist.update_layout(
            title="Risk Score Distribution",
            xaxis_title="Risk Score",
            yaxis_title="Frequency",
            height=300
        )
        
        return fig_map, stats_html, fig_dist
    
    # Time Series Callback
    @app.callback(
        [Output('time-series-plot', 'figure'),
         Output('seasonal-plot', 'figure')],
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date'),
         Input('time-aggregation', 'value')]
    )
    def update_time_series(start_date, end_date, aggregation):
        """Update time series plots based on date range and aggregation."""
        if DEMO_DATA is None:
            return go.Figure(), go.Figure()
        
        # Filter data by date range
        start_dt = pd.to_datetime(start_date) if start_date else DEMO_DATA['timestamp'].min()
        end_dt = pd.to_datetime(end_date) if end_date else DEMO_DATA['timestamp'].max()
        
        mask = (DEMO_DATA['timestamp'] >= start_dt) & (DEMO_DATA['timestamp'] <= end_dt)
        filtered_data = DEMO_DATA[mask].copy()
        
        if len(filtered_data) == 0:
            return go.Figure(), go.Figure()
        
        # Aggregate data
        filtered_data['date'] = filtered_data['timestamp'].dt.date
        if aggregation == 'W':
            filtered_data['date'] = filtered_data['timestamp'].dt.to_period('W').dt.start_time.dt.date
        elif aggregation == 'M':
            filtered_data['date'] = filtered_data['timestamp'].dt.to_period('M').dt.start_time.dt.date
        
        aggregated = filtered_data.groupby('date')['fire_risk'].agg(['mean', 'std']).reset_index()
        
        # Time series plot
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=aggregated['date'],
            y=aggregated['mean'],
            mode='lines+markers',
            name='Average Risk',
            line=dict(color='red', width=2)
        ))
        
        # Add confidence interval
        fig_ts.add_trace(go.Scatter(
            x=aggregated['date'],
            y=aggregated['mean'] + aggregated['std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=aggregated['date'],
            y=aggregated['mean'] - aggregated['std'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            showlegend=False
        ))
        
        fig_ts.update_layout(
            title="Fire Risk Over Time",
            xaxis_title="Date",
            yaxis_title="Risk Score",
            height=500
        )
        
        # Seasonal plot
        filtered_data['month'] = filtered_data['timestamp'].dt.month
        seasonal = filtered_data.groupby('month')['fire_risk'].mean().reset_index()
        
        fig_seasonal = go.Figure()
        fig_seasonal.add_trace(go.Bar(
            x=seasonal['month'],
            y=seasonal['fire_risk'],
            marker_color='orange'
        ))
        
        fig_seasonal.update_layout(
            title="Seasonal Risk Patterns",
            xaxis_title="Month",
            yaxis_title="Average Risk",
            xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), 
                      ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
            height=400
        )
        
        return fig_ts, fig_seasonal
    
    # Feature Importance Callback
    @app.callback(
        Output('feature-importance-plot', 'figure'),
        [Input('feature-model-selector', 'value')]
    )
    def update_feature_importance(model_type):
        """Update feature importance plot based on model selection."""
        if model_type not in DEMO_MODELS:
            return go.Figure()
        
        model = DEMO_MODELS[model_type]
        
        # Get feature importance based on model type
        if model_type == 'rf':
            # Random Forest has feature_importance attribute
            if not hasattr(model, 'feature_importance') or model.feature_importance is None:
                return go.Figure()
            importance_df = model.feature_importance.copy()
        elif model_type == 'xgb':
            # XGBoost has get_feature_importance_dict method
            if not hasattr(model, 'get_feature_importance_dict'):
                return go.Figure()
            
            importance_dict = model.get_feature_importance_dict()
            if not importance_dict:
                return go.Figure()
            
            importance_df = pd.DataFrame({
                'feature': list(importance_dict.keys()),
                'importance': list(importance_dict.values())
            })
        else:
            return go.Figure()
        
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        # Create horizontal bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=importance_df['feature'],
            x=importance_df['importance'],
            orientation='h',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title=f'{model_type.upper()} Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=500,
            showlegend=False
        )
        
        return fig
    
    # Correlation Matrix Callback
    @app.callback(
        Output('correlation-matrix', 'figure'),
        [Input('feature-model-selector', 'value')]
    )
    def update_correlation_matrix(model_type):
        """Update correlation matrix plot."""
        if DEMO_DATA is None:
            return go.Figure()
        
        # Calculate correlation matrix
        features = DEMO_DATA.drop(['fire_risk', 'latitude', 'longitude', 'timestamp'], axis=1)
        corr_matrix = features.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=500
        )
        
        return fig
    
    # Model Comparison Callback
    @app.callback(
        [Output('model-comparison-plot', 'figure'),
         Output('regression-metrics', 'children'),
         Output('classification-metrics', 'children')],
        [Input('model-selector', 'value')]
    )
    def update_model_comparison(selected_model):
        """Update model comparison plots and metrics."""
        if not DEMO_MODELS or DEMO_DATA is None:
            return go.Figure(), "No models available", "No models available"
        
        # Get features and targets
        features = DEMO_DATA.drop(['fire_risk', 'latitude', 'longitude', 'timestamp'], axis=1)
        targets = DEMO_DATA['fire_risk']
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = DEMO_MODELS['rf'].prepare_data(features, targets)
        
        # Get predictions from both models
        rf_pred = DEMO_MODELS['rf'].predict(X_test)
        xgb_pred = DEMO_MODELS['xgb'].predict(X_test)
        
        # Model comparison plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_test,
            y=rf_pred,
            mode='markers',
            name='Random Forest',
            marker=dict(color='blue', size=8, opacity=0.7)
        ))
        
        fig.add_trace(go.Scatter(
            x=y_test,
            y=xgb_pred,
            mode='markers',
            name='XGBoost',
            marker=dict(color='red', size=8, opacity=0.7)
        ))
        
        # Add perfect prediction line
        min_val = min(y_test.min(), rf_pred.min(), xgb_pred.min())
        max_val = max(y_test.max(), rf_pred.max(), xgb_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='black', dash='dash')
        ))
        
        fig.update_layout(
            title="Model Prediction Comparison",
            xaxis_title="Actual Risk",
            yaxis_title="Predicted Risk",
            height=500
        )
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        
        xgb_mse = mean_squared_error(y_test, xgb_pred)
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)
        
        # Regression metrics
        reg_metrics = html.Div([
            html.H5("Random Forest:"),
            html.P(f"MSE: {rf_mse:.2f}"),
            html.P(f"MAE: {rf_mae:.2f}"),
            html.P(f"R²: {rf_r2:.3f}"),
            html.H5("XGBoost:"),
            html.P(f"MSE: {xgb_mse:.2f}"),
            html.P(f"MAE: {xgb_mae:.2f}"),
            html.P(f"R²: {xgb_r2:.3f}")
        ])
        
        # Classification metrics (convert to risk categories)
        def risk_to_category(risk):
            if risk < 30: return 0  # Low
            elif risk < 70: return 1  # Medium
            else: return 2  # High
        
        y_test_cat = np.array([risk_to_category(r) for r in y_test])
        rf_pred_cat = np.array([risk_to_category(r) for r in rf_pred])
        xgb_pred_cat = np.array([risk_to_category(r) for r in xgb_pred])
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        rf_acc = accuracy_score(y_test_cat, rf_pred_cat)
        xgb_acc = accuracy_score(y_test_cat, xgb_pred_cat)
        
        rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(y_test_cat, rf_pred_cat, average='weighted')
        xgb_prec, xgb_rec, xgb_f1, _ = precision_recall_fscore_support(y_test_cat, xgb_pred_cat, average='weighted')
        
        class_metrics = html.Div([
            html.H5("Random Forest:"),
            html.P(f"Accuracy: {rf_acc:.3f}"),
            html.P(f"Precision: {rf_prec:.3f}"),
            html.P(f"Recall: {rf_rec:.3f}"),
            html.P(f"F1-Score: {rf_f1:.3f}"),
            html.H5("XGBoost:"),
            html.P(f"Accuracy: {xgb_acc:.3f}"),
            html.P(f"Precision: {xgb_prec:.3f}"),
            html.P(f"Recall: {xgb_rec:.3f}"),
            html.P(f"F1-Score: {xgb_f1:.3f}")
        ])
        
        return fig, reg_metrics, class_metrics
    
    # Prediction Interface Callback
    @app.callback(
        Output('manual-prediction-results', 'children'),
        [Input('predict-button', 'n_clicks')],
        [State('temp-input', 'value'),
         State('humidity-input', 'value'),
         State('wind-input', 'value'),
         State('precip-input', 'value'),
         State('elevation-input', 'value'),
         State('slope-input', 'value'),
         State('veg-density-input', 'value'),
         State('fuel-moisture-input', 'value')]
    )
    def make_prediction(n_clicks, temp, humidity, wind, precip, elevation, slope, veg_density, fuel_moisture):
        """Make a prediction based on user inputs."""
        if n_clicks == 0 or not DEMO_MODELS:
            return "Enter values and click 'Predict Risk' to get started."
        
        # Create feature vector
        features = np.array([[
            temp, humidity, wind, precip, fuel_moisture, veg_density,
            slope, elevation, 5000, 0.3, 40, 50, 180, 10000, 100
        ]])
        
        # Get predictions from both models
        rf_pred = DEMO_MODELS['rf'].predict(features)[0]
        xgb_pred = DEMO_MODELS['xgb'].predict(features)[0]
        
        # Determine risk level
        def get_risk_level(risk):
            if risk < 30: return "LOW", "green"
            elif risk < 70: return "MEDIUM", "orange"
            else: return "HIGH", "red"
        
        rf_level, rf_color = get_risk_level(rf_pred)
        xgb_level, xgb_color = get_risk_level(xgb_pred)
        
        # Create results display
        results = html.Div([
            html.H4("Prediction Results", style={'color': '#2E8B57'}),
            
            html.Div([
                html.Div([
                    html.H5("Random Forest Model"),
                    html.P(f"Risk Score: {rf_pred:.1f}/100"),
                    html.P(f"Risk Level: {rf_level}", style={'color': rf_color, 'fontWeight': 'bold'})
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.H5("XGBoost Model"),
                    html.P(f"Risk Score: {xgb_pred:.1f}/100"),
                    html.P(f"Risk Level: {xgb_level}", style={'color': xgb_color, 'fontWeight': 'bold'})
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ]),
            
            html.Hr(),
            
            html.Div([
                html.H5("Recommendations:"),
                html.Ul([
                    html.Li("Monitor weather conditions closely" if rf_pred > 50 or xgb_pred > 50 else "Normal monitoring recommended"),
                    html.Li("Check fuel moisture levels" if fuel_moisture < 25 else "Fuel moisture levels are adequate"),
                    html.Li("Prepare fire response resources" if rf_pred > 70 or xgb_pred > 70 else "Standard preparedness maintained")
                ])
            ])
        ])
        
        return results

    # Make Predictions Callback
    @app.callback(
        Output('location-prediction-results', 'children'),
        [Input('calculate-risk-btn', 'n_clicks')],
        [State('pred-lat', 'value'),
         State('pred-lon', 'value')]
    )
    def calculate_risk_prediction(n_clicks, lat, lon):
        """Calculate risk prediction for a specific location."""
        if n_clicks == 0 or lat is None or lon is None:
            return "Enter coordinates and click 'Calculate Risk' to get started."
        
        try:
            if DATA_INTEGRATION:
                # Use real data integration
                risk_data = DATA_INTEGRATION.calculate_comprehensive_risk(lat, lon)
                
                if 'error' in risk_data:
                    return html.Div([
                        html.H4("Error", style={'color': 'red'}),
                        html.P(f"Could not calculate risk: {risk_data['error']}")
                    ])
                
                return html.Div([
                    html.H4(f"Risk Assessment for ({lat:.4f}, {lon:.4f})", style={'color': '#2E8B57'}),
                    html.Div([
                        html.Div([
                            html.H5("Overall Risk", style={'color': '#FF6B35'}),
                            html.P(f"Risk Score: {risk_data['total_risk']:.1f}/100"),
                            html.P(f"Risk Category: {risk_data['risk_category']}", 
                                   style={'fontWeight': 'bold', 'color': get_risk_color(risk_data['risk_category'])})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                        
                        html.Div([
                            html.H5("Risk Factors", style={'color': '#2E8B57'}),
                            html.P(f"Weather: {risk_data['risk_factors']['weather_risk']*100:.1f}%"),
                            html.P(f"Vegetation: {risk_data['risk_factors']['vegetation_risk']*100:.1f}%"),
                            html.P(f"Topography: {risk_data['risk_factors']['topography_risk']*100:.1f}%"),
                            html.P(f"Fire History: {risk_data['risk_factors']['fire_history_risk']*100:.1f}%")
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                    ]),
                    html.P(f"Data Source: Real Environmental Data", style={'color': 'green', 'fontWeight': 'bold', 'marginTop': '20px'})
                ])
            else:
                # Use demo models
                return html.Div([
                    html.H4(f"Demo Prediction for ({lat:.4f}, {lon:.4f})", style={'color': '#2E8B57'}),
                    html.P("Demo mode: Using synthetic data for demonstration"),
                    html.P(f"Estimated Risk: {np.random.randint(20, 80)}/100"),
                    html.P("Data Source: Demo Data", style={'color': 'orange', 'fontWeight': 'bold'})
                ])
                
        except Exception as e:
            return html.Div([
                html.H4("Error", style={'color': 'red'}),
                html.P(f"Could not calculate risk: {str(e)}")
            ])
    
    # Environmental Monitoring Callbacks
    @app.callback(
        [Output('weather-display', 'children'),
         Output('satellite-display', 'children'),
         Output('topography-display', 'children'),
         Output('fire-history-display', 'children')],
        [Input('update-env-data-btn', 'n_clicks'),
         Input('monitor-location', 'value')],
        [State('custom-lat', 'value'),
         State('custom-lon', 'value')]
    )
    def update_environmental_data(n_clicks, location, custom_lat, custom_lon):
        """Update environmental monitoring data for selected location."""
        if n_clicks == 0:
            return "Click 'Update Data' to load environmental information.", "", "", ""
        
        try:
            # Get coordinates based on selection
            if location == 'custom' and custom_lat is not None and custom_lon is not None:
                lat, lon = custom_lat, custom_lon
            else:
                # Predefined locations
                locations = {
                    'sf': (37.7749, -122.4194),
                    'la': (34.0522, -118.2437),
                    'sd': (32.7157, -117.1611),
                    'sac': (38.5816, -121.4944)
                }
                lat, lon = locations.get(location, (37.7749, -122.4194))
            
            if DATA_INTEGRATION:
                # Get real environmental data
                weather = DATA_INTEGRATION.get_weather_data(lat, lon)
                satellite = DATA_INTEGRATION.get_satellite_data(lat, lon)
                topography = DATA_INTEGRATION.get_topographical_data(lat, lon)
                fire_history = DATA_INTEGRATION.get_historical_fire_data(lat, lon)
                
                # Weather display
                weather_html = html.Div([
                    html.P(f"Temperature: {weather['temperature']:.1f}°C"),
                    html.P(f"Humidity: {weather['humidity']:.1f}%"),
                    html.P(f"Wind Speed: {weather['wind_speed']:.1f} m/s"),
                    html.P(f"Pressure: {weather['pressure']:.0f} hPa"),
                    html.P(f"Fire Weather Index: {weather['fire_weather_index']:.1f}/100"),
                    html.P(f"Last Updated: {weather['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                ])
                
                # Satellite display
                satellite_html = html.Div([
                    html.P(f"NDVI: {satellite['ndvi']['mean']:.3f}"),
                    html.P(f"NBR: {satellite['nbr']['mean']:.3f}"),
                    html.P(f"NDWI: {satellite['ndwi']['mean']:.3f}"),
                    html.P(f"Images Available: {satellite['image_count']}"),
                    html.P(f"Cloud Cover: {satellite['cloud_cover']:.1f}%"),
                    html.P(f"Last Update: {satellite['last_update']}")
                ])
                
                # Topography display
                topography_html = html.Div([
                    html.P(f"Elevation: {topography['elevation']:.0f} m"),
                    html.P(f"Slope: {topography['slope']:.1f}°"),
                    html.P(f"Aspect: {topography['aspect']:.1f}°"),
                    html.P(f"Roughness: {topography['roughness']:.1f} m"),
                    html.P(f"Elevation Factor: {topography['elevation_factor']:.3f}"),
                    html.P(f"Slope Factor: {topography['slope_factor']:.3f}")
                ])
                
                # Fire history display
                fire_history_html = html.Div([
                    html.P(f"Fire Frequency: {fire_history['fire_frequency']:.3f}"),
                    html.P(f"Last Fire Year: {fire_history['last_fire_year']}"),
                    html.P(f"Years Since Fire: {fire_history['years_since_fire']}"),
                    html.P(f"Fire Severity: {fire_history['fire_severity']:.3f}"),
                    html.P(f"History Score: {fire_history['fire_history_score']:.3f}")
                ])
                
                return weather_html, satellite_html, topography_html, fire_history_html
                
            else:
                # Demo data
                demo_html = html.Div([
                    html.P("Demo Mode: Using synthetic environmental data"),
                    html.P("Enable real data integration for live monitoring")
                ])
                
                return demo_html, demo_html, demo_html, demo_html
                
        except Exception as e:
            error_html = html.Div([
                html.P(f"Error loading data: {str(e)}", style={'color': 'red'})
            ])
            return error_html, error_html, error_html, error_html

def get_risk_color(category):
    """Get color for risk category."""
    colors = {
        'Low': '#00FF00',
        'Moderate': '#FFFF00', 
        'High': '#FFA500',
        'Extreme': '#FF0000'
    }
    return colors.get(category, '#666666')
