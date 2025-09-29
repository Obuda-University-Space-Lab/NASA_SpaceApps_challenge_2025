"""
Visualization Dashboard - Interactive Data Visualization
Dash-based dashboard for Terra & Luna Analytics
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


def create_dashboard(analytics_engine: Any) -> dash.Dash:
    """Create the main dashboard application."""
    
    app = dash.Dash(__name__)
    app.title = "Terra & Luna Analytics - NASA Space Apps 2025"
    
    # Define the layout
    app.layout = html.Div([
        html.Div([
            html.H1("🚀 Terra & Luna Analytics", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
            html.H3("NASA Space Apps Challenge 2025 - Óbuda University Space Lab",
                   style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 30})
        ]),
        
        # Navigation tabs
        dcc.Tabs(id="main-tabs", value='overview', children=[
            dcc.Tab(label='📊 Overview', value='overview'),
            dcc.Tab(label='🌍 Terra Analytics', value='terra'),
            dcc.Tab(label='🌙 Luna Analytics', value='luna'),
            dcc.Tab(label='🔬 Cross-Platform Analysis', value='cross-platform')
        ]),
        
        # Content area
        html.Div(id='tab-content', style={'padding': 20})
    ])
    
    @app.callback(
        Output('tab-content', 'children'),
        Input('main-tabs', 'value')
    )
    def render_tab_content(active_tab):
        """Render content based on selected tab."""
        
        if active_tab == 'overview':
            return create_overview_tab(analytics_engine)
        elif active_tab == 'terra':
            return create_terra_tab()
        elif active_tab == 'luna':
            return create_luna_tab()
        elif active_tab == 'cross-platform':
            return create_cross_platform_tab()
        
        return html.Div("Content loading...")
    
    return app


def create_overview_tab(analytics_engine: Any) -> html.Div:
    """Create the overview tab content."""
    
    # Get data summary
    summary = analytics_engine.get_data_summary()
    
    return html.Div([
        html.H2("Project Overview", style={'color': '#2c3e50'}),
        
        html.Div([
            html.Div([
                html.H4("🌍 Terra Data Sources"),
                html.P(f"Active sources: {len(summary['terra_sources'])}"),
                html.Ul([html.Li(source) for source in summary['terra_sources']] if summary['terra_sources'] else [html.Li("No data sources loaded")])
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H4("🌙 Luna Data Sources"),
                html.P(f"Active sources: {len(summary['luna_sources'])}"),
                html.Ul([html.Li(source) for source in summary['luna_sources']] if summary['luna_sources'] else [html.Li("No data sources loaded")])
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
        ]),
        
        html.Hr(),
        
        html.Div([
            html.H4("🎯 Project Goals"),
            html.Ul([
                html.Li("Analyze Earth observation data for environmental monitoring"),
                html.Li("Process lunar surface and subsurface data for mission planning"),
                html.Li("Develop cross-platform analytics for comparative studies"),
                html.Li("Create interactive visualizations for scientific exploration")
            ])
        ]),
        
        html.Div([
            html.H4("📊 Quick Stats"),
            dcc.Graph(
                figure=create_sample_overview_chart(),
                style={'height': 400}
            )
        ])
    ])


def create_terra_tab() -> html.Div:
    """Create the Terra analytics tab."""
    return html.Div([
        html.H2("🌍 Terra (Earth) Analytics", style={'color': '#27ae60'}),
        
        html.Div([
            html.H4("Environmental Indices"),
            dcc.Graph(
                figure=create_environmental_indices_chart(),
                style={'height': 400}
            )
        ]),
        
        html.Div([
            html.H4("Climate Analysis"),
            dcc.Graph(
                figure=create_climate_trends_chart(),
                style={'height': 400}
            )
        ])
    ])


def create_luna_tab() -> html.Div:
    """Create the Luna analytics tab."""
    return html.Div([
        html.H2("🌙 Luna (Moon) Analytics", style={'color': '#8e44ad'}),
        
        html.Div([
            html.H4("Surface Composition Analysis"),
            dcc.Graph(
                figure=create_lunar_composition_chart(),
                style={'height': 400}
            )
        ]),
        
        html.Div([
            html.H4("Landing Site Suitability"),
            dcc.Graph(
                figure=create_landing_site_chart(),
                style={'height': 400}
            )
        ])
    ])


def create_cross_platform_tab() -> html.Div:
    """Create the cross-platform analysis tab."""
    return html.Div([
        html.H2("🔬 Cross-Platform Analysis", style={'color': '#e74c3c'}),
        
        html.Div([
            html.H4("Terra vs Luna Comparative Analysis"),
            dcc.Graph(
                figure=create_comparison_chart(),
                style={'height': 400}
            )
        ]),
        
        html.Div([
            html.H4("Correlation Matrix"),
            dcc.Graph(
                figure=create_correlation_heatmap(),
                style={'height': 400}
            )
        ])
    ])


# Sample chart creation functions
def create_sample_overview_chart():
    """Create sample overview chart."""
    data = pd.DataFrame({
        'Category': ['Terra Datasets', 'Luna Datasets', 'Processed Analyses', 'Visualizations'],
        'Count': [5, 3, 8, 12]
    })
    
    fig = px.bar(data, x='Category', y='Count', 
                title="Project Status Overview",
                color='Category',
                color_discrete_sequence=['#27ae60', '#8e44ad', '#e74c3c', '#f39c12'])
    
    return fig


def create_environmental_indices_chart():
    """Create environmental indices chart."""
    indices = ['NDVI', 'NDWI', 'EVI', 'LST']
    values = [0.65, 0.12, 0.45, 295.5]
    
    fig = go.Figure(data=[
        go.Bar(x=indices, y=values, 
               marker_color=['#27ae60', '#3498db', '#2ecc71', '#e74c3c'])
    ])
    
    fig.update_layout(
        title="Environmental Indices - Latest Analysis",
        yaxis_title="Index Value"
    )
    
    return fig


def create_climate_trends_chart():
    """Create climate trends chart."""
    dates = pd.date_range('2020-01-01', periods=48, freq='M')
    temperature = np.cumsum(np.random.randn(48) * 0.1) + 15
    
    fig = px.line(x=dates, y=temperature,
                 title="Temperature Trends Analysis",
                 labels={'x': 'Date', 'y': 'Temperature (°C)'})
    
    return fig


def create_lunar_composition_chart():
    """Create lunar surface composition chart."""
    minerals = ['Anorthite', 'Pyroxene', 'Olivine', 'Ilmenite', 'Other']
    percentages = [45, 25, 15, 8, 7]
    
    fig = px.pie(values=percentages, names=minerals,
                title="Lunar Surface Composition Analysis",
                color_discrete_sequence=['#8e44ad', '#9b59b6', '#af7ac5', '#c39bd3', '#d7bde2'])
    
    return fig


def create_landing_site_chart():
    """Create landing site suitability chart."""
    sites = ['Site Alpha', 'Site Beta', 'Site Gamma', 'Site Delta', 'Site Epsilon']
    safety = [0.85, 0.72, 0.91, 0.68, 0.79]
    science = [0.92, 0.88, 0.76, 0.94, 0.83]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=safety, y=science, 
                           mode='markers+text',
                           text=sites,
                           textposition="top center",
                           marker=dict(size=15, color='#8e44ad'),
                           name='Landing Sites'))
    
    fig.update_layout(
        title="Landing Site Suitability Assessment",
        xaxis_title="Safety Score",
        yaxis_title="Science Value"
    )
    
    return fig


def create_comparison_chart():
    """Create Terra vs Luna comparison chart."""
    categories = ['Data Volume', 'Processing Speed', 'Analysis Complexity', 'Visualization Quality']
    terra_scores = [8.5, 7.2, 6.8, 9.1]
    luna_scores = [6.2, 8.1, 8.7, 8.3]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=terra_scores,
        theta=categories,
        fill='toself',
        name='Terra Analytics',
        line_color='#27ae60'
    ))
    fig.add_trace(go.Scatterpolar(
        r=luna_scores,
        theta=categories,
        fill='toself',
        name='Luna Analytics',
        line_color='#8e44ad'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        title="Terra vs Luna Analytics Comparison"
    )
    
    return fig


def create_correlation_heatmap():
    """Create correlation heatmap."""
    # Sample correlation data
    variables = ['Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Solar Radiation']
    correlation_matrix = np.random.rand(5, 5)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(correlation_matrix, 1)  # Set diagonal to 1
    
    fig = px.imshow(correlation_matrix,
                   x=variables,
                   y=variables,
                   color_continuous_scale='RdBu_r',
                   title="Environmental Variables Correlation Matrix")
    
    return fig