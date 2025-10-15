"""
Dashboard Layout Components

Defines the layout for different pages in the QVP dashboard.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime
import pandas as pd
from pathlib import Path


def overview_layout():
    """Main overview dashboard layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("QVP Platform Overview", className="text-center mb-4"),
                html.P(
                    "Quantitative Volatility Trading Platform - Real-time Monitoring & Analytics",
                    className="lead text-center text-muted"
                ),
            ], width=12)
        ]),
        
        html.Hr(),
        
        # Key Metrics Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Total Strategies", className="card-title"),
                        html.H2(id="total-strategies", children="3", className="text-primary"),
                        html.P("Active trading strategies", className="card-text text-muted")
                    ])
                ], className="mb-4")
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Total Return", className="card-title"),
                        html.H2(id="total-return", children="--", className="text-success"),
                        html.P("Year-to-date performance", className="card-text text-muted")
                    ])
                ], className="mb-4")
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Sharpe Ratio", className="card-title"),
                        html.H2(id="sharpe-ratio", children="--", className="text-info"),
                        html.P("Risk-adjusted returns", className="card-text text-muted")
                    ])
                ], className="mb-4")
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Max Drawdown", className="card-title"),
                        html.H2(id="max-drawdown", children="--", className="text-danger"),
                        html.P("Maximum decline", className="card-text text-muted")
                    ])
                ], className="mb-4")
            ], width=3),
        ]),
        
        # Main Charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Portfolio Equity Curve")),
                    dbc.CardBody([
                        dcc.Graph(id='equity-curve-chart', config={'displayModeBar': False})
                    ])
                ], className="mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Volatility Analysis")),
                    dbc.CardBody([
                        dcc.Graph(id='volatility-chart', config={'displayModeBar': False})
                    ])
                ], className="mb-4")
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Drawdown Analysis")),
                    dbc.CardBody([
                        dcc.Graph(id='drawdown-chart', config={'displayModeBar': False})
                    ])
                ], className="mb-4")
            ], width=6),
        ]),
        
        # System Status
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("System Status")),
                    dbc.CardBody([
                        html.Div(id="system-status", children=[
                            dbc.Alert([
                                html.I(className="fas fa-check-circle me-2"),
                                "All systems operational"
                            ], color="success"),
                            html.P(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                                   className="text-muted small mb-0")
                        ])
                    ])
                ], className="mb-4")
            ], width=12)
        ])
    ], fluid=True)


def strategy_layout():
    """Strategy comparison and analysis layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Strategy Analysis", className="mb-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Strategy Selection")),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='strategy-dropdown',
                            options=[
                                {'label': 'VIX Mean Reversion', 'value': 'vix_mr'},
                                {'label': 'Simple Volatility Filter', 'value': 'simple_vol'},
                                {'label': 'Volatility Risk Premium', 'value': 'vol_risk_premium'}
                            ],
                            value='vix_mr',
                            clearable=False
                        )
                    ])
                ], className="mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Strategy Performance")),
                    dbc.CardBody([
                        dcc.Graph(id='strategy-equity-curve')
                    ])
                ], className="mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Performance Metrics")),
                    dbc.CardBody([
                        html.Div(id='strategy-metrics-table')
                    ])
                ], className="mb-4")
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Returns Distribution")),
                    dbc.CardBody([
                        dcc.Graph(id='returns-histogram')
                    ])
                ], className="mb-4")
            ], width=6),
        ])
    ], fluid=True)


def risk_layout():
    """Risk monitoring dashboard layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Risk Management Dashboard", className="mb-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Value at Risk (95%)", className="card-title"),
                        html.H2(id="var-metric", children="--", className="text-warning"),
                    ])
                ], className="mb-4")
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("CVaR (ES)", className="card-title"),
                        html.H2(id="cvar-metric", children="--", className="text-warning"),
                    ])
                ], className="mb-4")
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Portfolio Beta", className="card-title"),
                        html.H2(id="beta-metric", children="--", className="text-info"),
                    ])
                ], className="mb-4")
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Leverage", className="card-title"),
                        html.H2(id="leverage-metric", children="--", className="text-primary"),
                    ])
                ], className="mb-4")
            ], width=3),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Risk Limit Monitor")),
                    dbc.CardBody([
                        html.Div(id='risk-limits-status')
                    ])
                ], className="mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Rolling VaR")),
                    dbc.CardBody([
                        dcc.Graph(id='rolling-var-chart')
                    ])
                ], className="mb-4")
            ], width=12)
        ])
    ], fluid=True)


def analytics_layout():
    """Advanced analytics layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Performance Analytics", className="mb-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Rolling Sharpe Ratio")),
                    dbc.CardBody([
                        dcc.Graph(id='rolling-sharpe-chart')
                    ])
                ], className="mb-4")
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Rolling Volatility")),
                    dbc.CardBody([
                        dcc.Graph(id='rolling-vol-chart')
                    ])
                ], className="mb-4")
            ], width=6),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Monthly Returns Heatmap")),
                    dbc.CardBody([
                        dcc.Graph(id='monthly-returns-heatmap')
                    ])
                ], className="mb-4")
            ], width=12)
        ])
    ], fluid=True)


def live_layout():
    """Live trading simulation layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Live Trading Simulation", className="mb-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Simulation Controls")),
                    dbc.CardBody([
                        dbc.ButtonGroup([
                            dbc.Button("Start", id="sim-start-btn", color="success", className="me-2"),
                            dbc.Button("Stop", id="sim-stop-btn", color="danger", className="me-2"),
                            dbc.Button("Reset", id="sim-reset-btn", color="warning"),
                        ]),
                        html.Hr(),
                        html.Div(id="sim-status", children=[
                            dbc.Badge("Stopped", color="secondary", className="me-2"),
                            html.Span("Ready to start simulation", className="text-muted")
                        ])
                    ])
                ], className="mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Live P&L")),
                    dbc.CardBody([
                        dcc.Graph(id='live-pnl-chart')
                    ])
                ], className="mb-4")
            ], width=8),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Current Positions")),
                    dbc.CardBody([
                        html.Div(id='live-positions-table')
                    ])
                ], className="mb-4")
            ], width=4),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Live Market Data Stream")),
                    dbc.CardBody([
                        dcc.Graph(id='live-market-data-chart')
                    ])
                ], className="mb-4")
            ], width=12)
        ])
    ], fluid=True)
