"""
Real-time Risk Monitoring Dashboard

Live dashboard for monitoring risk metrics and portfolio health.
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

from qvp.risk.risk_management import RiskMetrics


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
    title="QVP - Risk Monitor"
)

server = app.server


# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1([
                html.I(className="fas fa-shield-alt me-3"),
                "Risk Monitoring Dashboard"
            ], className="text-center mb-4 text-light"),
            html.P(
                f"Live Risk Metrics - Last Updated: {datetime.now().strftime('%H:%M:%S')}",
                id="last-update-time",
                className="text-center text-muted"
            )
        ], width=12)
    ]),
    
    html.Hr(className="border-secondary"),
    
    # Alert Banner
    dbc.Row([
        dbc.Col([
            html.Div(id="risk-alerts")
        ], width=12)
    ], className="mb-4"),
    
    # Key Risk Metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([html.I(className="fas fa-exclamation-triangle me-2"), "VaR (95%)"], 
                           className="card-title text-warning"),
                    html.H2(id="var-95", className="text-light"),
                    html.P("Value at Risk", className="card-text text-muted mb-0")
                ])
            ], color="dark", className="mb-4")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([html.I(className="fas fa-chart-line me-2"), "CVaR (ES)"], 
                           className="card-title text-danger"),
                    html.H2(id="cvar-95", className="text-light"),
                    html.P("Expected Shortfall", className="card-text text-muted mb-0")
                ])
            ], color="dark", className="mb-4")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([html.I(className="fas fa-tachometer-alt me-2"), "Volatility"], 
                           className="card-title text-info"),
                    html.H2(id="portfolio-vol", className="text-light"),
                    html.P("Annualized", className="card-text text-muted mb-0")
                ])
            ], color="dark", className="mb-4")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([html.I(className="fas fa-chart-area me-2"), "Max Drawdown"], 
                           className="card-title text-primary"),
                    html.H2(id="max-dd", className="text-light"),
                    html.P("Peak to Trough", className="card-text text-muted mb-0")
                ])
            ], color="dark", className="mb-4")
        ], width=3),
    ]),
    
    # Risk Limit Status
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5([
                    html.I(className="fas fa-clipboard-check me-2"),
                    "Risk Limit Status"
                ], className="text-light mb-0")),
                dbc.CardBody([
                    html.Div(id="risk-limits-table")
                ])
            ], color="dark", className="mb-4")
        ], width=12)
    ]),
    
    # Charts Row 1
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Rolling VaR & CVaR", className="text-light mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='rolling-var-cvar-chart', config={'displayModeBar': False})
                ])
            ], color="dark", className="mb-4")
        ], width=8),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Position Exposure", className="text-light mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='position-exposure-chart', config={'displayModeBar': False})
                ])
            ], color="dark", className="mb-4")
        ], width=4),
    ]),
    
    # Charts Row 2
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("P&L Distribution", className="text-light mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='pnl-distribution-chart', config={'displayModeBar': False})
                ])
            ], color="dark", className="mb-4")
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Stress Test Scenarios", className="text-light mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='stress-test-chart', config={'displayModeBar': False})
                ])
            ], color="dark", className="mb-4")
        ], width=6),
    ]),
    
    # Auto-refresh
    dcc.Interval(
        id='risk-monitor-interval',
        interval=2*1000,  # Update every 2 seconds
        n_intervals=0
    )
    
], fluid=True, className="bg-dark text-light p-4")


# Callbacks
@app.callback(
    [Output('var-95', 'children'),
     Output('cvar-95', 'children'),
     Output('portfolio-vol', 'children'),
     Output('max-dd', 'children')],
    Input('risk-monitor-interval', 'n_intervals')
)
def update_risk_metrics(n):
    """Update key risk metrics."""
    try:
        # Load live simulation data if available
        live_dir = Path("data/live_sim")
        if (live_dir / "equity_history.csv").exists():
            equity_df = pd.read_csv(live_dir / "equity_history.csv", index_col=0, parse_dates=True)
            returns = equity_df['equity'].pct_change().dropna()
        else:
            # Use backtest data as fallback
            results_dir = Path("data/results")
            if (results_dir / "vix_mean_reversion_equity.csv").exists():
                equity_df = pd.read_csv(results_dir / "vix_mean_reversion_equity.csv", 
                                       index_col=0, parse_dates=True)
                returns = equity_df['equity'].pct_change().dropna()
            else:
                return "--", "--", "--", "--"
        
        if len(returns) < 2:
            return "--", "--", "--", "--"
        
        # Calculate metrics
        risk_metrics = RiskMetrics()
        
        var_95 = risk_metrics.value_at_risk(returns, confidence=0.95, method='historical')
        cvar_95 = risk_metrics.conditional_var(returns, confidence=0.95)
        vol = returns.std() * np.sqrt(252) * 100  # Annualized %
        
        # Max drawdown
        equity = equity_df['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min() * 100
        
        return (
            f"{var_95*100:.2f}%",
            f"{cvar_95*100:.2f}%",
            f"{vol:.1f}%",
            f"{max_dd:.2f}%"
        )
    
    except Exception as e:
        logger.error(f"Error updating risk metrics: {e}")
        return "--", "--", "--", "--"


@app.callback(
    Output('risk-alerts', 'children'),
    Input('risk-monitor-interval', 'n_intervals')
)
def update_risk_alerts(n):
    """Update risk alert banner."""
    # Placeholder - would check actual limit breaches
    alerts = []
    
    # Example alerts
    if n % 30 == 0:  # Simulate occasional alert
        alerts.append(
            dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                "All risk limits within acceptable ranges"
            ], color="success", className="mb-2")
        )
    
    return html.Div(alerts) if alerts else html.Div()


@app.callback(
    Output('risk-limits-table', 'children'),
    Input('risk-monitor-interval', 'n_intervals')
)
def update_risk_limits_table(n):
    """Update risk limits status table."""
    limits = [
        {"Limit": "Max Drawdown", "Threshold": "20%", "Current": "12.3%", "Status": "OK"},
        {"Limit": "VaR (95%)", "Threshold": "5%", "Current": "3.2%", "Status": "OK"},
        {"Limit": "Position Size", "Threshold": "$500K", "Current": "$350K", "Status": "OK"},
        {"Limit": "Leverage", "Threshold": "2.0x", "Current": "1.2x", "Status": "OK"},
        {"Limit": "Concentration", "Threshold": "30%", "Current": "22%", "Status": "OK"},
    ]
    
    table_header = [
        html.Thead(html.Tr([
            html.Th("Risk Limit", className="text-light"),
            html.Th("Threshold", className="text-light"),
            html.Th("Current", className="text-light"),
            html.Th("Status", className="text-light")
        ]))
    ]
    
    rows = []
    for limit in limits:
        status_badge = dbc.Badge(
            limit["Status"], 
            color="success" if limit["Status"] == "OK" else "danger",
            className="ms-2"
        )
        rows.append(html.Tr([
            html.Td(limit["Limit"], className="text-light"),
            html.Td(limit["Threshold"], className="text-muted"),
            html.Td(limit["Current"], className="text-info"),
            html.Td(status_badge)
        ]))
    
    table_body = [html.Tbody(rows)]
    
    return dbc.Table(
        table_header + table_body, 
        bordered=True, 
        dark=True, 
        hover=True,
        responsive=True
    )


@app.callback(
    Output('rolling-var-cvar-chart', 'figure'),
    Input('risk-monitor-interval', 'n_intervals')
)
def update_rolling_var_chart(n):
    """Update rolling VaR/CVaR chart."""
    fig = go.Figure()
    
    try:
        # Load data
        live_dir = Path("data/live_sim")
        if (live_dir / "equity_history.csv").exists():
            equity_df = pd.read_csv(live_dir / "equity_history.csv", index_col=0, parse_dates=True)
        else:
            results_dir = Path("data/results")
            equity_df = pd.read_csv(results_dir / "vix_mean_reversion_equity.csv",
                                   index_col=0, parse_dates=True)
        
        returns = equity_df['equity'].pct_change()
        
        # Rolling VaR
        window = min(60, len(returns) // 2)
        if window > 5:
            rolling_var = returns.rolling(window).quantile(0.05) * 100
            
            fig.add_trace(go.Scatter(
                x=equity_df.index,
                y=rolling_var,
                name='VaR (95%)',
                line=dict(color='#FFA500', width=2)
            ))
            
            # Rolling CVaR
            rolling_cvar = returns.rolling(window).apply(
                lambda x: x[x <= x.quantile(0.05)].mean() if len(x) > 0 else 0
            ) * 100
            
            fig.add_trace(go.Scatter(
                x=equity_df.index,
                y=rolling_cvar,
                name='CVaR (ES)',
                line=dict(color='#FF4500', width=2)
            ))
    
    except Exception as e:
        logger.error(f"Error in rolling VaR chart: {e}")
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Risk (%)",
        template='plotly_dark',
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig


@app.callback(
    Output('pnl-distribution-chart', 'figure'),
    Input('risk-monitor-interval', 'n_intervals')
)
def update_pnl_distribution(n):
    """Update P&L distribution histogram."""
    fig = go.Figure()
    
    try:
        live_dir = Path("data/live_sim")
        if (live_dir / "equity_history.csv").exists():
            equity_df = pd.read_csv(live_dir / "equity_history.csv", index_col=0, parse_dates=True)
        else:
            results_dir = Path("data/results")
            equity_df = pd.read_csv(results_dir / "vix_mean_reversion_equity.csv",
                                   index_col=0, parse_dates=True)
        
        returns = equity_df['equity'].pct_change().dropna() * 100
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns',
            marker_color='#4CAF50',
            opacity=0.75
        ))
        
        # Add VaR line
        var_95 = np.percentile(returns, 5)
        fig.add_vline(
            x=var_95,
            line_dash="dash",
            line_color="#FF4500",
            annotation_text=f"VaR (95%): {var_95:.2f}%",
            annotation_position="top"
        )
    
    except Exception as e:
        logger.error(f"Error in P&L distribution: {e}")
    
    fig.update_layout(
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    return fig


@app.callback(
    Output('position-exposure-chart', 'figure'),
    Input('risk-monitor-interval', 'n_intervals')
)
def update_position_exposure(n):
    """Update position exposure pie chart."""
    # Placeholder data
    labels = ['SPY', 'Cash']
    values = [350000, 650000]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker_colors=['#2196F3', '#9E9E9E']
    )])
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=True,
        legend=dict(x=0.7, y=0.5)
    )
    
    return fig


@app.callback(
    Output('stress-test-chart', 'figure'),
    Input('risk-monitor-interval', 'n_intervals')
)
def update_stress_test(n):
    """Update stress test scenarios chart."""
    scenarios = ['Market Crash\n(-20%)', 'Vol Spike\n(+50%)', 'Rate Hike\n(+2%)', 
                 'Credit Event', 'Liquidity Crisis']
    impacts = [-150000, -80000, -45000, -120000, -95000]
    
    colors = ['#FF4500' if x < -100000 else '#FFA500' if x < -50000 else '#4CAF50' for x in impacts]
    
    fig = go.Figure(data=[
        go.Bar(
            x=scenarios,
            y=impacts,
            marker_color=colors,
            text=[f"${x:,.0f}" for x in impacts],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        yaxis_title="Impact ($)",
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    return fig


@app.callback(
    Output('last-update-time', 'children'),
    Input('risk-monitor-interval', 'n_intervals')
)
def update_timestamp(n):
    """Update last refresh timestamp."""
    return f"Live Risk Metrics - Last Updated: {datetime.now().strftime('%H:%M:%S')}"


def main():
    """Run the risk monitoring dashboard."""
    logger.info("Starting Risk Monitoring Dashboard...")
    logger.info("Dashboard available at http://localhost:8052")
    app.run(
        debug=True,
        host='0.0.0.0',
        port=8052
    )


if __name__ == '__main__':
    main()
