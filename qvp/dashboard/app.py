"""
Main Dash Application for QVP Platform

Interactive dashboard for backtesting results, strategy comparison,
and portfolio analytics.
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="QVP - Quantitative Volatility Platform"
)

# Server for deployment
server = app.server

# Navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Overview", href="/", id="nav-overview")),
        dbc.NavItem(dbc.NavLink("Strategies", href="/strategies", id="nav-strategies")),
        dbc.NavItem(dbc.NavLink("Risk", href="/risk", id="nav-risk")),
        dbc.NavItem(dbc.NavLink("Analytics", href="/analytics", id="nav-analytics")),
        dbc.NavItem(dbc.NavLink("Live", href="/live", id="nav-live")),
    ],
    brand="QVP Dashboard",
    brand_href="/",
    color="dark",
    dark=True,
    fluid=True,
)

# App layout with URL routing
app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Hr(),
    dbc.Container(id='page-content', fluid=True),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    )
], fluid=True, className="p-0")


# Router callback
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    """Route to different pages based on URL."""
    if pathname == '/strategies':
        from qvp.dashboard.layouts import strategy_layout
        return strategy_layout()
    elif pathname == '/risk':
        from qvp.dashboard.layouts import risk_layout
        return risk_layout()
    elif pathname == '/analytics':
        from qvp.dashboard.layouts import analytics_layout
        return analytics_layout()
    elif pathname == '/live':
        from qvp.dashboard.layouts import live_layout
        return live_layout()
    else:
        from qvp.dashboard.layouts import overview_layout
        return overview_layout()


def main():
    """Run the Dash application."""
    # Import callbacks to register them
    from qvp.dashboard import callbacks  # noqa: F401
    
    logger.info("Starting QVP Dashboard...")
    logger.info("Dashboard available at http://localhost:8050")
    app.run_server(
        debug=True,
        host='0.0.0.0',
        port=8050,
        dev_tools_hot_reload=True
    )


if __name__ == '__main__':
    main()
