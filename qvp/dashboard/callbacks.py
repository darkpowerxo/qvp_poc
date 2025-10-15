"""
Dashboard Callbacks

Interactive callback functions for the Dash dashboard.
"""

from dash import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from qvp.dashboard.app import app


def load_backtest_results():
    """Load backtest results from files."""
    try:
        data_dir = Path("data/results")
        
        results = {}
        if (data_dir / "vix_mean_reversion_equity.csv").exists():
            results['vix_mr'] = {
                'equity': pd.read_csv(data_dir / "vix_mean_reversion_equity.csv", 
                                     index_col=0, parse_dates=True),
                'metrics': pd.read_csv(data_dir / "tearsheet_vix_mr.csv")
            }
        
        if (data_dir / "simple_vol_filter_equity.csv").exists():
            results['simple_vol'] = {
                'equity': pd.read_csv(data_dir / "simple_vol_filter_equity.csv",
                                     index_col=0, parse_dates=True),
                'metrics': pd.read_csv(data_dir / "tearsheet_simple.csv")
            }
        
        return results
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return {}


# Overview Page Callbacks
@app.callback(
    [Output('total-return', 'children'),
     Output('sharpe-ratio', 'children'),
     Output('max-drawdown', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_overview_metrics(n):
    """Update overview metrics."""
    results = load_backtest_results()
    
    if 'vix_mr' in results:
        metrics = results['vix_mr']['metrics']
        total_return = metrics[metrics['Metric'] == 'total_return']['Value'].values[0]
        sharpe = metrics[metrics['Metric'] == 'sharpe_ratio']['Value'].values[0]
        max_dd = metrics[metrics['Metric'] == 'max_drawdown_pct']['Value'].values[0]
        
        return f"{total_return:.2%}", f"{sharpe:.3f}", f"{max_dd:.2%}"
    
    return "--", "--", "--"


@app.callback(
    Output('equity-curve-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_equity_curve(n):
    """Update equity curve chart."""
    results = load_backtest_results()
    
    fig = go.Figure()
    
    for strategy_name, data in results.items():
        if 'equity' in data:
            equity_df = data['equity']
            label = "VIX Mean Reversion" if strategy_name == 'vix_mr' else "Simple Vol Filter"
            
            fig.add_trace(go.Scatter(
                x=equity_df.index,
                y=equity_df['equity'],
                name=label,
                mode='lines',
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Strategy Equity Curves",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


@app.callback(
    Output('volatility-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_volatility_chart(n):
    """Update volatility chart."""
    fig = go.Figure()
    
    # Placeholder data
    dates = pd.date_range(start='2021-01-01', end='2024-12-31', freq='D')
    vol = np.random.randn(len(dates)).cumsum() * 0.01 + 0.15
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=vol,
        name='Realized Volatility',
        mode='lines',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Realized Volatility (20-day)",
        xaxis_title="Date",
        yaxis_title="Volatility",
        template='plotly_white',
        height=400
    )
    
    return fig


@app.callback(
    Output('drawdown-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_drawdown_chart(n):
    """Update drawdown chart."""
    results = load_backtest_results()
    
    fig = go.Figure()
    
    for strategy_name, data in results.items():
        if 'equity' in data:
            equity_df = data['equity']
            label = "VIX Mean Reversion" if strategy_name == 'vix_mr' else "Simple Vol Filter"
            
            # Calculate drawdown
            running_max = equity_df['equity'].expanding().max()
            drawdown = (equity_df['equity'] - running_max) / running_max * 100
            
            fig.add_trace(go.Scatter(
                x=equity_df.index,
                y=drawdown,
                name=label,
                mode='lines',
                fill='tozeroy',
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template='plotly_white',
        height=400
    )
    
    return fig


# Strategy Page Callbacks
@app.callback(
    Output('strategy-equity-curve', 'figure'),
    Input('strategy-dropdown', 'value')
)
def update_strategy_equity_curve(strategy):
    """Update strategy-specific equity curve."""
    results = load_backtest_results()
    
    fig = go.Figure()
    
    if strategy in results and 'equity' in results[strategy]:
        equity_df = results[strategy]['equity']
        
        fig.add_trace(go.Scatter(
            x=equity_df.index,
            y=equity_df['equity'],
            name='Portfolio Value',
            mode='lines',
            line=dict(color='green', width=3)
        ))
    
    fig.update_layout(
        title=f"Strategy Performance: {strategy}",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template='plotly_white',
        height=500
    )
    
    return fig


@app.callback(
    Output('strategy-metrics-table', 'children'),
    Input('strategy-dropdown', 'value')
)
def update_strategy_metrics_table(strategy):
    """Update strategy metrics table."""
    results = load_backtest_results()
    
    if strategy in results and 'metrics' in results[strategy]:
        metrics = results[strategy]['metrics']
        
        # Create table
        from dash import html
        import dash_bootstrap_components as dbc
        
        table_header = [
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")]))
        ]
        
        rows = []
        for _, row in metrics.iterrows():
            metric_name = row['Metric'].replace('_', ' ').title()
            value = row['Value']
            
            # Format value
            if 'ratio' in row['Metric'] or 'rate' in row['Metric']:
                formatted_value = f"{value:.3f}"
            elif 'return' in row['Metric'] or 'drawdown_pct' in row['Metric']:
                formatted_value = f"{value:.2%}"
            else:
                formatted_value = f"{value:,.2f}"
            
            rows.append(html.Tr([html.Td(metric_name), html.Td(formatted_value)]))
        
        table_body = [html.Tbody(rows)]
        
        return dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True)
    
    return html.P("No metrics available")


@app.callback(
    Output('returns-histogram', 'figure'),
    Input('strategy-dropdown', 'value')
)
def update_returns_histogram(strategy):
    """Update returns distribution histogram."""
    results = load_backtest_results()
    
    fig = go.Figure()
    
    if strategy in results and 'equity' in results[strategy]:
        equity_df = results[strategy]['equity']
        returns = equity_df['equity'].pct_change().dropna() * 100
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Daily Returns',
            marker_color='lightblue',
            opacity=0.75
        ))
    
    fig.update_layout(
        title="Returns Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template='plotly_white',
        height=400
    )
    
    return fig


# Risk Page Callbacks
@app.callback(
    [Output('var-metric', 'children'),
     Output('cvar-metric', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_risk_metrics(n):
    """Update risk metrics."""
    results = load_backtest_results()
    
    if 'vix_mr' in results and 'equity' in results['vix_mr']:
        equity_df = results['vix_mr']['equity']
        returns = equity_df['equity'].pct_change().dropna()
        
        # Calculate VaR and CVaR
        var_95 = np.percentile(returns, 5) * 100
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        return f"{var_95:.2f}%", f"{cvar_95:.2f}%"
    
    return "--", "--"


@app.callback(
    Output('rolling-var-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_rolling_var_chart(n):
    """Update rolling VaR chart."""
    results = load_backtest_results()
    
    fig = go.Figure()
    
    if 'vix_mr' in results and 'equity' in results['vix_mr']:
        equity_df = results['vix_mr']['equity']
        returns = equity_df['equity'].pct_change()
        
        # Calculate rolling VaR
        window = 60
        rolling_var = returns.rolling(window).quantile(0.05) * 100
        
        fig.add_trace(go.Scatter(
            x=equity_df.index,
            y=rolling_var,
            name='Rolling VaR (95%)',
            mode='lines',
            line=dict(color='red', width=2)
        ))
    
    fig.update_layout(
        title="Rolling Value at Risk (60-day window)",
        xaxis_title="Date",
        yaxis_title="VaR (%)",
        template='plotly_white',
        height=400
    )
    
    return fig


logger.info("Dashboard callbacks registered successfully")
