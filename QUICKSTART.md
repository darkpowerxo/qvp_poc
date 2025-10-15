# QVP Platform - Quick Start Guide

## Installation

Follow these steps to get started with the Quantitative Volatility Platform:

### 1. Install uv (Recommended)

**uv** is a fast Python package installer and resolver, written in Rust.

```powershell
# Install uv
pip install uv

# Or use the standalone installer (see https://github.com/astral-sh/uv)
```

### 2. Install dependencies

Using **uv** (recommended - much faster):
```powershell
# This will create .venv and install all dependencies
uv sync
```

Alternative using traditional pip:
```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment (optional)

```powershell
# Copy template
cp .env.template .env

# Edit .env with your preferences (optional)
```

### 4. Run the demo

Using **uv**:
```powershell
uv run scripts/run_demo.py
```

Or with activated venv:
```powershell
python scripts/run_demo.py
```

## Expected Output

The demo will:
1. Download SPY and VIX data (2021-2024)
2. Calculate volatility metrics
3. Run two backtests:
   - VIX Mean Reversion Strategy
   - Simple Volatility Filter Strategy
4. Display performance metrics
5. Save results to `data/results/`

## Sample Results

```
PERFORMANCE SUMMARY - VIX Mean Reversion Strategy
================================================================================
Total Return:              15.23%
Annual Return:               4.85%
Volatility:                  8.12%
Sharpe Ratio:                0.92
Sortino Ratio:               1.34
Calmar Ratio:                0.45
Max Drawdown:              -10.78%
Win Rate:                   54.23%
Profit Factor:               1.42
================================================================================
```

## Next Steps

### Explore Jupyter Notebooks

Open the interactive notebook in VS Code or Jupyter:
```powershell
# Open in VS Code (recommended)
code notebooks/01_demo_walkthrough.ipynb

# Or use Jupyter
uv run jupyter lab
# Then open notebooks/01_demo_walkthrough.ipynb
```

### Run Tests

Using **uv**:
```powershell
# Run all tests
uv run pytest tests/ -v

# With coverage report
uv run pytest tests/ --cov=qvp --cov-report=html
```

Or with activated venv:
```powershell
pytest tests/ -v
```

### Customize Configuration

Edit `config/config.yaml` to change:
- Symbols to trade
- Backtest date range
- Strategy parameters
- Risk limits
- Transaction costs

### Create Your Own Strategy

```python
from qvp.backtest import Strategy, Order, OrderType, OrderSide

class MyStrategy(Strategy):
    def initialize(self, initial_capital):
        self.capital = initial_capital
    
    def on_data(self, timestamp, data):
        orders = []
        # Your logic here
        return orders
```

## Troubleshooting

### Module Not Found Error

Make sure you're in the project root directory:
```powershell
cd qvp_poc
python scripts/run_demo.py
```

### Data Download Fails

- Check internet connection
- yfinance may be rate-limited, wait a few minutes
- Use `force_download=True` to retry

### Import Errors

Reinstall dependencies:
```powershell
pip install -r requirements.txt --force-reinstall
```

## Project Structure Overview

```
qvp_poc/
├── qvp/                    # Main package
│   ├── data/              # Data ingestion
│   ├── research/          # Volatility research
│   ├── backtest/          # Backtesting engine
│   ├── strategies/        # Trading strategies
│   ├── portfolio/         # Portfolio optimization
│   ├── risk/              # Risk management
│   └── analytics/         # Performance metrics
├── data/                  # Data storage
├── config/                # Configuration
├── scripts/               # Utility scripts
├── tests/                 # Unit tests
└── notebooks/             # Jupyter notebooks
```

## Common Tasks

### Download Fresh Data

```python
from qvp.data import DataIngester

ingester = DataIngester()
data = ingester.download_equity_data(
    symbols=['SPY', 'QQQ'],
    start_date='2021-01-01',
    end_date='2024-12-31',
    force_download=True  # Force re-download
)
```

### Calculate Volatility

```python
from qvp.research import VolatilityEstimator

vol = VolatilityEstimator.calculate_all_estimators(
    df=price_data,
    window=20
)
```

### Run Custom Backtest

```python
from qvp.backtest import BacktestEngine
from qvp.strategies import VIXMeanReversionStrategy

engine = BacktestEngine(
    initial_capital=1000000,
    start_date='2022-01-01',
    end_date='2024-12-31'
)

engine.add_data('SPY', spy_data)
engine.add_data('VIX', vix_data)

strategy = VIXMeanReversionStrategy()
results = engine.run(strategy)
```

### Calculate Performance Metrics

```python
from qvp.analytics import PerformanceMetrics

metrics = PerformanceMetrics.calculate_all_metrics(
    equity_curve=results['equity'],
    risk_free_rate=0.04
)

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2%}")
```

## Support

For questions or issues:
1. Check the documentation in README.md
2. Review example code in scripts/run_demo.py
3. Explore Jupyter notebooks for detailed examples
4. Check test files for usage patterns

## License

MIT License - See LICENSE file for details
