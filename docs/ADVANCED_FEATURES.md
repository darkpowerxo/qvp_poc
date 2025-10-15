# QVP Platform - Advanced Features Guide

## 🚀 Overview

This guide covers the advanced features added to the QVP platform:

1. **Docker Containerization** - Production deployment
2. **Interactive Dashboards** - Plotly/Dash visualization
3. **Live Trading Simulation** - Async real-time trading
4. **WebSocket Data Feeds** - Real-time market data
5. **Risk Monitoring** - Live risk dashboard

## 📦 Docker Containerization

### Quick Start

```powershell
# Build the image
docker build -t qvp-platform .

# Run demo in container
docker run --rm -v "${PWD}/data:/app/data" qvp-platform

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f qvp-app

# Stop services
docker-compose down
```

### Services

The `docker-compose.yml` defines multiple services:

- **qvp-app** (port 8050) - Main application
- **qvp-dashboard** (port 8051) - Interactive dashboard
- **qvp-live** - Live trading simulator
- **qvp-risk-monitor** (port 8052) - Risk monitoring dashboard

### Helper Scripts

**PowerShell:**
```powershell
. .\scripts\docker-helpers.ps1
Build-QVP
Start-QVPServices
Get-QVPLogs
```

**Bash:**
```bash
source ./scripts/docker.sh
./scripts/docker.sh build
./scripts/docker.sh start
```

## 📊 Interactive Dashboard

### Launch

```powershell
uv run python -m qvp.dashboard.app
```

Access at: `http://localhost:8050`

### Features

**Overview Page:**
- Key metrics cards (Total Return, Sharpe, Max Drawdown)
- Portfolio equity curves
- Volatility analysis
- Drawdown charts
- System status

**Strategy Page:**
- Strategy selection dropdown
- Individual strategy performance
- Performance metrics table
- Returns distribution

**Risk Page:**
- VaR and CVaR metrics
- Risk limit monitoring
- Rolling risk metrics

**Analytics Page:**
- Rolling Sharpe ratio
- Rolling volatility
- Monthly returns heatmap

**Live Page:**
- Simulation controls (Start/Stop/Reset)
- Live P&L tracking
- Current positions
- Real-time market data

### Architecture

```
qvp/dashboard/
├── __init__.py
├── app.py          # Main Dash application
├── layouts.py      # Page layouts
├── callbacks.py    # Interactive callbacks
└── risk_monitor.py # Risk monitoring dashboard
```

## ⚡ Live Trading Simulation

### Quick Start

```python
import asyncio
from qvp.live.simulator import LiveSimulator
from qvp.live.async_strategy import SimpleVIXMeanReversion

async def main():
    # Create simulator
    sim = LiveSimulator(
        initial_capital=1_000_000,
        symbols=["SPY", "^VIX"],
        initial_prices={"SPY": 450.0, "^VIX": 18.0}
    )
    
    # Add strategy
    strategy = SimpleVIXMeanReversion(
        portfolio=sim.portfolio,
        vix_threshold_high=22.0,
        vix_threshold_low=16.0,
        position_size=100_000
    )
    sim.add_strategy(strategy)
    
    # Run simulation
    await sim.start(duration=60)  # Run for 60 seconds

asyncio.run(main())
```

### Or use the built-in simulator:

```powershell
uv run python -m qvp.live.simulator
```

### Components

**AsyncPortfolio** - Async portfolio management
- Thread-safe position tracking
- Real-time P&L calculation
- Equity history recording

**SimulatedDataFeed** - Market data simulation
- Geometric Brownian Motion price generation
- Configurable volatility
- Bid/ask spread simulation
- Multiple symbol support

**AsyncStrategy** - Base class for async strategies
- Real-time signal generation
- Async order execution
- Portfolio position updates

### Architecture

```
qvp/live/
├── __init__.py
├── async_portfolio.py  # Portfolio manager
├── feeds.py            # Data feeds (simulated & WebSocket)
├── async_strategy.py   # Strategy base class
└── simulator.py        # Main simulator
```

## 🌐 WebSocket Data Feeds

### Simulated Feed

```python
from qvp.live.feeds import SimulatedDataFeed, MarketData

# Create feed
feed = SimulatedDataFeed(
    symbols=["SPY", "^VIX"],
    initial_prices={"SPY": 450.0, "^VIX": 18.0},
    volatility=0.15,  # 15% annualized
    tick_interval=1.0  # 1 second per tick
)

# Subscribe to updates
async def on_tick(tick: MarketData):
    print(f"{tick.symbol}: ${tick.price:.2f}")

feed.subscribe(on_tick)

# Start feed
await feed.start()
```

### Real WebSocket Feed

```python
from qvp.live.feeds import WebSocketDataFeed

# Connect to real data provider
feed = WebSocketDataFeed(
    url="wss://api.provider.com/stream",
    symbols=["SPY", "QQQ"],
    api_key="your_api_key"
)

feed.subscribe(on_tick)
await feed.start()
```

### Features

- **Automatic reconnection** with exponential backoff
- **Message buffering** during disconnections
- **Multiple subscribers** for fan-out
- **Error handling** and logging
- **Graceful shutdown**

## 🛡️ Risk Monitoring Dashboard

### Launch

```powershell
uv run python -m qvp.dashboard.risk_monitor
```

Access at: `http://localhost:8052`

### Features

**Real-time Metrics:**
- Value at Risk (VaR 95%)
- Conditional VaR (Expected Shortfall)
- Portfolio volatility
- Maximum drawdown

**Risk Limits:**
- Configurable thresholds
- Real-time breach detection
- Alert system
- Limit status table

**Visualizations:**
- Rolling VaR/CVaR charts
- P&L distribution histogram
- Position exposure pie chart
- Stress test scenarios

**Auto-refresh:**
- Updates every 2 seconds
- Live timestamp
- Real-time data streaming

### Dark Theme

The risk monitor uses a dark theme optimized for monitoring:
- High contrast metrics
- Color-coded alerts (green/yellow/red)
- FontAwesome icons
- Responsive layout

## 🎮 Running All Demos

### Interactive Menu

```powershell
uv run python scripts/run_all_demos.py
```

This launches an interactive menu with all demos:

1. **Traditional Backtesting** - Historical data analysis
2. **Live Simulation** - 30-second async simulation
3. **Interactive Dashboard** - Full visualization dashboard
4. **Risk Monitor** - Real-time risk dashboard
5. **Docker Info** - Container deployment guide
6. **Run All** - Execute all automated demos

## 📁 Project Structure

```
qvp_poc/
├── qvp/
│   ├── dashboard/          # NEW: Dash dashboards
│   │   ├── app.py
│   │   ├── layouts.py
│   │   ├── callbacks.py
│   │   └── risk_monitor.py
│   └── live/               # NEW: Live trading
│       ├── async_portfolio.py
│       ├── feeds.py
│       ├── async_strategy.py
│       └── simulator.py
├── scripts/
│   ├── run_all_demos.py    # NEW: Demo orchestrator
│   ├── docker-helpers.ps1  # NEW: PowerShell helpers
│   └── docker.sh           # NEW: Bash helpers
├── Dockerfile              # NEW: Docker image
├── docker-compose.yml      # NEW: Multi-service orchestration
└── .dockerignore           # NEW: Docker exclusions
```

## 🔧 Requirements

### Additional Dependencies

All new dependencies are in `requirements.txt`:

```
dash>=2.14.0
dash-bootstrap-components>=1.5.0
websockets>=12.0
```

Install with:

```powershell
uv pip install -r requirements.txt
```

## 🚀 Deployment

### Production Deployment

1. **Build Production Image:**
```bash
docker build -t qvp-platform:production .
```

2. **Configure Environment:**
```bash
cp .env.template .env
# Edit .env with production settings
```

3. **Start Services:**
```bash
docker-compose up -d
```

4. **Monitor Logs:**
```bash
docker-compose logs -f
```

5. **Access Services:**
- Dashboard: http://localhost:8050
- Risk Monitor: http://localhost:8052

### Scaling

To run multiple instances:

```bash
docker-compose up -d --scale qvp-live=3
```

## 📊 Performance

### Async Benefits

- **Non-blocking I/O** - Handle multiple data feeds simultaneously
- **Concurrent strategies** - Run multiple strategies in parallel
- **Real-time processing** - Sub-millisecond latency
- **Scalability** - Handle thousands of symbols

### Optimization Tips

1. **Tune tick intervals** - Balance data freshness vs CPU usage
2. **Use connection pooling** - Reuse WebSocket connections
3. **Implement caching** - Cache frequently accessed data
4. **Monitor memory** - Use memory profiling for long runs

## 🧪 Testing

### Run Live Simulation Tests

```python
import pytest

# Test async portfolio
pytest tests/test_async_portfolio.py

# Test data feeds
pytest tests/test_feeds.py

# Test strategies
pytest tests/test_async_strategy.py
```

## 📖 API Reference

### AsyncPortfolio

```python
portfolio = AsyncPortfolio(initial_capital=1_000_000)

# Open position
await portfolio.open_position("SPY", PositionSide.LONG, 100, 450.0)

# Update price
await portfolio.update_position_price("SPY", 452.0)

# Close position
await portfolio.close_position("SPY", 455.0)

# Get stats
stats = await portfolio.get_stats()
```

### SimulatedDataFeed

```python
feed = SimulatedDataFeed(
    symbols=["SPY"],
    initial_prices={"SPY": 450.0},
    volatility=0.15,
    tick_interval=1.0
)

feed.subscribe(callback)
await feed.start()
await feed.stop()
```

### AsyncStrategy

```python
class MyStrategy(AsyncStrategy):
    async def on_market_data(self, tick: MarketData):
        # Implement strategy logic
        if tick.price > threshold:
            await self.portfolio.open_position(...)
```

## 🎯 Next Steps

1. **Customize strategies** - Implement your own async strategies
2. **Connect real data** - Integrate with live data providers
3. **Add more dashboards** - Create custom visualization pages
4. **Deploy to cloud** - Use Kubernetes for orchestration
5. **Add ML models** - Integrate machine learning signals

## 📚 Resources

- [Dash Documentation](https://dash.plotly.com/)
- [AsyncIO Guide](https://docs.python.org/3/library/asyncio.html)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [WebSocket Protocol](https://websockets.readthedocs.io/)

---

**Version:** 2.0  
**Last Updated:** October 2025  
**Author:** Sam Abtahi
