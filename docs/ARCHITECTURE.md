# Quantitative Volatility Platform - Architecture Documentation

**Document Version:** 1.0  
**Last Updated:** October 15, 2025  
**Author:** Solutions Architecture Team

---

## Executive Summary

The Quantitative Volatility Platform (QVP) is a production-grade, event-driven trading infrastructure designed for systematic volatility strategy research, backtesting, and simulation. Built on modern Python stack with emphasis on modularity, extensibility, and performance.

### Key Architectural Principles

- **Separation of Concerns**: Clear boundaries between data, research, execution, and analytics
- **Event-Driven Design**: Asynchronous, non-blocking architecture for realistic backtesting
- **Configuration-Driven**: Externalized configuration for environment portability
- **Production-Ready**: Comprehensive logging, error handling, and transaction cost modeling
- **Type Safety**: Full type hints for static analysis and IDE support

---

## Table of Contents

1. [System Context](#1-system-context)
2. [Container Architecture](#2-container-architecture)
3. [Component Architecture](#3-component-architecture)
4. [Data Flow Architecture](#4-data-flow-architecture)
5. [Class Design](#5-class-design)
6. [Deployment Architecture](#6-deployment-architecture)
7. [Technology Stack](#7-technology-stack)

---

## 1. System Context

### High-Level System Overview

```mermaid
C4Context
    title System Context - Quantitative Volatility Platform

    Person(trader, "Quantitative Trader", "Develops and tests volatility strategies")
    Person(researcher, "Quant Researcher", "Analyzes volatility patterns and models")
    
    System(qvp, "QVP Platform", "Systematic volatility trading infrastructure for research, backtesting, and analytics")
    
    System_Ext(yfinance, "Yahoo Finance API", "Historical market data provider")
    System_Ext(broker, "Broker API", "Live trading execution (future)")
    SystemDb_Ext(external_db, "External Data Warehouse", "Alternative data sources (future)")
    
    Rel(trader, qvp, "Runs backtests, optimizes portfolios", "Python Scripts")
    Rel(researcher, qvp, "Develops strategies, analyzes volatility", "Jupyter Notebooks")
    Rel(qvp, yfinance, "Fetches OHLCV, VIX, options data", "HTTPS/REST")
    Rel(qvp, broker, "Sends orders (future)", "FIX/REST")
    Rel(qvp, external_db, "Imports alternative data (future)", "SQL/API")
    
    UpdateRelStyle(trader, qvp, $offsetY="-40")
    UpdateRelStyle(researcher, qvp, $offsetY="-40")
    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

### System Boundaries

**In Scope:**
- Historical data ingestion and storage
- Volatility estimation (5 estimators + GARCH)
- Feature engineering and regime detection
- Event-driven backtesting engine
- Portfolio optimization and risk management
- Performance analytics and reporting

**Out of Scope (Future Phases):**
- Real-time live trading execution
- Order management system (OMS)
- Risk limit enforcement in production
- Multi-asset class support

---

## 2. Container Architecture

### Application Container View

```mermaid
C4Container
    title Container Diagram - QVP Platform Components

    Person(user, "User", "Trader/Researcher")
    
    Container_Boundary(qvp_boundary, "QVP Platform") {
        Container(scripts, "Demo Scripts", "Python", "Entry points for backtesting and demonstrations")
        Container(data_layer, "Data Layer", "Python Modules", "Data ingestion, caching, and storage")
        Container(research_layer, "Research Layer", "Python Modules", "Volatility models, GARCH, feature engineering")
        Container(backtest_engine, "Backtest Engine", "Python Modules", "Event-driven backtesting framework")
        Container(strategies, "Strategy Library", "Python Modules", "Trading strategy implementations")
        Container(portfolio_mgmt, "Portfolio Management", "Python Modules", "Optimization and position management")
        Container(risk_mgmt, "Risk Management", "Python Modules", "VaR, CVaR, stress testing")
        Container(analytics, "Analytics Engine", "Python Modules", "Performance metrics and tearsheets")
        
        ContainerDb(parquet_store, "Parquet Store", "Parquet/Snappy", "Cached market data")
        ContainerDb(results_store, "Results Store", "CSV/JSON", "Backtest results and reports")
        Container(config, "Configuration", "YAML/ENV", "System configuration and parameters")
    }
    
    System_Ext(yfinance_api, "Yahoo Finance", "Market data API")
    
    Rel(user, scripts, "Executes", "CLI")
    Rel(scripts, data_layer, "Requests data", "Function calls")
    Rel(scripts, backtest_engine, "Runs backtests", "Function calls")
    Rel(data_layer, yfinance_api, "Fetches data", "HTTPS")
    Rel(data_layer, parquet_store, "Reads/Writes", "Parquet")
    Rel(research_layer, data_layer, "Gets processed data", "Function calls")
    Rel(backtest_engine, strategies, "Executes signals", "Event callbacks")
    Rel(strategies, research_layer, "Uses indicators", "Function calls")
    Rel(backtest_engine, portfolio_mgmt, "Manages positions", "Function calls")
    Rel(backtest_engine, risk_mgmt, "Checks limits", "Function calls")
    Rel(analytics, backtest_engine, "Analyzes results", "Function calls")
    Rel(analytics, results_store, "Saves reports", "File I/O")
    Rel(data_layer, config, "Reads settings", "YAML/ENV")
    Rel(backtest_engine, config, "Reads params", "YAML/ENV")
    
    UpdateRelStyle(user, scripts, $offsetY="-30")
    UpdateRelStyle(data_layer, yfinance_api, $offsetY="-20")
```

---

## 3. Component Architecture

### Core Component Breakdown

```mermaid
C4Component
    title Component Diagram - QVP Core Components

    Container_Boundary(data_boundary, "Data Layer") {
        Component(ingestion, "DataIngester", "Python Class", "Downloads and validates market data")
        Component(storage, "DataStorage", "Python Class", "Manages Parquet storage and caching")
        Component(validation, "DataValidator", "Python Module", "Quality checks and cleaning")
    }
    
    Container_Boundary(research_boundary, "Research Layer") {
        Component(vol_estimators, "VolatilityEstimator", "Python Class", "5 volatility estimators (Close-to-Close, Parkinson, etc.)")
        Component(garch, "GARCHModeler", "Python Class", "GARCH/EGARCH/GJR-GARCH models")
        Component(features, "FeatureEngine", "Python Classes", "Technical indicators, regime detection, PCA")
        Component(implied_vol, "ImpliedVolatility", "Python Class", "Black-Scholes IV calculation")
    }
    
    Container_Boundary(backtest_boundary, "Backtesting Engine") {
        Component(engine, "BacktestEngine", "Python Class", "Event loop coordinator")
        Component(portfolio, "Portfolio", "Python Class", "Position tracking and P&L")
        Component(orders, "OrderManagement", "Python Classes", "Order/Fill dataclasses")
        Component(costs, "TransactionCostModel", "Python Class", "Commission and slippage")
    }
    
    Container_Boundary(strategy_boundary, "Strategy Library") {
        Component(base_strategy, "Strategy", "Abstract Base Class", "Strategy interface")
        Component(vix_mean_rev, "VIXMeanReversion", "Concrete Strategy", "VIX z-score based")
        Component(vol_premium, "VolRiskPremium", "Concrete Strategy", "IV-RV spread")
        Component(simple_vol, "SimpleVolFilter", "Concrete Strategy", "VIX threshold")
    }
    
    ContainerDb(parquet, "Parquet Files", "Parquet", "Cached data")
    System_Ext(yfinance, "Yahoo Finance", "Data source")
    
    Rel(ingestion, yfinance, "Downloads", "HTTPS")
    Rel(ingestion, validation, "Validates", "Function call")
    Rel(storage, parquet, "Reads/Writes", "Parquet I/O")
    Rel(ingestion, storage, "Caches", "Function call")
    
    Rel(vol_estimators, ingestion, "Gets OHLCV", "Function call")
    Rel(garch, vol_estimators, "Uses estimates", "Function call")
    Rel(features, vol_estimators, "Uses volatility", "Function call")
    Rel(implied_vol, ingestion, "Gets options data", "Function call")
    
    Rel(engine, portfolio, "Updates positions", "Method call")
    Rel(engine, orders, "Creates fills", "Constructor")
    Rel(portfolio, costs, "Calculates costs", "Method call")
    
    Rel(vix_mean_rev, base_strategy, "Implements", "Inheritance")
    Rel(vol_premium, base_strategy, "Implements", "Inheritance")
    Rel(simple_vol, base_strategy, "Implements", "Inheritance")
    Rel(engine, base_strategy, "Calls on_bar()", "Polymorphism")
    Rel(vix_mean_rev, features, "Uses indicators", "Function call")
```

---

## 4. Data Flow Architecture

### End-to-End Data Pipeline

```mermaid
flowchart TB
    subgraph External["External Data Sources"]
        YF[("Yahoo Finance API")]
    end
    
    subgraph Ingestion["Data Ingestion Layer"]
        DI["DataIngester<br/>- download_equity_data()<br/>- download_vix_data()<br/>- download_options_chain()"]
        DV["DataValidator<br/>- validate_data_quality()<br/>- clean_ohlcv_data()"]
    end
    
    subgraph Storage["Storage Layer"]
        CACHE{{"Cache Check"}}
        PARQUET[("Parquet Files<br/>(Snappy Compression)")]
    end
    
    subgraph Research["Research & Feature Engineering"]
        VOL["VolatilityEstimator<br/>5 Estimators"]
        GARCH["GARCH Models<br/>Forecasting"]
        FEAT["Feature Engine<br/>Indicators & Regimes"]
    end
    
    subgraph Strategy["Strategy Layer"]
        STRAT1["VIX Mean Reversion"]
        STRAT2["Vol Risk Premium"]
        STRAT3["Simple Vol Filter"]
    end
    
    subgraph Execution["Backtesting Engine"]
        ENGINE["BacktestEngine<br/>Event Loop"]
        PORT["Portfolio<br/>Position Tracking"]
        COST["Transaction Costs<br/>Commission + Slippage"]
    end
    
    subgraph Analytics["Analytics & Reporting"]
        METRICS["PerformanceMetrics<br/>Sharpe, Sortino, etc."]
        RISK["RiskMetrics<br/>VaR, CVaR"]
        REPORT["Tearsheet Generator"]
    end
    
    subgraph Output["Output"]
        RESULTS[("Results Store<br/>CSV/JSON")]
    end
    
    YF -->|"HTTPS Request"| DI
    DI --> DV
    DV --> CACHE
    CACHE -->|"Cache Miss"| PARQUET
    CACHE -->|"Cache Hit"| PARQUET
    PARQUET -->|"Read Data"| VOL
    
    VOL --> GARCH
    VOL --> FEAT
    GARCH --> FEAT
    
    FEAT --> STRAT1
    FEAT --> STRAT2
    FEAT --> STRAT3
    
    STRAT1 -->|"Generate Signals"| ENGINE
    STRAT2 -->|"Generate Signals"| ENGINE
    STRAT3 -->|"Generate Signals"| ENGINE
    
    ENGINE --> PORT
    PORT --> COST
    COST -->|"Update Equity"| PORT
    
    PORT -->|"Equity Curve"| METRICS
    PORT -->|"Returns"| RISK
    METRICS --> REPORT
    RISK --> REPORT
    
    REPORT --> RESULTS
    
    style External fill:#e1f5ff
    style Ingestion fill:#fff4e6
    style Storage fill:#f3e5f5
    style Research fill:#e8f5e9
    style Strategy fill:#fff3e0
    style Execution fill:#fce4ec
    style Analytics fill:#e0f2f1
    style Output fill:#f1f8e9
```

### Critical Data Flows

1. **Data Acquisition Flow**: Yahoo Finance → DataIngester → Validator → Parquet Cache
2. **Research Flow**: Parquet → Volatility Estimators → GARCH → Features → Strategies
3. **Backtest Flow**: Strategy Signals → Engine → Portfolio → Transaction Costs → Results
4. **Analytics Flow**: Portfolio Equity → Metrics Calculation → Risk Analysis → Tearsheet

---

## 5. Class Design

### Core Domain Model

```mermaid
classDiagram
    %% Configuration Management
    class Config {
        -_instance: Config
        -_config: Dict
        +get(key: str, default: Any) Any
        +set(key: str, value: Any) void
        +get_instance() Config
    }
    
    %% Data Layer
    class DataIngester {
        +download_equity_data(symbol: str, start: date, end: date) DataFrame
        +download_vix_data(start: date, end: date) DataFrame
        +download_options_chain(symbol: str, date: date) DataFrame
        -_clean_ohlcv_data(df: DataFrame) DataFrame
        +validate_data_quality(df: DataFrame) bool
    }
    
    class DataStorage {
        +save_to_parquet(df: DataFrame, filename: str) void
        +load_from_parquet(filename: str) DataFrame
        +list_cached_files() List~str~
        +get_cache_path(symbol: str) Path
    }
    
    %% Research Layer
    class VolatilityEstimator {
        +close_to_close(prices: Series, window: int) Series
        +parkinson(high: Series, low: Series, window: int) Series
        +garman_klass(ohlc: DataFrame, window: int) Series
        +rogers_satchell(ohlc: DataFrame, window: int) Series
        +yang_zhang(ohlc: DataFrame, window: int) Series
    }
    
    class GARCHModeler {
        +fit(returns: Series, model_type: str, p: int, q: int) ARCHModelResult
        +forecast(fitted_model: ARCHModelResult, horizon: int) DataFrame
        +get_model_summary(fitted_model: ARCHModelResult) str
    }
    
    class ImpliedVolatility {
        +black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) float
        +black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) float
        +implied_volatility_newton(price: float, S: float, K: float, T: float, r: float) float
    }
    
    class TechnicalIndicators {
        +rsi(prices: Series, period: int) Series
        +macd(prices: Series) Tuple~Series~
        +bollinger_bands(prices: Series, window: int, num_std: float) Tuple~Series~
        +atr(ohlc: DataFrame, period: int) Series
    }
    
    class VolatilityFeatures {
        +volatility_spread(realized_vol: Series, implied_vol: Series) Series
        +volatility_ratio(vol1: Series, vol2: Series) Series
        +volatility_zscore(vol: Series, window: int) Series
    }
    
    class RegimeDetection {
        +threshold_regimes(vol: Series, low_thresh: float, high_thresh: float) Series
        +kmeans_regimes(features: DataFrame, n_clusters: int) Series
        +quantile_regimes(vol: Series, quantiles: List~float~) Series
    }
    
    class MLFeatureEngine {
        -scaler: StandardScaler
        -pca: PCA
        +create_features(ohlc: DataFrame, vol: Series) DataFrame
        +fit_transform(features: DataFrame, n_components: int) DataFrame
    }
    
    %% Backtesting Engine
    class Order {
        <<dataclass>>
        +timestamp: datetime
        +symbol: str
        +order_type: OrderType
        +side: OrderSide
        +quantity: float
        +price: float
        +status: OrderStatus
    }
    
    class Fill {
        <<dataclass>>
        +timestamp: datetime
        +symbol: str
        +side: OrderSide
        +quantity: float
        +price: float
        +commission: float
        +slippage: float
    }
    
    class Portfolio {
        -cash: float
        -positions: Dict~str, float~
        -equity_curve: List~float~
        +process_fill(fill: Fill) void
        +get_position(symbol: str) float
        +get_equity(current_prices: Dict~str, float~) float
        +get_returns() Series
    }
    
    class TransactionCostModel {
        -commission_rate: float
        -slippage_bps: float
        +calculate_commission(fill: Fill) float
        +calculate_slippage(fill: Fill) float
        +apply_costs(fill: Fill) Fill
    }
    
    class Strategy {
        <<abstract>>
        +on_bar(timestamp: datetime, data: Dict) List~Order~*
        +on_fill(fill: Fill) void*
        #calculate_position_size(signal: float) float*
    }
    
    class BacktestEngine {
        -portfolio: Portfolio
        -strategy: Strategy
        -cost_model: TransactionCostModel
        +run(data: DataFrame, initial_capital: float) DataFrame
        -_process_bar(timestamp: datetime, bar_data: Series) void
        -_execute_orders(orders: List~Order~, current_price: float) void
    }
    
    %% Strategies
    class VIXMeanReversionStrategy {
        -lookback_period: int
        -entry_zscore: float
        -exit_zscore: float
        +on_bar(timestamp: datetime, data: Dict) List~Order~
    }
    
    class VolatilityRiskPremiumStrategy {
        -threshold: float
        -holding_period: int
        +on_bar(timestamp: datetime, data: Dict) List~Order~
    }
    
    class SimpleVolatilityStrategy {
        -vix_threshold: float
        +on_bar(timestamp: datetime, data: Dict) List~Order~
    }
    
    %% Portfolio Optimization
    class PortfolioOptimizer {
        +mean_variance_optimization(returns: DataFrame, target_return: float) ndarray
        +minimum_variance(returns: DataFrame) ndarray
        +maximum_sharpe(returns: DataFrame, risk_free_rate: float) ndarray
        +apply_constraints(weights: ndarray, constraints: Dict) ndarray
    }
    
    class RiskParityOptimizer {
        +risk_parity_weights(cov_matrix: ndarray) ndarray
        +risk_contribution(weights: ndarray, cov_matrix: ndarray) ndarray
    }
    
    %% Risk Management
    class RiskMetrics {
        +value_at_risk(returns: Series, confidence: float, method: str) float
        +conditional_var(returns: Series, confidence: float) float
        +downside_risk(returns: Series, threshold: float) float
        +max_drawdown(equity_curve: Series) float
    }
    
    class StressTesting {
        +apply_stress_scenario(portfolio: Portfolio, scenario: Dict) float
        +monte_carlo_simulation(returns: Series, n_sims: int) ndarray
    }
    
    class RiskLimitMonitor {
        -limits: Dict
        +check_limits(portfolio: Portfolio, metrics: Dict) bool
        +get_violations() List~str~
    }
    
    %% Performance Analytics
    class PerformanceMetrics {
        +sharpe_ratio(returns: Series, risk_free_rate: float) float
        +sortino_ratio(returns: Series, risk_free_rate: float) float
        +calmar_ratio(returns: Series, equity_curve: Series) float
        +max_drawdown(equity_curve: Series) float
        +win_rate(returns: Series) float
        +calculate_all_metrics(returns: Series, equity_curve: Series) Dict
    }
    
    class RollingMetrics {
        +rolling_sharpe(returns: Series, window: int) Series
        +rolling_volatility(returns: Series, window: int) Series
        +rolling_beta(returns: Series, benchmark: Series, window: int) Series
    }
    
    %% Relationships
    Config --|> DataIngester : configures
    Config --|> BacktestEngine : configures
    
    DataIngester --> DataStorage : uses
    
    VolatilityEstimator --> DataIngester : uses
    GARCHModeler --> VolatilityEstimator : uses
    TechnicalIndicators --> DataIngester : uses
    VolatilityFeatures --> VolatilityEstimator : uses
    MLFeatureEngine --> VolatilityFeatures : uses
    
    Strategy <|-- VIXMeanReversionStrategy : implements
    Strategy <|-- VolatilityRiskPremiumStrategy : implements
    Strategy <|-- SimpleVolatilityStrategy : implements
    
    VIXMeanReversionStrategy --> VolatilityFeatures : uses
    VIXMeanReversionStrategy --> TechnicalIndicators : uses
    
    BacktestEngine --> Portfolio : manages
    BacktestEngine --> Strategy : executes
    BacktestEngine --> TransactionCostModel : uses
    BacktestEngine ..> Order : creates
    
    Portfolio ..> Fill : processes
    TransactionCostModel ..> Fill : modifies
    
    PortfolioOptimizer --> RiskMetrics : uses
    RiskLimitMonitor --> RiskMetrics : uses
    
    PerformanceMetrics --> Portfolio : analyzes
    RollingMetrics --> Portfolio : analyzes
```

### Key Design Patterns

1. **Singleton Pattern**: Config class ensures single configuration instance
2. **Strategy Pattern**: Modular strategy implementations via abstract base class
3. **Factory Pattern**: Strategy instantiation based on configuration
4. **Observer Pattern**: Event-driven backtesting with callbacks
5. **Template Method**: Base strategy class with hooks for customization

---

## 6. Deployment Architecture

### Development Environment

```mermaid
C4Deployment
    title Deployment Diagram - Development Environment

    Deployment_Node(dev_machine, "Developer Workstation", "Windows/macOS/Linux") {
        Deployment_Node(python_env, "Python Environment", "Python 3.10+") {
            Container(qvp_app, "QVP Application", "Python Package", "Main application code")
            Container(jupyter, "Jupyter Server", "JupyterLab", "Interactive analysis")
        }
        
        Deployment_Node(vscode, "VS Code", "IDE") {
            Container(editor, "Code Editor", "VS Code", "Development environment")
        }
        
        Deployment_Node(filesystem, "Local Filesystem") {
            ContainerDb(data_cache, "Data Cache", "Parquet Files", "Cached market data")
            ContainerDb(results, "Results", "CSV/JSON", "Backtest outputs")
        }
    }
    
    Deployment_Node(external_services, "External Services", "Cloud") {
        System_Ext(yfinance_svc, "Yahoo Finance", "Market Data API")
    }
    
    Rel(qvp_app, data_cache, "Reads/Writes", "File I/O")
    Rel(qvp_app, results, "Writes", "File I/O")
    Rel(qvp_app, yfinance_svc, "Downloads data", "HTTPS")
    Rel(jupyter, qvp_app, "Imports", "Python")
    Rel(editor, qvp_app, "Edits", "File I/O")
```

### Future Production Environment (Conceptual)

```mermaid
flowchart TB
    subgraph Cloud["Cloud Infrastructure (AWS/Azure/GCP)"]
        subgraph Compute["Compute Layer"]
            APP["QVP Application<br/>(Container/Lambda)"]
            SCHED["Scheduler<br/>(Cron/Airflow)"]
        end
        
        subgraph Storage["Storage Layer"]
            S3["Object Storage<br/>(S3/Blob)"]
            RDS["Time-Series DB<br/>(TimescaleDB)"]
        end
        
        subgraph Monitoring["Monitoring & Logging"]
            LOGS["Centralized Logging<br/>(CloudWatch/ELK)"]
            METRICS["Metrics Dashboard<br/>(Grafana/Datadog)"]
            ALERTS["Alerting<br/>(PagerDuty)"]
        end
    end
    
    subgraph External["External Services"]
        MARKET["Market Data Provider<br/>(Bloomberg/Refinitiv)"]
        BROKER["Broker API<br/>(Interactive Brokers)"]
    end
    
    SCHED -->|"Trigger"| APP
    APP -->|"Store data"| S3
    APP -->|"Write metrics"| RDS
    APP -->|"Fetch prices"| MARKET
    APP -->|"Send orders"| BROKER
    APP -->|"Send logs"| LOGS
    RDS -->|"Query"| METRICS
    LOGS --> METRICS
    METRICS -->|"Threshold breach"| ALERTS
    
    style Cloud fill:#e3f2fd
    style Compute fill:#fff3e0
    style Storage fill:#f3e5f5
    style Monitoring fill:#e8f5e9
    style External fill:#fce4ec
```

---

## 7. Technology Stack

### Technology Decision Matrix

```mermaid
flowchart LR
    subgraph Core["Core Technologies"]
        PYTHON["Python 3.10+<br/>Type hints, async support"]
        NUMPY["NumPy<br/>Numerical computing"]
        PANDAS["pandas<br/>Data manipulation"]
        SCIPY["SciPy<br/>Statistical functions"]
    end
    
    subgraph Financial["Financial Libraries"]
        YFINANCE["yfinance<br/>Market data"]
        ARCH["arch<br/>GARCH models"]
        SKLEARN["scikit-learn<br/>ML features"]
        CVXPY["cvxpy<br/>Optimization"]
    end
    
    subgraph Infrastructure["Infrastructure"]
        PYTEST["pytest<br/>Unit testing"]
        LOGURU["loguru<br/>Logging"]
        PYYAML["PyYAML<br/>Configuration"]
        DOTENV["python-dotenv<br/>Environment"]
    end
    
    subgraph Storage["Data Storage"]
        PARQUET["Parquet<br/>Columnar format"]
        SNAPPY["Snappy<br/>Compression"]
        CSV["CSV<br/>Results export"]
    end
    
    subgraph Future["Future Additions"]
        DASH["Plotly Dash<br/>Dashboards"]
        ASYNCIO["asyncio<br/>Real-time"]
        REDIS["Redis<br/>Caching"]
        POSTGRES["PostgreSQL<br/>Persistence"]
    end
    
    Core --> Financial
    Core --> Infrastructure
    Financial --> Storage
    Infrastructure --> Storage
    Storage --> Future
    
    style Core fill:#e3f2fd
    style Financial fill:#fff3e0
    style Infrastructure fill:#f3e5f5
    style Storage fill:#e8f5e9
    style Future fill:#fce4ec
```

### Technology Justifications

| Technology | Purpose | Justification |
|------------|---------|---------------|
| **Python 3.10+** | Primary language | Type hints, performance, ecosystem |
| **NumPy/pandas** | Data processing | Industry standard, optimized C backends |
| **yfinance** | Market data | Free, reliable, comprehensive |
| **arch** | GARCH modeling | Specialized volatility forecasting |
| **cvxpy** | Optimization | Convex optimization, multiple solvers |
| **Parquet** | Data storage | Columnar, compressed, efficient queries |
| **pytest** | Testing | Fixtures, parametrization, plugins |
| **loguru** | Logging | Structured logging, rotation, threading |

---

## Appendix

### A. Configuration Schema

```yaml
# config/config.yaml structure
data:
  cache_dir: str              # Data cache directory
  default_start_date: str     # Default start for downloads
  quality_threshold: float    # Data quality minimum
  
research:
  default_window: int         # Volatility window
  garch_order: [int, int]     # GARCH(p,q)
  
backtest:
  initial_capital: float      # Starting capital
  commission_rate: float      # Per-trade commission
  slippage_bps: float        # Slippage in basis points
  
risk:
  var_confidence: float       # VaR confidence level
  max_position_size: float    # Position limit
  max_leverage: float         # Leverage limit
```

### B. Directory Structure

```
qvp_poc/
├── qvp/                    # Main package
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration management
│   ├── data/               # Data layer
│   │   ├── __init__.py
│   │   └── ingestion.py    # Data ingestion
│   ├── research/           # Research layer
│   │   ├── __init__.py
│   │   ├── volatility.py   # Volatility estimators
│   │   ├── garch.py        # GARCH models
│   │   └── features.py     # Feature engineering
│   ├── backtest/           # Backtesting
│   │   ├── __init__.py
│   │   └── engine.py       # Backtest engine
│   ├── strategies/         # Trading strategies
│   │   ├── __init__.py
│   │   └── volatility_strategies.py
│   ├── portfolio/          # Portfolio management
│   │   ├── __init__.py
│   │   └── optimization.py
│   ├── risk/               # Risk management
│   │   ├── __init__.py
│   │   └── risk_management.py
│   └── analytics/          # Performance analytics
│       ├── __init__.py
│       └── performance.py
├── scripts/                # Executable scripts
│   ├── run_demo.py
│   └── example_volatility.py
├── tests/                  # Unit tests
│   ├── conftest.py
│   ├── test_volatility.py
│   └── test_backtest.py
├── config/                 # Configuration files
│   └── config.yaml
├── data/                   # Data directory (git-ignored)
│   ├── cache/
│   └── results/
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # This file
│   └── API.md
├── pyproject.toml          # Project metadata
├── requirements.txt        # Dependencies
├── .env                    # Environment variables
└── README.md               # Main documentation
```

### C. Glossary

- **GARCH**: Generalized Autoregressive Conditional Heteroskedasticity
- **VaR**: Value at Risk
- **CVaR**: Conditional Value at Risk (Expected Shortfall)
- **IV**: Implied Volatility
- **RV**: Realized Volatility
- **VIX**: CBOE Volatility Index
- **PCA**: Principal Component Analysis
- **OHLCV**: Open, High, Low, Close, Volume
- **P&L**: Profit and Loss
- **Parquet**: Apache Parquet columnar storage format
- **Snappy**: Google's compression algorithm

---

**Document Status:** Production-Ready  
**Review Cycle:** Quarterly  
**Next Review:** January 2026
