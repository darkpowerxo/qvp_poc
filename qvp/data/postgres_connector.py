"""
PostgreSQL + TimescaleDB Integration for Time-Series Data Storage

This module provides a comprehensive database layer for:
- Market data persistence (OHLCV, ticks, quotes)
- Position and trade tracking
- Risk metrics storage
- Performance analytics
- TimescaleDB hypertables for efficient time-series queries
- Connection pooling and async support

Features:
- SQLAlchemy ORM models
- TimescaleDB hypertables with automatic partitioning
- Connection pooling for performance
- Async queries with asyncpg
- Automatic schema creation
- Data retention policies
"""

import os
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from contextlib import contextmanager

import pandas as pd
import numpy as np
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    Boolean, Text, Index, ForeignKey, BigInteger, Numeric, Date
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import text
from loguru import logger

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logger.warning("asyncpg not available. Async features disabled.")

Base = declarative_base()


# ============================================================================
# SQLAlchemy Models
# ============================================================================

class MarketData(Base):
    """OHLCV market data table (TimescaleDB hypertable)."""
    __tablename__ = 'market_data'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)
    adj_close = Column(Float)
    source = Column(String(50), default='yfinance')
    
    __table_args__ = (
        Index('idx_market_data_symbol_time', 'symbol', 'timestamp'),
    )


class TickData(Base):
    """High-frequency tick data table (TimescaleDB hypertable)."""
    __tablename__ = 'tick_data'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    price = Column(Float, nullable=False)
    size = Column(Integer, nullable=False)
    exchange = Column(String(20))
    conditions = Column(String(100))
    
    __table_args__ = (
        Index('idx_tick_data_symbol_time', 'symbol', 'timestamp'),
    )


class Position(Base):
    """Portfolio positions table."""
    __tablename__ = 'positions'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    market_value = Column(Float)
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float)
    strategy = Column(String(100))
    
    __table_args__ = (
        Index('idx_positions_symbol_time', 'symbol', 'timestamp'),
    )


class Trade(Base):
    """Trade executions table."""
    __tablename__ = 'trades'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    strategy = Column(String(100))
    order_type = Column(String(20))  # 'market', 'limit', etc.
    status = Column(String(20), default='filled')  # 'filled', 'partial', 'rejected'
    
    __table_args__ = (
        Index('idx_trades_symbol_time', 'symbol', 'timestamp'),
    )


class PortfolioValue(Base):
    """Portfolio value over time (TimescaleDB hypertable)."""
    __tablename__ = 'portfolio_value'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    total_value = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    returns = Column(Float)
    cumulative_returns = Column(Float)
    
    __table_args__ = (
        Index('idx_portfolio_value_time', 'timestamp'),
    )


class RiskMetrics(Base):
    """Risk metrics over time (TimescaleDB hypertable)."""
    __tablename__ = 'risk_metrics'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    var_95 = Column(Float)
    var_99 = Column(Float)
    cvar_95 = Column(Float)
    cvar_99 = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    volatility = Column(Float)
    beta = Column(Float)
    
    __table_args__ = (
        Index('idx_risk_metrics_time', 'timestamp'),
    )


class VolatilityEstimates(Base):
    """Volatility estimates table (TimescaleDB hypertable)."""
    __tablename__ = 'volatility_estimates'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    close_to_close = Column(Float)
    parkinson = Column(Float)
    garman_klass = Column(Float)
    rogers_satchell = Column(Float)
    yang_zhang = Column(Float)
    realized_vol = Column(Float)
    implied_vol = Column(Float)
    
    __table_args__ = (
        Index('idx_volatility_symbol_time', 'symbol', 'timestamp'),
    )


class SignalLog(Base):
    """Trading signals log."""
    __tablename__ = 'signal_log'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    signal_type = Column(String(20))  # 'buy', 'sell', 'hold'
    strength = Column(Float)
    strategy = Column(String(100))
    metadata = Column(Text)  # JSON string with additional info
    
    __table_args__ = (
        Index('idx_signals_symbol_time', 'symbol', 'timestamp'),
    )


# ============================================================================
# PostgreSQL Connector
# ============================================================================

class PostgreSQLConnector:
    """
    High-performance PostgreSQL connector with TimescaleDB support.
    
    Features:
    - Connection pooling
    - Automatic schema creation
    - TimescaleDB hypertable setup
    - Async query support
    - Batch inserts
    - Time-series optimized queries
    
    Examples
    --------
    >>> pg = PostgreSQLConnector()
    >>> pg.create_tables()
    >>> 
    >>> # Insert market data
    >>> data = pd.DataFrame({...})
    >>> pg.insert_market_data(data)
    >>> 
    >>> # Query time series
    >>> df = pg.get_market_data('SPY', start_date='2024-01-01')
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20
    ):
        """
        Initialize PostgreSQL connection.
        
        Parameters
        ----------
        host : str, optional
            Database host (default from env)
        port : int, optional
            Database port (default from env)
        database : str, optional
            Database name (default from env)
        user : str, optional
            Username (default from env)
        password : str, optional
            Password (default from env)
        pool_size : int, default=10
            Connection pool size
        max_overflow : int, default=20
            Max pool overflow
        """
        # Load configuration from environment
        self.host = host or os.getenv('POSTGRES_HOST', 'localhost')
        self.port = port or int(os.getenv('POSTGRES_PORT', 5432))
        self.database = database or os.getenv('POSTGRES_DB', 'qvp_db')
        self.user = user or os.getenv('POSTGRES_USER', 'qvp_user')
        self.password = password or os.getenv('POSTGRES_PASSWORD', 'changeme')
        
        # Build connection string
        self.connection_string = (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Verify connections
            echo=False
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )
        
        logger.info(
            f"PostgreSQL connector initialized: {self.host}:{self.port}/{self.database}"
        )
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self, drop_existing: bool = False):
        """
        Create all database tables.
        
        Parameters
        ----------
        drop_existing : bool, default=False
            Drop existing tables before creating
        """
        if drop_existing:
            Base.metadata.drop_all(self.engine)
            logger.warning("Dropped all existing tables")
        
        Base.metadata.create_all(self.engine)
        logger.info("Created all tables")
        
        # Setup TimescaleDB hypertables
        self._setup_timescale_hypertables()
    
    def _setup_timescale_hypertables(self):
        """Convert tables to TimescaleDB hypertables."""
        if not os.getenv('TIMESCALEDB_ENABLED', 'true').lower() == 'true':
            logger.info("TimescaleDB disabled, skipping hypertable setup")
            return
        
        hypertables = [
            ('market_data', 'timestamp'),
            ('tick_data', 'timestamp'),
            ('portfolio_value', 'timestamp'),
            ('risk_metrics', 'timestamp'),
            ('volatility_estimates', 'timestamp')
        ]
        
        with self.engine.connect() as conn:
            # Check if TimescaleDB extension exists
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
                conn.commit()
                logger.info("TimescaleDB extension enabled")
            except Exception as e:
                logger.warning(f"Could not enable TimescaleDB: {e}")
                return
            
            # Convert tables to hypertables
            for table_name, time_column in hypertables:
                try:
                    conn.execute(text(
                        f"SELECT create_hypertable('{table_name}', '{time_column}', "
                        f"if_not_exists => TRUE, migrate_data => TRUE)"
                    ))
                    conn.commit()
                    logger.info(f"Created hypertable: {table_name}")
                except Exception as e:
                    logger.warning(f"Could not create hypertable {table_name}: {e}")
    
    def insert_market_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        source: str = 'yfinance'
    ) -> int:
        """
        Insert OHLCV market data.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with OHLCV columns
        symbol : str
            Stock symbol
        source : str, default='yfinance'
            Data source
        
        Returns
        -------
        n_inserted : int
            Number of rows inserted
        """
        records = []
        for idx, row in data.iterrows():
            record = MarketData(
                timestamp=idx if isinstance(idx, datetime) else pd.to_datetime(idx),
                symbol=symbol,
                open=float(row.get('open', row.get('Open', 0))),
                high=float(row.get('high', row.get('High', 0))),
                low=float(row.get('low', row.get('Low', 0))),
                close=float(row.get('close', row.get('Close', 0))),
                volume=int(row.get('volume', row.get('Volume', 0))),
                adj_close=float(row.get('adj_close', row.get('Adj Close', row.get('close', 0)))),
                source=source
            )
            records.append(record)
        
        with self.get_session() as session:
            session.bulk_save_objects(records)
        
        logger.info(f"Inserted {len(records)} market data rows for {symbol}")
        return len(records)
    
    def get_market_data(
        self,
        symbol: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve market data for a symbol.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        start_date : str or datetime, optional
            Start date filter
        end_date : str or datetime, optional
            End date filter
        limit : int, optional
            Maximum rows to return
        
        Returns
        -------
        data : pd.DataFrame
            Market data with timestamp index
        """
        with self.get_session() as session:
            query = session.query(MarketData).filter(MarketData.symbol == symbol)
            
            if start_date:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                query = query.filter(MarketData.timestamp >= start_date)
            
            if end_date:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                query = query.filter(MarketData.timestamp <= end_date)
            
            query = query.order_by(MarketData.timestamp)
            
            if limit:
                query = query.limit(limit)
            
            results = query.all()
        
        if not results:
            return pd.DataFrame()
        
        data = pd.DataFrame([{
            'timestamp': r.timestamp,
            'open': r.open,
            'high': r.high,
            'low': r.low,
            'close': r.close,
            'volume': r.volume,
            'adj_close': r.adj_close
        } for r in results])
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def insert_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: Optional[datetime] = None,
        commission: float = 0.0,
        slippage: float = 0.0,
        strategy: Optional[str] = None
    ) -> int:
        """
        Insert a trade execution record.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        side : str
            'buy' or 'sell'
        quantity : float
            Trade quantity
        price : float
            Execution price
        timestamp : datetime, optional
            Trade timestamp (default: now)
        commission : float, default=0.0
            Commission paid
        slippage : float, default=0.0
            Slippage incurred
        strategy : str, optional
            Strategy name
        
        Returns
        -------
        trade_id : int
            Inserted trade ID
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage,
            strategy=strategy,
            status='filled'
        )
        
        with self.get_session() as session:
            session.add(trade)
            session.flush()
            trade_id = trade.id
        
        logger.debug(f"Inserted trade: {side} {quantity} {symbol} @ {price}")
        return trade_id
    
    def insert_portfolio_value(
        self,
        total_value: float,
        cash: float,
        positions_value: float,
        timestamp: Optional[datetime] = None,
        returns: Optional[float] = None
    ) -> int:
        """Insert portfolio value snapshot."""
        if timestamp is None:
            timestamp = datetime.now()
        
        pv = PortfolioValue(
            timestamp=timestamp,
            total_value=total_value,
            cash=cash,
            positions_value=positions_value,
            returns=returns
        )
        
        with self.get_session() as session:
            session.add(pv)
            session.flush()
            pv_id = pv.id
        
        return pv_id
    
    def get_portfolio_history(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """Get portfolio value history."""
        with self.get_session() as session:
            query = session.query(PortfolioValue).order_by(PortfolioValue.timestamp)
            
            if start_date:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                query = query.filter(PortfolioValue.timestamp >= start_date)
            
            if end_date:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                query = query.filter(PortfolioValue.timestamp <= end_date)
            
            results = query.all()
        
        if not results:
            return pd.DataFrame()
        
        data = pd.DataFrame([{
            'timestamp': r.timestamp,
            'total_value': r.total_value,
            'cash': r.cash,
            'positions_value': r.positions_value,
            'returns': r.returns,
            'cumulative_returns': r.cumulative_returns
        } for r in results])
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def execute_timescale_query(
        self,
        query: str,
        params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Execute a raw TimescaleDB query.
        
        Parameters
        ----------
        query : str
            SQL query string
        params : dict, optional
            Query parameters
        
        Returns
        -------
        result : pd.DataFrame
            Query result
        """
        with self.engine.connect() as conn:
            result = pd.read_sql_query(query, conn, params=params)
        
        return result
    
    def get_time_bucket_aggregates(
        self,
        table: str,
        bucket_size: str = '1 hour',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get time-bucketed aggregates using TimescaleDB.
        
        Parameters
        ----------
        table : str
            Table name
        bucket_size : str, default='1 hour'
            Time bucket size (e.g., '1 hour', '1 day')
        start_time : datetime, optional
            Start time filter
        end_time : datetime, optional
            End time filter
        
        Returns
        -------
        aggregates : pd.DataFrame
            Time-bucketed aggregates
        """
        query = f"""
        SELECT
            time_bucket('{bucket_size}', timestamp) AS bucket,
            symbol,
            first(open, timestamp) as open,
            max(high) as high,
            min(low) as low,
            last(close, timestamp) as close,
            sum(volume) as volume
        FROM {table}
        WHERE 1=1
        """
        
        params = {}
        if start_time:
            query += " AND timestamp >= :start_time"
            params['start_time'] = start_time
        
        if end_time:
            query += " AND timestamp <= :end_time"
            params['end_time'] = end_time
        
        query += " GROUP BY bucket, symbol ORDER BY bucket"
        
        return self.execute_timescale_query(query, params)
    
    def close(self):
        """Close database connections."""
        self.engine.dispose()
        logger.info("PostgreSQL connections closed")


def load_postgres_config() -> Dict[str, Any]:
    """
    Load PostgreSQL configuration from environment.
    
    Returns
    -------
    config : dict
        PostgreSQL connection configuration
    """
    config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'database': os.getenv('POSTGRES_DB', 'qvp_db'),
        'user': os.getenv('POSTGRES_USER', 'qvp_user'),
        'password': os.getenv('POSTGRES_PASSWORD', 'changeme'),
        'pool_size': int(os.getenv('POSTGRES_POOL_SIZE', 10)),
        'max_overflow': int(os.getenv('POSTGRES_MAX_OVERFLOW', 20))
    }
    
    return config
