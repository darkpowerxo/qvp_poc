"""
KDB+ Integration for High-Performance Time-Series Data Storage and Analysis

This module provides a comprehensive interface to kdb+/q for handling
high-frequency market data, tick-by-tick analysis, and ultra-fast volatility calculations.

Key Features:
- PyKX integration (embedded or IPC mode)
- Tick data storage and retrieval
- Q-based volatility calculations
- Time-series aggregations (OHLCV, VWAP, etc.)
- Partitioned database management
- Async query support
"""

import os
from typing import Optional, Dict, List, Any, Union
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

try:
    import pykx as kx
    PYKX_AVAILABLE = True
except ImportError:
    PYKX_AVAILABLE = False
    logger.warning("PyKX not installed. KDB+ functionality will be limited.")


class KDBConnector:
    """
    High-performance connector for kdb+/q time-series database.
    
    Supports both embedded mode (PyKX runs q in-process) and IPC mode
    (connect to external q process).
    
    Examples
    --------
    >>> # Embedded mode (no q process needed)
    >>> kdb = KDBConnector(mode='embedded')
    >>> kdb.create_tick_table('trades')
    >>> kdb.insert_ticks('trades', tick_data)
    
    >>> # IPC mode (connect to q process on port 5000)
    >>> kdb = KDBConnector(mode='ipc', host='localhost', port=5000)
    >>> trades = kdb.query("select from trades where date=.z.d")
    """
    
    def __init__(
        self,
        mode: str = 'embedded',
        host: str = 'localhost',
        port: int = 5000,
        username: str = '',
        password: str = '',
        timeout: int = 10000,
        use_tls: bool = False
    ):
        """
        Initialize KDB+ connection.
        
        Parameters
        ----------
        mode : str, default='embedded'
            Connection mode: 'embedded' or 'ipc'
        host : str, default='localhost'
            KDB+ server hostname (IPC mode only)
        port : int, default=5000
            KDB+ server port (IPC mode only)
        username : str, default=''
            Authentication username
        password : str, default=''
            Authentication password
        timeout : int, default=10000
            Query timeout in milliseconds
        use_tls : bool, default=False
            Use TLS encryption for connection
        """
        if not PYKX_AVAILABLE:
            raise ImportError(
                "PyKX is required for KDB+ integration. "
                "Install with: pip install pykx"
            )
        
        self.mode = mode
        self.host = host
        self.port = port
        self.timeout = timeout
        self.connection = None
        
        logger.info(f"Initializing KDB+ connector in {mode} mode")
        
        if mode == 'embedded':
            # Use PyKX embedded q
            self.q = kx.q
            logger.info("Using PyKX embedded q engine")
            
        elif mode == 'ipc':
            # Connect to external q process
            try:
                self.connection = kx.QConnection(
                    host=host,
                    port=port,
                    username=username,
                    password=password,
                    timeout=timeout,
                    tls=use_tls
                )
                self.q = self.connection
                logger.info(f"Connected to kdb+ at {host}:{port}")
                
            except Exception as e:
                logger.error(f"Failed to connect to kdb+: {e}")
                raise
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'embedded' or 'ipc'")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def close(self):
        """Close connection to kdb+."""
        if self.connection is not None:
            self.connection.close()
            logger.info("Closed kdb+ connection")
    
    def query(self, q_expression: str, *args) -> Any:
        """
        Execute a q expression and return result.
        
        Parameters
        ----------
        q_expression : str
            Q language expression to execute
        *args : optional
            Arguments to pass to the q expression
        
        Returns
        -------
        result : Any
            Query result (automatically converted to Python types)
        
        Examples
        --------
        >>> result = kdb.query("til 10")  # Returns [0,1,2,...,9]
        >>> trades = kdb.query("select from trades where sym=`AAPL")
        """
        try:
            if args:
                result = self.q(q_expression, *args)
            else:
                result = self.q(q_expression)
            
            # Convert to pandas/numpy if appropriate
            if hasattr(result, 'pd'):
                return result.pd()
            elif hasattr(result, 'np'):
                return result.np()
            else:
                return result
                
        except Exception as e:
            logger.error(f"Query failed: {e}\nQuery: {q_expression}")
            raise
    
    def create_tick_table(self, table_name: str = 'ticks') -> None:
        """
        Create a tick data table with standard schema.
        
        Schema includes: time, sym, price, size, exchange, conditions
        
        Parameters
        ----------
        table_name : str, default='ticks'
            Name of the table to create
        """
        q_code = f"""
        {table_name}:([] 
            time:`timestamp$(); 
            sym:`symbol$(); 
            price:`float$(); 
            size:`long$();
            exchange:`symbol$();
            conditions:`symbol$()
        )
        """
        self.query(q_code)
        logger.info(f"Created tick table: {table_name}")
    
    def create_ohlcv_table(self, table_name: str = 'ohlcv') -> None:
        """
        Create OHLCV bar data table.
        
        Schema: time, sym, open, high, low, close, volume
        
        Parameters
        ----------
        table_name : str, default='ohlcv'
            Name of the table to create
        """
        q_code = f"""
        {table_name}:([]
            time:`timestamp$();
            sym:`symbol$();
            open:`float$();
            high:`float$();
            low:`float$();
            close:`float$();
            volume:`long$()
        )
        """
        self.query(q_code)
        logger.info(f"Created OHLCV table: {table_name}")
    
    def insert_ticks(
        self, 
        table_name: str,
        data: pd.DataFrame
    ) -> int:
        """
        Insert tick data into kdb+ table.
        
        Parameters
        ----------
        table_name : str
            Target table name
        data : pd.DataFrame
            DataFrame with columns: time, sym, price, size, exchange, conditions
        
        Returns
        -------
        n_inserted : int
            Number of rows inserted
        
        Examples
        --------
        >>> ticks = pd.DataFrame({
        ...     'time': pd.date_range('2025-01-01', periods=100, freq='1s'),
        ...     'sym': ['AAPL'] * 100,
        ...     'price': np.random.randn(100).cumsum() + 150,
        ...     'size': np.random.randint(100, 1000, 100),
        ...     'exchange': ['NASDAQ'] * 100,
        ...     'conditions': [''] * 100
        ... })
        >>> kdb.insert_ticks('ticks', ticks)
        """
        # Convert DataFrame to kdb+ table
        kdb_table = kx.toq(data)
        
        # Insert into table
        self.q[table_name] = self.q('{x,y}', self.q[table_name], kdb_table)
        
        n_inserted = len(data)
        logger.debug(f"Inserted {n_inserted} ticks into {table_name}")
        
        return n_inserted
    
    def insert_ohlcv(
        self,
        table_name: str,
        data: pd.DataFrame
    ) -> int:
        """
        Insert OHLCV bar data into kdb+ table.
        
        Parameters
        ----------
        table_name : str
            Target table name
        data : pd.DataFrame
            DataFrame with OHLCV columns
        
        Returns
        -------
        n_inserted : int
            Number of rows inserted
        """
        kdb_table = kx.toq(data)
        self.q[table_name] = self.q('{x,y}', self.q[table_name], kdb_table)
        
        n_inserted = len(data)
        logger.debug(f"Inserted {n_inserted} bars into {table_name}")
        
        return n_inserted
    
    def get_ticks(
        self,
        table_name: str = 'ticks',
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve tick data with optional filters.
        
        Parameters
        ----------
        table_name : str, default='ticks'
            Source table name
        symbol : str, optional
            Filter by symbol
        start_time : datetime, optional
            Filter by start time
        end_time : datetime, optional
            Filter by end time
        limit : int, optional
            Maximum number of rows to return
        
        Returns
        -------
        ticks : pd.DataFrame
            Tick data
        """
        # Build query
        where_clauses = []
        
        if symbol:
            where_clauses.append(f"sym=`{symbol}")
        
        if start_time:
            start_str = start_time.strftime('%Y.%m.%dD%H:%M:%S.%f')[:-3]
            where_clauses.append(f"time>={start_str}")
        
        if end_time:
            end_str = end_time.strftime('%Y.%m.%dD%H:%M:%S.%f')[:-3]
            where_clauses.append(f"time<={end_str}")
        
        where_clause = ','.join(where_clauses) if where_clauses else ''
        
        if where_clause:
            query = f"select from {table_name} where {where_clause}"
        else:
            query = f"select from {table_name}"
        
        if limit:
            query = f"{limit} sublist {query}"
        
        return self.query(query)
    
    def calculate_ohlcv_from_ticks(
        self,
        table_name: str = 'ticks',
        symbol: str = 'AAPL',
        interval: str = '1m',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Aggregate tick data into OHLCV bars using q.
        
        Parameters
        ----------
        table_name : str, default='ticks'
            Source tick table
        symbol : str, default='AAPL'
            Symbol to aggregate
        interval : str, default='1m'
            Bar interval (e.g., '1m', '5m', '1h', '1D')
        start_time : datetime, optional
            Start time filter
        end_time : datetime, optional
            End time filter
        
        Returns
        -------
        ohlcv : pd.DataFrame
            OHLCV bars
        """
        # Parse interval to kdb+ timespan
        interval_map = {
            '1s': '0D00:00:01',
            '1m': '0D00:01:00',
            '5m': '0D00:05:00',
            '15m': '0D00:15:00',
            '1h': '0D01:00:00',
            '1D': '1D00:00:00'
        }
        
        kdb_interval = interval_map.get(interval, '0D00:01:00')
        
        # Build query
        where_clause = f"sym=`{symbol}"
        if start_time:
            start_str = start_time.strftime('%Y.%m.%dD%H:%M:%S.%f')[:-3]
            where_clause += f",time>={start_str}"
        if end_time:
            end_str = end_time.strftime('%Y.%m.%dD%H:%M:%S.%f')[:-3]
            where_clause += f",time<={end_str}"
        
        q_code = f"""
        select 
            open:first price,
            high:max price,
            low:min price,
            close:last price,
            volume:sum size
        by {kdb_interval} xbar time
        from {table_name}
        where {where_clause}
        """
        
        return self.query(q_code)
    
    def calculate_vwap(
        self,
        table_name: str = 'ticks',
        symbol: str = 'AAPL',
        interval: str = '1m'
    ) -> pd.DataFrame:
        """
        Calculate Volume-Weighted Average Price (VWAP).
        
        Parameters
        ----------
        table_name : str
            Source tick table
        symbol : str
            Symbol
        interval : str
            Aggregation interval
        
        Returns
        -------
        vwap : pd.DataFrame
            VWAP by interval
        """
        interval_map = {
            '1m': '0D00:01:00',
            '5m': '0D00:05:00',
            '15m': '0D00:15:00',
            '1h': '0D01:00:00'
        }
        
        kdb_interval = interval_map.get(interval, '0D00:01:00')
        
        q_code = f"""
        select vwap:size wavg price by {kdb_interval} xbar time 
        from {table_name} 
        where sym=`{symbol}
        """
        
        return self.query(q_code)
    
    def calculate_realized_volatility_q(
        self,
        table_name: str = 'ticks',
        symbol: str = 'AAPL',
        window: int = 20,
        frequency: str = '1D'
    ) -> pd.DataFrame:
        """
        Calculate realized volatility using q.
        
        Uses high-frequency tick data to compute realized variance
        as sum of squared returns.
        
        Parameters
        ----------
        table_name : str
            Source tick table
        symbol : str
            Symbol
        window : int
            Rolling window for annualization
        frequency : str
            Sampling frequency ('1m', '5m', '1D')
        
        Returns
        -------
        realized_vol : pd.DataFrame
            Realized volatility time series
        """
        freq_map = {
            '1m': '0D00:01:00',
            '5m': '0D00:05:00',
            '1h': '0D01:00:00',
            '1D': '1D00:00:00'
        }
        
        kdb_freq = freq_map.get(frequency, '1D00:00:00')
        
        q_code = f"""
        / Get tick data and compute log returns
        ticks:select time, sym, price from {table_name} where sym=`{symbol};
        
        / Compute returns at sampling frequency
        bars:select last price by {kdb_freq} xbar time from ticks;
        bars:update logret:log price % prev price from bars;
        
        / Realized variance = sum of squared returns
        bars:update rv:{window} msum logret*logret from bars;
        
        / Annualize (252 trading days)
        bars:update realized_vol:sqrt[252*rv] from bars;
        
        select time, realized_vol from bars where not null realized_vol
        """
        
        return self.query(q_code)
    
    def create_partitioned_db(
        self,
        db_path: str,
        table_name: str = 'ticks',
        partition_type: str = 'date'
    ) -> None:
        """
        Create a partitioned on-disk database for large-scale data.
        
        Parameters
        ----------
        db_path : str
            Path to database directory
        table_name : str
            Table name
        partition_type : str, default='date'
            Partition scheme: 'date', 'month', 'year'
        """
        # Ensure path exists
        Path(db_path).mkdir(parents=True, exist_ok=True)
        
        # Set database
        self.query(f"`:/{db_path} set `.")
        
        logger.info(f"Created partitioned database at {db_path}")
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get metadata about a table.
        
        Parameters
        ----------
        table_name : str
            Table name
        
        Returns
        -------
        info : dict
            Table metadata (columns, count, memory usage)
        """
        meta = self.query(f"meta {table_name}")
        count = self.query(f"count {table_name}")
        
        info = {
            'columns': meta.index.tolist() if hasattr(meta, 'index') else [],
            'types': meta['t'].tolist() if 't' in meta.columns else [],
            'count': int(count),
            'memory_mb': 0  # TODO: calculate actual memory usage
        }
        
        return info
    
    def benchmark_query(self, q_expression: str, n_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark query performance.
        
        Parameters
        ----------
        q_expression : str
            Q expression to benchmark
        n_runs : int, default=10
            Number of runs
        
        Returns
        -------
        stats : dict
            Timing statistics (mean, min, max, std in ms)
        """
        times = []
        
        for _ in range(n_runs):
            start = datetime.now()
            self.query(q_expression)
            elapsed = (datetime.now() - start).total_seconds() * 1000
            times.append(elapsed)
        
        stats = {
            'mean_ms': np.mean(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'std_ms': np.std(times),
            'n_runs': n_runs
        }
        
        logger.info(
            f"Benchmark: {stats['mean_ms']:.2f}ms ± {stats['std_ms']:.2f}ms "
            f"({n_runs} runs)"
        )
        
        return stats


def load_kdb_config() -> Dict[str, Any]:
    """
    Load kdb+ configuration from environment variables.
    
    Returns
    -------
    config : dict
        KDB+ connection configuration
    """
    from qvp.config import get_config
    
    config = {
        'mode': os.getenv('KDB_MODE', 'embedded'),
        'host': os.getenv('KDB_HOST', 'localhost'),
        'port': int(os.getenv('KDB_PORT', 5000)),
        'username': os.getenv('KDB_USER', ''),
        'password': os.getenv('KDB_PASSWORD', ''),
        'timeout': int(os.getenv('KDB_TIMEOUT', 10000)),
        'use_tls': os.getenv('KDB_USE_TLS', 'false').lower() == 'true'
    }
    
    return config
