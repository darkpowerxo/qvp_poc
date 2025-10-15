"""
KDB+ Integration Example for Quantitative Volatility Platform

This script demonstrates:
1. Connecting to kdb+ (embedded mode)
2. Loading high-frequency tick data
3. Q-based volatility calculations
4. Performance comparison: q vs Python
5. Time-series aggregations
6. Advanced analytics

Requirements:
- PyKX installed: pip install pykx
- Set KDB_MODE=embedded in .env file
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from loguru import logger

# QVP imports
from qvp.data.kdb_connector import KDBConnector, load_kdb_config
from qvp.data.ingestion import DataIngester
from qvp.research import VolatilityEstimator

logger.add("logs/kdb_example.log", rotation="10 MB")


def generate_sample_tick_data(
    symbol: str = 'AAPL',
    n_ticks: int = 100000,
    start_price: float = 150.0
) -> pd.DataFrame:
    """
    Generate realistic simulated tick data for testing.
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    n_ticks : int
        Number of ticks to generate
    start_price : float
        Initial price
    
    Returns
    -------
    ticks : pd.DataFrame
        Simulated tick data
    """
    logger.info(f"Generating {n_ticks:,} sample ticks for {symbol}")
    
    # Generate timestamps (1 second intervals with some randomness)
    start_time = datetime(2025, 1, 15, 9, 30, 0)
    timestamps = [
        start_time + timedelta(seconds=i + np.random.uniform(-0.5, 0.5))
        for i in range(n_ticks)
    ]
    timestamps.sort()
    
    # Generate prices using geometric Brownian motion
    dt = 1.0 / (252 * 6.5 * 3600)  # 1 second in trading time
    drift = 0.05 * dt
    vol = 0.20 * np.sqrt(dt)
    
    returns = np.random.normal(drift, vol, n_ticks)
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Generate sizes (log-normal distribution)
    sizes = np.random.lognormal(mean=5, sigma=1, size=n_ticks).astype(int)
    sizes = np.clip(sizes, 1, 10000)
    
    # Create DataFrame
    ticks = pd.DataFrame({
        'time': timestamps,
        'sym': symbol,
        'price': prices,
        'size': sizes,
        'exchange': 'NASDAQ',
        'conditions': ''
    })
    
    logger.info(f"Generated ticks: price range ${prices.min():.2f}-${prices.max():.2f}")
    
    return ticks


def example_basic_connection():
    """Example 1: Basic KDB+ connection and queries."""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 1: Basic KDB+ Connection")
    logger.info("="*80)
    
    # Load configuration
    config = load_kdb_config()
    
    # Connect to kdb+
    with KDBConnector(**config) as kdb:
        logger.info(f"Connected to kdb+ in {config['mode']} mode")
        
        # Simple queries
        result = kdb.query("til 10")
        logger.info(f"Query 'til 10': {result}")
        
        # Create a simple table
        kdb.query("trades:([] sym:`AAPL`MSFT`GOOGL; price:150 280 2800f)")
        trades = kdb.query("select from trades")
        logger.info(f"\nSample trades table:\n{trades}")
        
        # Calculate average
        avg_price = kdb.query("select avg price from trades")
        logger.info(f"\nAverage price: ${avg_price.iloc[0, 0]:.2f}")


def example_tick_data_ingestion():
    """Example 2: Load tick data into kdb+."""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 2: Tick Data Ingestion")
    logger.info("="*80)
    
    # Generate sample data
    ticks = generate_sample_tick_data(symbol='AAPL', n_ticks=50000)
    
    config = load_kdb_config()
    
    with KDBConnector(**config) as kdb:
        # Create tick table
        kdb.create_tick_table('ticks')
        
        # Insert data
        logger.info("Inserting ticks into kdb+...")
        start = time.time()
        n_inserted = kdb.insert_ticks('ticks', ticks)
        elapsed = time.time() - start
        
        logger.info(f"Inserted {n_inserted:,} ticks in {elapsed:.3f}s "
                   f"({n_inserted/elapsed:,.0f} ticks/sec)")
        
        # Get table info
        info = kdb.get_table_info('ticks')
        logger.info(f"\nTable info: {info}")
        
        # Query sample
        sample = kdb.query("10 sublist select from ticks")
        logger.info(f"\nFirst 10 ticks:\n{sample}")


def example_ohlcv_aggregation():
    """Example 3: Aggregate ticks to OHLCV bars."""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 3: OHLCV Aggregation")
    logger.info("="*80)
    
    config = load_kdb_config()
    
    with KDBConnector(**config) as kdb:
        # Assuming ticks already loaded from previous example
        logger.info("Aggregating ticks to 1-minute OHLCV bars...")
        
        start = time.time()
        ohlcv = kdb.calculate_ohlcv_from_ticks(
            table_name='ticks',
            symbol='AAPL',
            interval='1m'
        )
        elapsed = time.time() - start
        
        logger.info(f"Aggregated to {len(ohlcv)} bars in {elapsed:.3f}s")
        logger.info(f"\nSample OHLCV data:\n{ohlcv.head(10)}")
        
        # Calculate VWAP
        logger.info("\nCalculating VWAP...")
        vwap = kdb.calculate_vwap('ticks', 'AAPL', '5m')
        logger.info(f"5-minute VWAP:\n{vwap.head()}")


def example_volatility_calculations():
    """Example 4: Volatility calculations using q."""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 4: Q-based Volatility Calculations")
    logger.info("="*80)
    
    config = load_kdb_config()
    
    with KDBConnector(**config) as kdb:
        # Load volatility calculation functions
        q_script_path = Path(__file__).parent.parent / 'qvp' / 'data' / 'q_scripts' / 'volatility.q'
        
        if q_script_path.exists():
            logger.info(f"Loading q volatility functions from {q_script_path}")
            kdb.query(f"\\l {q_script_path}")
        
        # Generate OHLC data
        logger.info("Generating sample OHLC data...")
        n = 252
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        
        # Simulate prices
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, n)
        prices = 100 * np.exp(np.cumsum(returns))
        
        ohlc = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            'close': prices * (1 + np.random.normal(0, 0.01, n))
        })
        
        # Send to kdb+
        kdb_ohlc = kdb.query("", ohlc)  # Convert to kdb+ table
        kdb.q['ohlc'] = kdb_ohlc
        
        # Calculate Parkinson volatility
        logger.info("\nCalculating Parkinson volatility in q...")
        park_vol = kdb.query("parkinsonVol[ohlc`high;ohlc`low]")
        logger.info(f"Parkinson volatility: {park_vol:.4f}")
        
        # Calculate Garman-Klass volatility
        logger.info("Calculating Garman-Klass volatility in q...")
        gk_vol = kdb.query("garmanKlassVol[ohlc`open;ohlc`high;ohlc`low;ohlc`close]")
        logger.info(f"Garman-Klass volatility: {gk_vol:.4f}")
        
        # Calculate Yang-Zhang volatility
        logger.info("Calculating Yang-Zhang volatility in q...")
        yz_vol = kdb.query("yangZhangVol[ohlc`open;ohlc`high;ohlc`low;ohlc`close]")
        logger.info(f"Yang-Zhang volatility: {yz_vol:.4f}")
        
        # Annualize
        logger.info(f"\nAnnualized volatilities:")
        logger.info(f"  Parkinson:    {park_vol * np.sqrt(252):.2%}")
        logger.info(f"  Garman-Klass: {gk_vol * np.sqrt(252):.2%}")
        logger.info(f"  Yang-Zhang:   {yz_vol * np.sqrt(252):.2%}")


def example_performance_comparison():
    """Example 5: Performance comparison q vs Python."""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 5: Performance Comparison (q vs Python)")
    logger.info("="*80)
    
    # Generate large dataset
    n = 10000
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, n)
    prices = 100 * np.exp(np.cumsum(returns))
    
    ohlc_data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices * 1.005
    })
    
    config = load_kdb_config()
    
    with KDBConnector(**config) as kdb:
        # Load q functions
        q_script_path = Path(__file__).parent.parent / 'qvp' / 'data' / 'q_scripts' / 'volatility.q'
        if q_script_path.exists():
            kdb.query(f"\\l {q_script_path}")
        
        # Send data to kdb+
        kdb.q['testdata'] = kdb.query("", ohlc_data)
        
        # Test 1: Parkinson volatility
        logger.info(f"\nTest 1: Parkinson Volatility ({n:,} bars)")
        
        # Q implementation
        start = time.time()
        kdb.query("parkinsonVol[testdata`high;testdata`low]")
        q_time = time.time() - start
        logger.info(f"  q implementation:      {q_time*1000:.2f}ms")
        
        # Python implementation
        vol_est = VolatilityEstimator()
        start = time.time()
        vol_est.parkinson(ohlc_data['high'], ohlc_data['low'], window=20)
        py_time = time.time() - start
        logger.info(f"  Python implementation: {py_time*1000:.2f}ms")
        logger.info(f"  Speedup: {py_time/q_time:.1f}x")
        
        # Test 2: Yang-Zhang volatility
        logger.info(f"\nTest 2: Yang-Zhang Volatility ({n:,} bars)")
        
        # Q implementation
        start = time.time()
        kdb.query("yangZhangVol[testdata`open;testdata`high;testdata`low;testdata`close]")
        q_time = time.time() - start
        logger.info(f"  q implementation:      {q_time*1000:.2f}ms")
        
        # Python implementation
        start = time.time()
        vol_est.yang_zhang(
            ohlc_data['open'],
            ohlc_data['high'],
            ohlc_data['low'],
            ohlc_data['close'],
            window=20
        )
        py_time = time.time() - start
        logger.info(f"  Python implementation: {py_time*1000:.2f}ms")
        logger.info(f"  Speedup: {py_time/q_time:.1f}x")


def example_realtime_analytics():
    """Example 6: Real-time analytics simulation."""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 6: Real-time Analytics")
    logger.info("="*80)
    
    config = load_kdb_config()
    
    with KDBConnector(**config) as kdb:
        # Create tables
        kdb.create_tick_table('streaming_ticks')
        
        # Simulate streaming data
        logger.info("Simulating streaming tick data (10 batches)...")
        
        for batch in range(10):
            # Generate batch of ticks
            ticks = generate_sample_tick_data(
                symbol='AAPL',
                n_ticks=1000,
                start_price=150 + batch * 0.1
            )
            
            # Insert into kdb+
            kdb.insert_ticks('streaming_ticks', ticks)
            
            # Calculate rolling metrics
            latest_vwap = kdb.query(
                "select vwap:size wavg price from streaming_ticks"
            )
            
            total_ticks = kdb.query("count streaming_ticks")
            
            logger.info(
                f"  Batch {batch+1}: {total_ticks} total ticks, "
                f"VWAP = ${latest_vwap.iloc[0, 0]:.2f}"
            )
            
            time.sleep(0.1)  # Simulate delay
        
        logger.info("\nStreaming simulation complete")


def main():
    """Run all examples."""
    logger.info("="*80)
    logger.info("KDB+ INTEGRATION EXAMPLES FOR QVP")
    logger.info("="*80)
    
    try:
        # Check if PyKX is available
        import pykx
        logger.info(f"PyKX version: {pykx.__version__}")
        
    except ImportError:
        logger.error(
            "PyKX not installed. Please install: pip install pykx\n"
            "Then run: uv sync"
        )
        return
    
    try:
        # Run examples
        example_basic_connection()
        example_tick_data_ingestion()
        example_ohlcv_aggregation()
        example_volatility_calculations()
        example_performance_comparison()
        example_realtime_analytics()
        
        logger.info("\n" + "="*80)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("\nNext Steps:")
        logger.info("  1. Explore q scripts in qvp/data/q_scripts/")
        logger.info("  2. Read documentation: docs/KDB_INTEGRATION.md")
        logger.info("  3. Try connecting to external q process (KDB_MODE=ipc)")
        logger.info("  4. Benchmark with your own data")
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
