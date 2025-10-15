"""
PostgreSQL + TimescaleDB + pgvector Example

Comprehensive demonstration of database persistence features:
- PostgreSQL connector setup
- Market data insertion and querying
- TimescaleDB time-bucket aggregates
- pgvector similarity search for ML features
- Performance benchmarking

Usage:
    uv run python scripts/example_postgres.py --native-tls
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qvp.data.postgres_connector import PostgreSQLConnector
from qvp.data.vector_store import VectorStore, PGVECTOR_AVAILABLE


def setup_database():
    """Initialize database connection and create tables."""
    logger.info("Setting up PostgreSQL database...")
    
    try:
        pg = PostgreSQLConnector()
        
        # Create all tables
        logger.info("Creating database tables...")
        pg.create_tables(drop_existing=False)
        
        logger.success("✓ Database tables created successfully")
        return pg
        
    except Exception as e:
        logger.error(f"Failed to setup database: {e}")
        logger.info("\nMake sure PostgreSQL is running:")
        logger.info("  Docker: docker-compose up postgres")
        logger.info("  Or set POSTGRES_HOST, POSTGRES_USER, etc. in .env")
        raise


def demo_market_data_insertion(pg: PostgreSQLConnector):
    """Demonstrate market data insertion."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 1: Market Data Insertion")
    logger.info("="*60)
    
    # Generate sample OHLCV data
    symbols = ['SPY', 'QQQ', 'IWM']
    days = 252  # One trading year
    
    logger.info(f"Generating {days} days of data for {len(symbols)} symbols...")
    
    data_rows = []
    base_time = datetime.now() - timedelta(days=days)
    
    for symbol in symbols:
        base_price = 100.0 if symbol == 'SPY' else 150.0 if symbol == 'QQQ' else 75.0
        
        for i in range(days):
            timestamp = base_time + timedelta(days=i)
            
            # Random walk with drift
            change = np.random.normal(0.0005, 0.015)
            base_price *= (1 + change)
            
            open_price = base_price * (1 + np.random.normal(0, 0.002))
            high_price = base_price * (1 + abs(np.random.normal(0.005, 0.003)))
            low_price = base_price * (1 - abs(np.random.normal(0.005, 0.003)))
            close_price = base_price
            volume = int(np.random.lognormal(15, 1))
            
            data_rows.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
    
    # Bulk insert
    logger.info(f"Inserting {len(data_rows)} market data records...")
    start_time = time.time()
    
    pg.insert_market_data(data_rows)
    
    elapsed = time.time() - start_time
    logger.success(f"✓ Inserted {len(data_rows)} records in {elapsed:.2f}s")
    logger.info(f"  Throughput: {len(data_rows)/elapsed:.0f} records/sec")


def demo_querying(pg: PostgreSQLConnector):
    """Demonstrate querying capabilities."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 2: Querying Market Data")
    logger.info("="*60)
    
    # Query recent data for SPY
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    logger.info(f"Querying SPY data from {start_date.date()} to {end_date.date()}...")
    start_time = time.time()
    
    df = pg.get_market_data('SPY', start_date, end_date)
    
    elapsed = time.time() - start_time
    logger.success(f"✓ Retrieved {len(df)} records in {elapsed*1000:.2f}ms")
    
    if len(df) > 0:
        logger.info(f"\nSample data (first 5 rows):")
        print(df.head().to_string())
        
        logger.info(f"\nPrice statistics:")
        logger.info(f"  High: ${df['high'].max():.2f}")
        logger.info(f"  Low: ${df['low'].min():.2f}")
        logger.info(f"  Avg Volume: {df['volume'].mean():.0f}")


def demo_timescale_aggregates(pg: PostgreSQLConnector):
    """Demonstrate TimescaleDB time-bucket aggregates."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 3: TimescaleDB Time-Bucket Aggregates")
    logger.info("="*60)
    
    logger.info("Computing weekly OHLCV aggregates for SPY...")
    start_time = time.time()
    
    df = pg.get_time_bucket_aggregates(
        symbol='SPY',
        interval='1 week',
        start_time=datetime.now() - timedelta(days=180)
    )
    
    elapsed = time.time() - start_time
    logger.success(f"✓ Computed {len(df)} weekly buckets in {elapsed*1000:.2f}ms")
    
    if len(df) > 0:
        logger.info(f"\nWeekly aggregates (first 5 weeks):")
        print(df.head().to_string())
        
        logger.info(f"\nTimescaleDB benefits:")
        logger.info(f"  - Automatic time-based partitioning")
        logger.info(f"  - 10-100x faster time-range queries")
        logger.info(f"  - Built-in time-bucket aggregation")
        logger.info(f"  - Efficient data compression")


def demo_trade_logging(pg: PostgreSQLConnector):
    """Demonstrate trade and portfolio logging."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 4: Trade & Portfolio Logging")
    logger.info("="*60)
    
    # Log some sample trades
    trades = [
        {
            'timestamp': datetime.now() - timedelta(days=10),
            'symbol': 'SPY',
            'action': 'buy',
            'quantity': 100,
            'price': 450.50,
            'commission': 1.00,
            'strategy': 'mean_reversion'
        },
        {
            'timestamp': datetime.now() - timedelta(days=5),
            'symbol': 'SPY',
            'action': 'sell',
            'quantity': 50,
            'price': 455.75,
            'commission': 1.00,
            'strategy': 'mean_reversion'
        },
        {
            'timestamp': datetime.now() - timedelta(days=2),
            'symbol': 'QQQ',
            'action': 'buy',
            'quantity': 75,
            'price': 375.25,
            'commission': 1.00,
            'strategy': 'momentum'
        }
    ]
    
    logger.info(f"Logging {len(trades)} trades...")
    for trade in trades:
        pg.insert_trade(**trade)
    
    logger.success(f"✓ Logged {len(trades)} trades")
    
    # Log portfolio values
    logger.info("Logging portfolio snapshots...")
    for i in range(30):
        timestamp = datetime.now() - timedelta(days=30-i)
        total_value = 100000 * (1 + np.random.normal(0.001, 0.01)) ** i
        cash = total_value * 0.3
        
        pg.insert_portfolio_value(
            timestamp=timestamp,
            total_value=total_value,
            cash=cash,
            equity=total_value - cash
        )
    
    logger.success("✓ Logged 30 portfolio snapshots")
    
    # Retrieve portfolio history
    logger.info("\nRetrieving portfolio history...")
    history = pg.get_portfolio_history(days=30)
    
    if len(history) > 0:
        logger.info(f"\nPortfolio Performance (last 30 days):")
        logger.info(f"  Starting Value: ${history.iloc[0]['total_value']:.2f}")
        logger.info(f"  Ending Value: ${history.iloc[-1]['total_value']:.2f}")
        logger.info(f"  Return: {((history.iloc[-1]['total_value'] / history.iloc[0]['total_value']) - 1) * 100:.2f}%")


def demo_vector_search(pg: PostgreSQLConnector):
    """Demonstrate pgvector similarity search."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 5: pgvector ML Feature Similarity Search")
    logger.info("="*60)
    
    if not PGVECTOR_AVAILABLE:
        logger.warning("⚠ pgvector not available, skipping demo")
        return
    
    try:
        vs = VectorStore(pg)
        vs.create_tables()
        
        # Generate sample feature embeddings
        logger.info("Generating sample ML feature embeddings...")
        
        embeddings = []
        for i in range(50):
            # Create synthetic feature vectors
            base_features = np.random.randn(128)
            
            embeddings.append({
                'symbol': np.random.choice(['SPY', 'QQQ', 'IWM']),
                'embedding': base_features,
                'feature_type': np.random.choice(['technical', 'volatility', 'ml']),
                'timestamp': datetime.now() - timedelta(days=i),
                'metadata': {'source': 'example', 'version': '1.0'}
            })
        
        # Batch insert
        logger.info(f"Inserting {len(embeddings)} feature embeddings...")
        start_time = time.time()
        
        vs.batch_insert_embeddings(embeddings)
        
        elapsed = time.time() - start_time
        logger.success(f"✓ Inserted {len(embeddings)} embeddings in {elapsed:.2f}s")
        
        # Perform similarity search
        logger.info("\nPerforming similarity search...")
        query_vector = np.random.randn(128)
        
        start_time = time.time()
        similar = vs.find_similar_features(
            query_vector=query_vector,
            top_k=5,
            metric='cosine'
        )
        elapsed = time.time() - start_time
        
        logger.success(f"✓ Found {len(similar)} similar features in {elapsed*1000:.2f}ms")
        
        if len(similar) > 0:
            logger.info(f"\nTop 5 similar features:")
            print(similar[['symbol', 'feature_type', 'distance', 'timestamp']].to_string())
        
        # Demo volatility regime clustering
        logger.info("\n" + "-"*60)
        logger.info("Volatility Regime Clustering")
        logger.info("-"*60)
        
        # Insert volatility regimes
        regimes = [
            ('low', {'realized_vol': 0.10, 'vix_level': 12, 'skew': -1.5, 'kurtosis': 3.2}),
            ('medium', {'realized_vol': 0.20, 'vix_level': 18, 'skew': -2.0, 'kurtosis': 4.5}),
            ('high', {'realized_vol': 0.35, 'vix_level': 28, 'skew': -3.5, 'kurtosis': 6.0}),
            ('crisis', {'realized_vol': 0.60, 'vix_level': 45, 'skew': -5.0, 'kurtosis': 10.0}),
        ]
        
        logger.info("Inserting historical volatility regimes...")
        for label, features in regimes:
            vs.insert_volatility_regime(label, features)
        
        logger.success(f"✓ Inserted {len(regimes)} volatility regimes")
        
        # Find similar regime
        current_market = {
            'realized_vol': 0.25,
            'vix_level': 22,
            'skew': -2.5,
            'kurtosis': 5.0
        }
        
        logger.info(f"\nCurrent market conditions: {current_market}")
        logger.info("Finding similar historical regimes...")
        
        similar_regimes = vs.find_similar_regimes(current_market, top_k=3)
        
        if len(similar_regimes) > 0:
            logger.info(f"\nMost similar regimes:")
            print(similar_regimes[['regime_label', 'distance', 'realized_vol', 'vix_level']].to_string())
            
            logger.info(f"\n💡 Insight: Current market most similar to '{similar_regimes.iloc[0]['regime_label']}' regime")
        
        # Create vector index for performance
        logger.info("\nCreating vector index for faster searches...")
        vs.create_vector_index('feature_embeddings', 'embedding', 'ivfflat')
        logger.success("✓ Created IVFFlat index on embeddings")
        
    except Exception as e:
        logger.error(f"Vector search demo failed: {e}")
        logger.info("Make sure pgvector extension is enabled in PostgreSQL")


def demo_performance_comparison():
    """Compare performance with other storage backends."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 6: Performance Comparison")
    logger.info("="*60)
    
    logger.info("\nStorage Backend Comparison:")
    logger.info("-" * 60)
    
    comparison = pd.DataFrame({
        'Backend': ['PostgreSQL', 'TimescaleDB', 'KDB+', 'Parquet'],
        'Insert Speed': ['~10K/s', '~15K/s', '~100K/s', '~50K/s'],
        'Query Speed': ['Fast', 'Very Fast', 'Ultra Fast', 'Medium'],
        'Time-Series': ['Good', 'Excellent', 'Excellent', 'Medium'],
        'Indexing': ['B-tree', 'Hypertable', 'Column', 'Column'],
        'ACID': ['Yes', 'Yes', 'No', 'No'],
        'Best For': ['General', 'Time-series', 'Tick data', 'Analytics']
    })
    
    print(comparison.to_string(index=False))
    
    logger.info("\n💡 Recommendations:")
    logger.info("  - PostgreSQL: General purpose, ACID transactions, relations")
    logger.info("  - TimescaleDB: Time-series data, automatic partitioning, compression")
    logger.info("  - pgvector: ML features, similarity search, embeddings")
    logger.info("  - KDB+: Ultra-fast tick data, high-frequency queries")
    logger.info("  - Parquet: Large-scale analytics, columnar storage")
    
    logger.info("\n🏗️ Architecture:")
    logger.info("  Use multiple backends for different use cases:")
    logger.info("  - PostgreSQL + TimescaleDB: OHLCV, trades, portfolio")
    logger.info("  - pgvector: Strategy patterns, regime clustering")
    logger.info("  - KDB+: Real-time tick data, microsecond queries")
    logger.info("  - Parquet: Historical research, backtesting")


def main():
    """Run all PostgreSQL demos."""
    logger.info("╔" + "="*58 + "╗")
    logger.info("║" + " "*10 + "PostgreSQL + TimescaleDB + pgvector Demo" + " "*8 + "║")
    logger.info("╚" + "="*58 + "╝")
    
    try:
        # Setup
        pg = setup_database()
        
        # Run demos
        demo_market_data_insertion(pg)
        demo_querying(pg)
        demo_timescale_aggregates(pg)
        demo_trade_logging(pg)
        demo_vector_search(pg)
        demo_performance_comparison()
        
        # Cleanup
        pg.close()
        
        logger.info("\n" + "="*60)
        logger.success("✓ All demos completed successfully!")
        logger.info("="*60)
        
        logger.info("\n📚 Next Steps:")
        logger.info("  1. Review DATABASE_INTEGRATION.md for detailed docs")
        logger.info("  2. Run migrations: alembic upgrade head")
        logger.info("  3. Start Docker stack: docker-compose up")
        logger.info("  4. Integrate with your trading strategies")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Check PostgreSQL is running")
        logger.info("  2. Verify .env configuration")
        logger.info("  3. Install extensions: CREATE EXTENSION timescaledb; CREATE EXTENSION vector;")
        sys.exit(1)


if __name__ == "__main__":
    main()
