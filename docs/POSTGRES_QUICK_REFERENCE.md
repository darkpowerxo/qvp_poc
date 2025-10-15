# PostgreSQL Quick Reference

Quick commands and snippets for working with the QVP PostgreSQL database.

## 🚀 Getting Started

### Start Database
```bash
# Docker Compose
docker-compose up -d postgres

# View logs
docker-compose logs -f postgres

# Stop
docker-compose down
```

### Connect to Database
```bash
# Using psql
docker exec -it qvp-postgres psql -U qvp_user -d qvp_db

# Using Python
python
>>> from qvp.data.postgres_connector import PostgreSQLConnector
>>> pg = PostgreSQLConnector()
```

## 📊 Common Queries

### Market Data
```sql
-- Recent SPY data
SELECT * FROM market_data 
WHERE symbol = 'SPY' 
  AND timestamp >= NOW() - INTERVAL '7 days'
ORDER BY timestamp DESC;

-- Daily OHLCV aggregates
SELECT 
    time_bucket('1 day', timestamp) AS day,
    symbol,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume
FROM market_data
WHERE symbol = 'SPY'
GROUP BY day, symbol
ORDER BY day DESC;
```

### Portfolio Tracking
```sql
-- Current portfolio value
SELECT * FROM portfolio_values 
ORDER BY timestamp DESC 
LIMIT 1;

-- 30-day performance
SELECT 
    DATE(timestamp) AS date,
    AVG(total_value) AS avg_value,
    MAX(total_value) AS high,
    MIN(total_value) AS low
FROM portfolio_values
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY date
ORDER BY date;
```

### Trade History
```sql
-- Recent trades
SELECT * FROM trades 
ORDER BY timestamp DESC 
LIMIT 10;

-- Trade summary by symbol
SELECT 
    symbol,
    COUNT(*) AS num_trades,
    SUM(CASE WHEN action = 'buy' THEN quantity ELSE 0 END) AS total_bought,
    SUM(CASE WHEN action = 'sell' THEN quantity ELSE 0 END) AS total_sold,
    SUM(commission) AS total_commission
FROM trades
GROUP BY symbol;
```

## 🔧 Database Management

### Extensions
```sql
-- List extensions
SELECT * FROM pg_extension;

-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Check versions
SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

### Hypertables
```sql
-- Create hypertable
SELECT create_hypertable('market_data', 'timestamp', 
    chunk_time_interval => INTERVAL '7 days');

-- List hypertables
SELECT * FROM timescaledb_information.hypertables;

-- View chunks
SELECT * FROM timescaledb_information.chunks 
WHERE hypertable_name = 'market_data';
```

### Compression
```sql
-- Enable compression
ALTER TABLE market_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

-- Add compression policy (compress data older than 30 days)
SELECT add_compression_policy('market_data', INTERVAL '30 days');

-- View compression stats
SELECT 
    hypertable_name,
    total_chunks,
    number_compressed_chunks,
    before_compression_total_bytes,
    after_compression_total_bytes,
    pg_size_pretty(before_compression_total_bytes) AS uncompressed,
    pg_size_pretty(after_compression_total_bytes) AS compressed
FROM timescaledb_information.hypertable_compression_stats;
```

### Retention
```sql
-- Add retention policy (delete data older than 1 year)
SELECT add_retention_policy('market_data', INTERVAL '1 year');

-- Remove retention policy
SELECT remove_retention_policy('market_data');

-- View retention policies
SELECT * FROM timescaledb_information.jobs 
WHERE proc_name = 'policy_retention';
```

## 🎯 Vector Search

### Similarity Search
```sql
-- Find similar feature vectors (cosine distance)
SELECT 
    id,
    symbol,
    feature_type,
    embedding <=> '[0.1, 0.2, ...]'::vector AS distance
FROM feature_embeddings
ORDER BY distance
LIMIT 10;

-- Find similar volatility regimes (L2 distance)
SELECT 
    regime_label,
    regime_vector <-> '[0.25, 22, -2.5, 5.0, ...]'::vector AS distance,
    realized_vol,
    vix_level
FROM volatility_regimes
ORDER BY distance
LIMIT 5;
```

### Vector Indexes
```sql
-- Create IVFFlat index
CREATE INDEX ON feature_embeddings 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Create HNSW index (better recall)
CREATE INDEX ON feature_embeddings 
USING hnsw (embedding vector_cosine_ops);

-- List indexes
SELECT indexname, tablename, indexdef 
FROM pg_indexes 
WHERE schemaname = 'public';
```

## 🔍 Monitoring

### Table Sizes
```sql
-- Table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Index sizes
SELECT 
    indexname,
    tablename,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY pg_relation_size(indexrelid) DESC;
```

### Query Performance
```sql
-- Explain query plan
EXPLAIN ANALYZE
SELECT * FROM market_data 
WHERE symbol = 'SPY' 
  AND timestamp >= NOW() - INTERVAL '30 days';

-- Active queries
SELECT 
    pid,
    now() - pg_stat_activity.query_start AS duration,
    query,
    state
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC;

-- Table statistics
SELECT * FROM pg_stat_user_tables WHERE schemaname = 'public';
```

### Connection Pool
```sql
-- Current connections
SELECT 
    COUNT(*),
    state
FROM pg_stat_activity
GROUP BY state;

-- Max connections
SHOW max_connections;

-- Kill connection
SELECT pg_terminate_backend(pid) WHERE pid = 12345;
```

## 🛠️ Maintenance

### Vacuum
```sql
-- Vacuum table
VACUUM market_data;

-- Analyze table (update statistics)
ANALYZE market_data;

-- Full vacuum (reclaim space)
VACUUM FULL market_data;

-- Auto-vacuum settings
SELECT * FROM pg_settings WHERE name LIKE 'autovacuum%';
```

### Backup & Restore
```bash
# Backup database
docker exec qvp-postgres pg_dump -U qvp_user qvp_db > backup.sql

# Backup specific table
docker exec qvp-postgres pg_dump -U qvp_user -d qvp_db -t market_data > market_data.sql

# Restore database
docker exec -i qvp-postgres psql -U qvp_user qvp_db < backup.sql

# Backup with compression
docker exec qvp-postgres pg_dump -U qvp_user qvp_db | gzip > backup.sql.gz
```

### Migrations
```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one version
alembic downgrade -1

# View history
alembic history

# Current version
alembic current
```

## 🐍 Python Snippets

### Basic Usage
```python
from qvp.data.postgres_connector import PostgreSQLConnector
from datetime import datetime, timedelta

# Initialize
pg = PostgreSQLConnector()
pg.create_tables()

# Insert data
data = [{
    'timestamp': datetime.now(),
    'symbol': 'SPY',
    'open': 450.0,
    'high': 452.5,
    'low': 449.0,
    'close': 451.5,
    'volume': 50000000
}]
pg.insert_market_data(data)

# Query
df = pg.get_market_data('SPY', start_time=datetime.now() - timedelta(days=7))

# Aggregates
hourly = pg.get_time_bucket_aggregates('SPY', interval='1 hour')

# Close
pg.close()
```

### Vector Store
```python
from qvp.data.vector_store import VectorStore
import numpy as np

# Initialize
vs = VectorStore()
vs.create_tables()

# Insert embedding
features = np.random.randn(128)
vs.insert_feature_embedding('SPY', features, 'technical')

# Similarity search
similar = vs.find_similar_features(features, top_k=10, metric='cosine')

# Regime matching
current = {'realized_vol': 0.25, 'vix_level': 22, 'skew': -2.5, 'kurtosis': 5.0}
regimes = vs.find_similar_regimes(current, top_k=5)
```

### Context Manager
```python
with pg.get_session() as session:
    # Your queries here
    result = session.query(MarketData).filter_by(symbol='SPY').all()
    # Session auto-commits and closes
```

## 🔗 Useful Links

- **TimescaleDB Docs**: https://docs.timescale.com/
- **pgvector GitHub**: https://github.com/pgvector/pgvector
- **PostgreSQL Docs**: https://www.postgresql.org/docs/
- **Alembic Docs**: https://alembic.sqlalchemy.org/

## 📚 QVP Documentation

- `DATABASE_INTEGRATION.md` - Full PostgreSQL guide
- `POSTGRES_IMPLEMENTATION_SUMMARY.md` - Implementation overview
- `KDB_INTEGRATION.md` - KDB+/Q integration
- `ADVANCED_FEATURES.md` - Docker, Dashboards, Live Trading
