-- QVP PostgreSQL Database Initialization Script
-- This script sets up TimescaleDB and pgvector extensions

-- Enable TimescaleDB extension for time-series optimization
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable pgvector extension for ML feature similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema for QVP tables (optional, can use public)
-- CREATE SCHEMA IF NOT EXISTS qvp;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE qvp_db TO qvp_user;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'QVP Database initialized successfully';
    RAISE NOTICE 'TimescaleDB version: %', (SELECT extversion FROM pg_extension WHERE extname = 'timescaledb');
    RAISE NOTICE 'pgvector version: %', (SELECT extversion FROM pg_extension WHERE extname = 'vector');
END $$;
