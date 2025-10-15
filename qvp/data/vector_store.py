"""
pgvector Integration for ML Feature Storage and Similarity Search

This module provides vector database capabilities using PostgreSQL's pgvector extension:
- Store ML feature embeddings
- Semantic similarity search
- Strategy pattern matching
- Volatility regime clustering
- Nearest neighbor lookups

Features:
- High-dimensional vector storage
- Fast similarity search (cosine, L2, inner product)
- Batch vector operations
- Integration with scikit-learn
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import Column, Integer, String, DateTime, Text, Float
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from loguru import logger

try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    logger.warning("pgvector not available. Vector search features disabled.")
    # Fallback for when pgvector is not installed
    Vector = None

from qvp.data.postgres_connector import PostgreSQLConnector

VectorBase = declarative_base()


# ============================================================================
# Vector Models
# ============================================================================

if PGVECTOR_AVAILABLE:
    class FeatureEmbedding(VectorBase):
        """
        ML feature embeddings table.
        
        Stores high-dimensional feature vectors for similarity search.
        """
        __tablename__ = 'feature_embeddings'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        timestamp = Column(DateTime, nullable=False, index=True)
        symbol = Column(String(20), nullable=False, index=True)
        feature_type = Column(String(50), nullable=False)  # 'technical', 'volatility', 'ml'
        embedding = Column(Vector(128))  # 128-dimensional vector
        metadata = Column(Text)  # JSON string with additional info
        
    
    class VolatilityRegime(VectorBase):
        """
        Volatility regime embeddings for clustering and pattern matching.
        """
        __tablename__ = 'volatility_regimes'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        timestamp = Column(DateTime, nullable=False, index=True)
        regime_label = Column(String(50), nullable=False)  # 'low', 'medium', 'high', 'crisis'
        regime_vector = Column(Vector(64))  # 64-dimensional regime signature
        realized_vol = Column(Float)
        vix_level = Column(Float)
        skew = Column(Float)
        kurtosis = Column(Float)
    
    
    class StrategyPattern(VectorBase):
        """
        Strategy pattern vectors for matching market conditions.
        """
        __tablename__ = 'strategy_patterns'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        timestamp = Column(DateTime, nullable=False)
        strategy_name = Column(String(100), nullable=False, index=True)
        market_state_vector = Column(Vector(256))  # Market state embedding
        performance_score = Column(Float)  # How well strategy performed
        sharpe_ratio = Column(Float)
        win_rate = Column(Float)


# ============================================================================
# Vector Store Connector
# ============================================================================

class VectorStore:
    """
    Vector database for ML features and similarity search.
    
    Uses pgvector extension for:
    - Storing high-dimensional feature vectors
    - Fast similarity search (cosine, L2, inner product)
    - Nearest neighbor queries
    - Clustering and pattern matching
    
    Examples
    --------
    >>> vs = VectorStore()
    >>> vs.create_tables()
    >>> 
    >>> # Store feature embedding
    >>> features = np.random.rand(128)
    >>> vs.insert_feature_embedding('SPY', features, 'technical')
    >>> 
    >>> # Find similar market states
    >>> similar = vs.find_similar_features(features, top_k=10)
    """
    
    def __init__(self, pg_connector: Optional[PostgreSQLConnector] = None):
        """
        Initialize vector store.
        
        Parameters
        ----------
        pg_connector : PostgreSQLConnector, optional
            PostgreSQL connector instance (creates new if None)
        """
        if not PGVECTOR_AVAILABLE:
            raise ImportError(
                "pgvector is required for vector store. "
                "Install with: pip install pgvector"
            )
        
        self.pg = pg_connector or PostgreSQLConnector()
        logger.info("Vector store initialized")
    
    def create_tables(self):
        """Create vector tables and enable pgvector extension."""
        # Enable pgvector extension
        with self.pg.engine.connect() as conn:
            try:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                logger.info("pgvector extension enabled")
            except Exception as e:
                logger.error(f"Failed to enable pgvector: {e}")
                raise
        
        # Create tables
        VectorBase.metadata.create_all(self.pg.engine)
        logger.info("Created vector tables")
    
    def insert_feature_embedding(
        self,
        symbol: str,
        embedding: np.ndarray,
        feature_type: str = 'technical',
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Insert a feature embedding vector.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        embedding : np.ndarray
            Feature vector (must be same dimension as defined in model)
        feature_type : str, default='technical'
            Type of features ('technical', 'volatility', 'ml')
        timestamp : datetime, optional
            Timestamp (default: now)
        metadata : dict, optional
            Additional metadata as JSON
        
        Returns
        -------
        embedding_id : int
            ID of inserted embedding
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ensure embedding is the right size
        if len(embedding) != 128:
            # Pad or truncate to 128 dimensions
            if len(embedding) < 128:
                embedding = np.pad(embedding, (0, 128 - len(embedding)))
            else:
                embedding = embedding[:128]
        
        feature_emb = FeatureEmbedding(
            timestamp=timestamp,
            symbol=symbol,
            feature_type=feature_type,
            embedding=embedding.tolist(),
            metadata=str(metadata) if metadata else None
        )
        
        with self.pg.get_session() as session:
            session.add(feature_emb)
            session.flush()
            emb_id = feature_emb.id
        
        logger.debug(f"Inserted feature embedding for {symbol}")
        return emb_id
    
    def find_similar_features(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        feature_type: Optional[str] = None,
        metric: str = 'cosine'
    ) -> pd.DataFrame:
        """
        Find similar feature vectors using similarity search.
        
        Parameters
        ----------
        query_vector : np.ndarray
            Query vector
        top_k : int, default=10
            Number of results to return
        feature_type : str, optional
            Filter by feature type
        metric : str, default='cosine'
            Distance metric: 'cosine', 'l2', or 'inner_product'
        
        Returns
        -------
        results : pd.DataFrame
            Similar features with distance scores
        """
        # Ensure query vector is right size
        if len(query_vector) != 128:
            if len(query_vector) < 128:
                query_vector = np.pad(query_vector, (0, 128 - len(query_vector)))
            else:
                query_vector = query_vector[:128]
        
        # Choose distance operator
        if metric == 'cosine':
            op = '<=>'  # Cosine distance
        elif metric == 'l2':
            op = '<->'  # L2 distance
        elif metric == 'inner_product':
            op = '<#>'  # Negative inner product
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Build query
        query = f"""
        SELECT
            id,
            timestamp,
            symbol,
            feature_type,
            embedding {op} :query_vector AS distance,
            metadata
        FROM feature_embeddings
        WHERE 1=1
        """
        
        params = {'query_vector': str(query_vector.tolist())}
        
        if feature_type:
            query += " AND feature_type = :feature_type"
            params['feature_type'] = feature_type
        
        query += f" ORDER BY distance LIMIT {top_k}"
        
        results = self.pg.execute_timescale_query(query, params)
        return results
    
    def insert_volatility_regime(
        self,
        regime_label: str,
        regime_features: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> int:
        """
        Insert a volatility regime with its characteristic vector.
        
        Parameters
        ----------
        regime_label : str
            Regime label ('low', 'medium', 'high', 'crisis')
        regime_features : dict
            Dictionary of regime characteristics
        timestamp : datetime, optional
            Timestamp (default: now)
        
        Returns
        -------
        regime_id : int
            ID of inserted regime
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create 64-dimensional regime vector from features
        feature_vector = np.array([
            regime_features.get('realized_vol', 0),
            regime_features.get('vix_level', 0),
            regime_features.get('skew', 0),
            regime_features.get('kurtosis', 0),
            regime_features.get('return_mean', 0),
            regime_features.get('return_std', 0),
            # Pad to 64 dimensions
            *([0] * 58)
        ])
        
        regime = VolatilityRegime(
            timestamp=timestamp,
            regime_label=regime_label,
            regime_vector=feature_vector.tolist(),
            realized_vol=regime_features.get('realized_vol'),
            vix_level=regime_features.get('vix_level'),
            skew=regime_features.get('skew'),
            kurtosis=regime_features.get('kurtosis')
        )
        
        with self.pg.get_session() as session:
            session.add(regime)
            session.flush()
            regime_id = regime.id
        
        logger.debug(f"Inserted volatility regime: {regime_label}")
        return regime_id
    
    def find_similar_regimes(
        self,
        current_features: Dict[str, float],
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Find similar historical volatility regimes.
        
        Parameters
        ----------
        current_features : dict
            Current market characteristics
        top_k : int, default=5
            Number of similar regimes to return
        
        Returns
        -------
        similar_regimes : pd.DataFrame
            Similar regimes with distance scores
        """
        # Create query vector
        query_vector = np.array([
            current_features.get('realized_vol', 0),
            current_features.get('vix_level', 0),
            current_features.get('skew', 0),
            current_features.get('kurtosis', 0),
            current_features.get('return_mean', 0),
            current_features.get('return_std', 0),
            *([0] * 58)
        ])
        
        query = f"""
        SELECT
            id,
            timestamp,
            regime_label,
            regime_vector <=> :query_vector AS distance,
            realized_vol,
            vix_level,
            skew,
            kurtosis
        FROM volatility_regimes
        ORDER BY distance
        LIMIT {top_k}
        """
        
        params = {'query_vector': str(query_vector.tolist())}
        results = self.pg.execute_timescale_query(query, params)
        
        return results
    
    def insert_strategy_pattern(
        self,
        strategy_name: str,
        market_state: np.ndarray,
        performance_score: float,
        sharpe_ratio: float,
        win_rate: float,
        timestamp: Optional[datetime] = None
    ) -> int:
        """
        Insert a strategy performance pattern.
        
        Parameters
        ----------
        strategy_name : str
            Name of strategy
        market_state : np.ndarray
            256-dimensional market state vector
        performance_score : float
            Strategy performance score
        sharpe_ratio : float
            Sharpe ratio achieved
        win_rate : float
            Win rate
        timestamp : datetime, optional
            Timestamp (default: now)
        
        Returns
        -------
        pattern_id : int
            ID of inserted pattern
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ensure vector is 256-dimensional
        if len(market_state) != 256:
            if len(market_state) < 256:
                market_state = np.pad(market_state, (0, 256 - len(market_state)))
            else:
                market_state = market_state[:256]
        
        pattern = StrategyPattern(
            timestamp=timestamp,
            strategy_name=strategy_name,
            market_state_vector=market_state.tolist(),
            performance_score=performance_score,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate
        )
        
        with self.pg.get_session() as session:
            session.add(pattern)
            session.flush()
            pattern_id = pattern.id
        
        logger.debug(f"Inserted strategy pattern for {strategy_name}")
        return pattern_id
    
    def find_best_strategy_for_market_state(
        self,
        current_market_state: np.ndarray,
        min_sharpe: float = 0.5,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Find best performing strategies for current market conditions.
        
        Parameters
        ----------
        current_market_state : np.ndarray
            Current market state vector
        min_sharpe : float, default=0.5
            Minimum Sharpe ratio filter
        top_k : int, default=5
            Number of strategies to return
        
        Returns
        -------
        strategies : pd.DataFrame
            Best matching strategies with scores
        """
        # Ensure vector is 256-dimensional
        if len(current_market_state) != 256:
            if len(current_market_state) < 256:
                current_market_state = np.pad(
                    current_market_state,
                    (0, 256 - len(current_market_state))
                )
            else:
                current_market_state = current_market_state[:256]
        
        query = f"""
        SELECT
            strategy_name,
            timestamp,
            market_state_vector <=> :market_state AS similarity,
            performance_score,
            sharpe_ratio,
            win_rate
        FROM strategy_patterns
        WHERE sharpe_ratio >= :min_sharpe
        ORDER BY similarity, performance_score DESC
        LIMIT {top_k}
        """
        
        params = {
            'market_state': str(current_market_state.tolist()),
            'min_sharpe': min_sharpe
        }
        
        results = self.pg.execute_timescale_query(query, params)
        return results
    
    def batch_insert_embeddings(
        self,
        embeddings: List[Dict[str, Any]]
    ) -> int:
        """
        Batch insert multiple embeddings for efficiency.
        
        Parameters
        ----------
        embeddings : list of dict
            List of embedding dictionaries with keys:
            'symbol', 'embedding', 'feature_type', 'timestamp', 'metadata'
        
        Returns
        -------
        n_inserted : int
            Number of embeddings inserted
        """
        objects = []
        
        for emb in embeddings:
            embedding = emb['embedding']
            if len(embedding) != 128:
                if len(embedding) < 128:
                    embedding = np.pad(embedding, (0, 128 - len(embedding)))
                else:
                    embedding = embedding[:128]
            
            obj = FeatureEmbedding(
                timestamp=emb.get('timestamp', datetime.now()),
                symbol=emb['symbol'],
                feature_type=emb.get('feature_type', 'technical'),
                embedding=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                metadata=str(emb.get('metadata')) if emb.get('metadata') else None
            )
            objects.append(obj)
        
        with self.pg.get_session() as session:
            session.bulk_save_objects(objects)
        
        logger.info(f"Batch inserted {len(objects)} embeddings")
        return len(objects)
    
    def create_vector_index(
        self,
        table_name: str,
        column_name: str,
        index_type: str = 'ivfflat'
    ):
        """
        Create a vector index for faster similarity search.
        
        Parameters
        ----------
        table_name : str
            Table name
        column_name : str
            Vector column name
        index_type : str, default='ivfflat'
            Index type: 'ivfflat' or 'hnsw'
        """
        if index_type == 'ivfflat':
            # IVFFlat index (good for most use cases)
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS {table_name}_{column_name}_idx
            ON {table_name}
            USING ivfflat ({column_name} vector_cosine_ops)
            WITH (lists = 100)
            """
        elif index_type == 'hnsw':
            # HNSW index (better recall, more memory)
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS {table_name}_{column_name}_idx
            ON {table_name}
            USING hnsw ({column_name} vector_cosine_ops)
            """
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        with self.pg.engine.connect() as conn:
            conn.execute(index_sql)
            conn.commit()
        
        logger.info(f"Created {index_type} index on {table_name}.{column_name}")
