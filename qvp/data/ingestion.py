"""
Data ingestion module for market data
Handles downloading and caching of equity, options, and volatility data
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from loguru import logger

from qvp.config import config


class DataIngester:
    """
    Handles market data ingestion from various sources.
    Primary source: yfinance for equity and volatility index data.
    """
    
    def __init__(self, data_dir: Optional[Path] = None, verify_ssl: bool = True):
        """
        Initialize data ingester
        
        Args:
            data_dir: Directory to store downloaded data
            verify_ssl: Whether to verify SSL certificates (set to False to bypass SSL errors)
        """
        self.data_dir = data_dir or config.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verify_ssl = verify_ssl
        
        # Configure session for yfinance
        if not verify_ssl:
            logger.warning("SSL verification disabled - this is not recommended for production use")
            # Disable SSL warnings
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # Set environment variables to disable SSL verification for curl-based backends
            import os
            os.environ['CURL_CA_BUNDLE'] = ''
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            os.environ['SSL_CERT_FILE'] = ''
            os.environ['SSL_CERT_DIR'] = ''
            
            # Monkey-patch yfinance to disable SSL verification
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # Patch curl_cffi to disable SSL verification
            try:
                from curl_cffi import requests as curl_requests
                original_request = curl_requests.Session.request
                
                def patched_request(self, method, url, **kwargs):
                    # Force SSL verification to False
                    kwargs['verify'] = False
                    return original_request(self, method, url, **kwargs)
                
                curl_requests.Session.request = patched_request
                logger.debug("Successfully patched curl_cffi SSL verification")
            except ImportError:
                logger.debug("curl_cffi not found, skipping patch")
            
        # Create a custom session for yfinance
        self.session = requests.Session()
        self.session.verify = self.verify_ssl
        if not verify_ssl:
            # Additional adapter configuration
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            from urllib3.util.ssl_ import create_urllib3_context
            
            # Create custom SSL context
            class SSLAdapter(HTTPAdapter):
                def init_poolmanager(self, *args, **kwargs):
                    context = create_urllib3_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                    kwargs['ssl_context'] = context
                    return super().init_poolmanager(*args, **kwargs)
            
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = SSLAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
    
    def download_equity_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        force_download: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Download historical equity price data
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, 1m, etc.)
            force_download: Force re-download even if cached
            
        Returns:
            Dictionary mapping symbols to DataFrames with OHLCV data
        """
        logger.info(f"Downloading equity data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        data_dict = {}
        
        for symbol in symbols:
            cache_file = self.cache_dir / f"{symbol}_{interval}_{start_date}_{end_date}.parquet"
            
            # Check cache
            if not force_download and cache_file.exists():
                logger.debug(f"Loading {symbol} from cache: {cache_file}")
                df = pd.read_parquet(cache_file)
                data_dict[symbol] = df
                continue
            
            # Download from yfinance
            try:
                ticker = yf.Ticker(symbol)
                ticker.session = self.session
                
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,  # Adjust for splits and dividends
                    actions=True
                )
                
                if df.empty:
                    logger.warning(f"No data downloaded for {symbol}")
                    continue
                
                # Clean data
                df = self._clean_ohlcv_data(df, symbol)
                
                # Cache to parquet
                df.to_parquet(cache_file, compression='snappy')
                logger.info(f"Downloaded and cached {len(df)} rows for {symbol}")
                
                data_dict[symbol] = df
                
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                continue
        
        return data_dict
    
    def download_vix_data(
        self,
        start_date: str,
        end_date: str,
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Download VIX index data
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_download: Force re-download even if cached
            
        Returns:
            DataFrame with VIX index data
        """
        symbol = "^VIX"
        logger.info(f"Downloading VIX data from {start_date} to {end_date}")
        
        cache_file = self.cache_dir / f"VIX_{start_date}_{end_date}.parquet"
        
        # Check cache
        if not force_download and cache_file.exists():
            logger.debug(f"Loading VIX from cache: {cache_file}")
            return pd.read_parquet(cache_file)
        
        # Download
        try:
            ticker = yf.Ticker(symbol)
            ticker.session = self.session
            
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                logger.error("No VIX data downloaded")
                return pd.DataFrame()
            
            # VIX specific cleaning
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index.name = 'date'
            
            # Remove any invalid values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            # Cache
            df.to_parquet(cache_file, compression='snappy')
            logger.info(f"Downloaded and cached {len(df)} rows of VIX data")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to download VIX data: {e}")
            return pd.DataFrame()
    
    def download_options_chain(
        self,
        symbol: str,
        expiration_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download options chain data for a symbol
        
        Args:
            symbol: Ticker symbol
            expiration_date: Specific expiration date (YYYY-MM-DD), or None for nearest
            
        Returns:
            Tuple of (calls DataFrame, puts DataFrame)
        """
        logger.info(f"Downloading options chain for {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            ticker.session = self.session
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                logger.warning(f"No options available for {symbol}")
                return pd.DataFrame(), pd.DataFrame()
            
            # Use specified expiration or nearest one
            if expiration_date and expiration_date in expirations:
                exp = expiration_date
            else:
                exp = expirations[0]
                logger.info(f"Using expiration date: {exp}")
            
            # Get option chain
            opt = ticker.option_chain(exp)
            
            calls = opt.calls.copy()
            puts = opt.puts.copy()
            
            # Add metadata
            calls['expiration'] = exp
            calls['symbol'] = symbol
            calls['option_type'] = 'call'
            
            puts['expiration'] = exp
            puts['symbol'] = symbol
            puts['option_type'] = 'put'
            
            logger.info(f"Downloaded {len(calls)} calls and {len(puts)} puts for {symbol} exp {exp}")
            
            return calls, puts
            
        except Exception as e:
            logger.error(f"Failed to download options for {symbol}: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _clean_ohlcv_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and validate OHLCV data
        
        Args:
            df: Raw OHLCV DataFrame
            symbol: Ticker symbol (for logging)
            
        Returns:
            Cleaned DataFrame
        """
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Remove rows with invalid data
        original_len = len(df)
        
        # Remove inf and -inf
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows where OHLC are all the same (possible data error)
        # But keep if volume is > 0 (could be legitimate low volatility day)
        
        # Remove negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df = df[df[col] > 0]
        
        # Remove rows with NaN in critical columns
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Validate OHLC relationships
        # High should be >= max(open, close) and low should be <= min(open, close)
        valid_ohlc = (
            (df['high'] >= df[['open', 'close']].max(axis=1)) &
            (df['low'] <= df[['open', 'close']].min(axis=1))
        )
        df = df[valid_ohlc]
        
        # Check for extreme price movements (potential errors)
        if len(df) > 1:
            returns = df['close'].pct_change()
            extreme_threshold = config.get('data.quality_checks.price_change_threshold', 0.5)
            df = df[abs(returns) < extreme_threshold]
        
        cleaned_len = len(df)
        if cleaned_len < original_len:
            logger.warning(
                f"{symbol}: Removed {original_len - cleaned_len} invalid rows "
                f"({(original_len - cleaned_len) / original_len * 100:.1f}%)"
            )
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df.index.name = 'date'
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Validate data quality against configured thresholds
        
        Args:
            df: DataFrame to validate
            symbol: Symbol name for logging
            
        Returns:
            True if data passes quality checks
        """
        if df.empty:
            logger.error(f"{symbol}: DataFrame is empty")
            return False
        
        # Check missing data percentage
        max_missing_pct = config.get('data.quality_checks.max_missing_pct', 0.05)
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        
        if missing_pct > max_missing_pct:
            logger.warning(
                f"{symbol}: High missing data percentage: {missing_pct:.2%} "
                f"(threshold: {max_missing_pct:.2%})"
            )
            return False
        
        # Check minimum volume
        if 'volume' in df.columns:
            min_volume = config.get('data.quality_checks.min_volume', 1000)
            low_volume_pct = (df['volume'] < min_volume).sum() / len(df)
            
            if low_volume_pct > 0.1:  # More than 10% of days have low volume
                logger.warning(
                    f"{symbol}: High percentage of low volume days: {low_volume_pct:.2%}"
                )
        
        logger.info(f"{symbol}: Data quality validation passed")
        return True


class DataStorage:
    """
    Handles storage and retrieval of processed data in Parquet/HDF5 format
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data storage
        
        Args:
            data_dir: Directory for data storage
        """
        self.data_dir = data_dir or config.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.storage_format = config.get('data.storage.format', 'parquet')
        logger.info(f"Initialized data storage at {self.data_dir} with format: {self.storage_format}")
    
    def save_equity_data(self, df: pd.DataFrame, symbol: str) -> Path:
        """
        Save equity data to storage
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Ticker symbol
            
        Returns:
            Path to saved file
        """
        if self.storage_format == 'parquet':
            filepath = self.data_dir / f"{symbol}_equity.parquet"
            df.to_parquet(filepath, compression='snappy')
        elif self.storage_format == 'hdf5':
            filepath = self.data_dir / f"{symbol}_equity.h5"
            df.to_hdf(filepath, key='data', mode='w', complevel=9)
        else:
            raise ValueError(f"Unsupported storage format: {self.storage_format}")
        
        logger.info(f"Saved {len(df)} rows for {symbol} to {filepath}")
        return filepath
    
    def load_equity_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load equity data from storage
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            DataFrame with OHLCV data, or None if not found
        """
        if self.storage_format == 'parquet':
            filepath = self.data_dir / f"{symbol}_equity.parquet"
            if filepath.exists():
                return pd.read_parquet(filepath)
        elif self.storage_format == 'hdf5':
            filepath = self.data_dir / f"{symbol}_equity.h5"
            if filepath.exists():
                return pd.read_hdf(filepath, key='data')
        
        logger.warning(f"No stored data found for {symbol}")
        return None
    
    def save_vix_data(self, df: pd.DataFrame) -> Path:
        """Save VIX data to storage"""
        return self.save_equity_data(df, "VIX")
    
    def load_vix_data(self) -> Optional[pd.DataFrame]:
        """Load VIX data from storage"""
        return self.load_equity_data("VIX")
