"""
Feature engineering for volatility trading strategies
Includes technical indicators, regime detection, and ML features
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from loguru import logger


class TechnicalIndicators:
    """
    Technical indicator calculations for volatility and price data
    """
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            DataFrame with macd, signal, and histogram
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return pd.DataFrame({
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Bollinger Bands
        
        Args:
            prices: Price series
            period: Moving average period
            num_std: Number of standard deviations
            
        Returns:
            DataFrame with middle, upper, and lower bands
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return pd.DataFrame({
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'bandwidth': (upper - lower) / middle
        })
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Average True Range
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            ATR series
        """
        high_low = high - low
        high_close = abs(high - close.shift(1))
        low_close = abs(low - close.shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr


class VolatilityFeatures:
    """
    Volatility-specific feature engineering
    """
    
    @staticmethod
    def volatility_spread(
        implied_vol: pd.Series,
        realized_vol: pd.Series
    ) -> pd.Series:
        """
        Volatility risk premium (IV - RV)
        
        Args:
            implied_vol: Implied volatility
            realized_vol: Realized volatility
            
        Returns:
            Volatility spread
        """
        return implied_vol - realized_vol
    
    @staticmethod
    def volatility_ratio(
        short_vol: pd.Series,
        long_vol: pd.Series
    ) -> pd.Series:
        """
        Ratio of short-term to long-term volatility
        
        Args:
            short_vol: Short-term volatility
            long_vol: Long-term volatility
            
        Returns:
            Volatility ratio
        """
        return short_vol / long_vol
    
    @staticmethod
    def volatility_zscore(
        vol: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """
        Z-score of volatility (for mean reversion signals)
        
        Args:
            vol: Volatility series
            window: Lookback window
            
        Returns:
            Volatility z-score
        """
        mean = vol.rolling(window=window).mean()
        std = vol.rolling(window=window).std()
        
        zscore = (vol - mean) / std
        
        return zscore
    
    @staticmethod
    def volatility_percentile(
        vol: pd.Series,
        window: int = 252
    ) -> pd.Series:
        """
        Percentile rank of current volatility
        
        Args:
            vol: Volatility series
            window: Lookback window
            
        Returns:
            Percentile rank (0-100)
        """
        def percentile_rank(x):
            if len(x) == 0:
                return np.nan
            return (x < x.iloc[-1]).sum() / len(x) * 100
        
        return vol.rolling(window=window).apply(percentile_rank, raw=False)


class RollingStatistics:
    """
    Rolling window statistical features
    """
    
    @staticmethod
    def calculate_rolling_features(
        series: pd.Series,
        windows: List[int] = [5, 10, 20, 60]
    ) -> pd.DataFrame:
        """
        Calculate multiple rolling statistics
        
        Args:
            series: Input series
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        features = pd.DataFrame(index=series.index)
        
        for window in windows:
            prefix = f"rolling_{window}"
            
            features[f'{prefix}_mean'] = series.rolling(window=window).mean()
            features[f'{prefix}_std'] = series.rolling(window=window).std()
            features[f'{prefix}_min'] = series.rolling(window=window).min()
            features[f'{prefix}_max'] = series.rolling(window=window).max()
            features[f'{prefix}_skew'] = series.rolling(window=window).skew()
            features[f'{prefix}_kurt'] = series.rolling(window=window).kurt()
            
            # Distance from current value to rolling mean (standardized)
            features[f'{prefix}_zscore'] = (
                (series - features[f'{prefix}_mean']) / features[f'{prefix}_std']
            )
        
        return features


class RegimeDetection:
    """
    Volatility regime detection using various methods
    """
    
    @staticmethod
    def threshold_regimes(
        vol: pd.Series,
        low_threshold: float = 0.15,
        high_threshold: float = 0.30
    ) -> pd.Series:
        """
        Simple threshold-based regime classification
        
        Args:
            vol: Volatility series (annualized)
            low_threshold: Threshold for low volatility
            high_threshold: Threshold for high volatility
            
        Returns:
            Series with regime labels (0=low, 1=medium, 2=high)
        """
        regimes = pd.Series(index=vol.index, dtype=int)
        regimes[vol < low_threshold] = 0  # Low vol
        regimes[(vol >= low_threshold) & (vol < high_threshold)] = 1  # Medium vol
        regimes[vol >= high_threshold] = 2  # High vol
        
        return regimes
    
    @staticmethod
    def kmeans_regimes(
        vol: pd.Series,
        n_regimes: int = 3,
        features: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.Series, KMeans]:
        """
        K-means clustering for regime detection
        
        Args:
            vol: Volatility series
            n_regimes: Number of regimes
            features: Additional features for clustering (optional)
            
        Returns:
            Tuple of (regime labels, fitted model)
        """
        # Prepare features
        if features is None:
            # Use volatility and its rolling statistics
            X = pd.DataFrame({
                'vol': vol,
                'vol_ma20': vol.rolling(20).mean(),
                'vol_ma60': vol.rolling(60).mean(),
                'vol_std20': vol.rolling(20).std()
            }).dropna()
        else:
            X = features.dropna()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit K-means
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Create series with original index
        regimes = pd.Series(index=X.index, data=labels)
        
        # Reindex to match original series
        regimes = regimes.reindex(vol.index)
        
        return regimes, kmeans
    
    @staticmethod
    def quantile_regimes(
        vol: pd.Series,
        quantiles: List[float] = [0.33, 0.67]
    ) -> pd.Series:
        """
        Regime classification based on historical quantiles
        
        Args:
            vol: Volatility series
            quantiles: Quantile thresholds
            
        Returns:
            Series with regime labels
        """
        thresholds = vol.quantile(quantiles)
        
        regimes = pd.Series(index=vol.index, dtype=int)
        regimes[vol < thresholds.iloc[0]] = 0
        
        for i in range(len(thresholds) - 1):
            mask = (vol >= thresholds.iloc[i]) & (vol < thresholds.iloc[i + 1])
            regimes[mask] = i + 1
        
        regimes[vol >= thresholds.iloc[-1]] = len(thresholds)
        
        return regimes


class MLFeatureEngine:
    """
    Machine learning feature engineering pipeline
    """
    
    def __init__(self, n_components: int = 5, standardize: bool = True):
        """
        Initialize feature engine
        
        Args:
            n_components: Number of PCA components
            standardize: Whether to standardize features
        """
        self.n_components = n_components
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.pca = PCA(n_components=n_components)
        self.fitted = False
    
    def create_features(
        self,
        price_data: pd.DataFrame,
        vol_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set
        
        Args:
            price_data: DataFrame with OHLCV data
            vol_data: DataFrame with volatility estimates
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating feature set")
        
        features = pd.DataFrame(index=price_data.index)
        
        # Price-based features
        features['returns'] = price_data['close'].pct_change()
        features['log_returns'] = np.log(price_data['close'] / price_data['close'].shift(1))
        
        # Technical indicators
        features['rsi'] = TechnicalIndicators.rsi(price_data['close'])
        
        macd = TechnicalIndicators.macd(price_data['close'])
        features['macd'] = macd['macd']
        features['macd_signal'] = macd['signal']
        
        bb = TechnicalIndicators.bollinger_bands(price_data['close'])
        features['bb_bandwidth'] = bb['bandwidth']
        features['bb_position'] = (price_data['close'] - bb['lower']) / (bb['upper'] - bb['lower'])
        
        features['atr'] = TechnicalIndicators.atr(
            price_data['high'],
            price_data['low'],
            price_data['close']
        )
        
        # Volatility features
        if 'vol_close' in vol_data.columns:
            features['realized_vol'] = vol_data['vol_close']
            features['vol_zscore'] = VolatilityFeatures.volatility_zscore(vol_data['vol_close'])
            features['vol_percentile'] = VolatilityFeatures.volatility_percentile(vol_data['vol_close'])
        
        # Rolling statistics for returns
        rolling_stats = RollingStatistics.calculate_rolling_features(
            features['returns'],
            windows=[5, 10, 20]
        )
        features = pd.concat([features, rolling_stats], axis=1)
        
        logger.info(f"Created {len(features.columns)} features")
        
        return features
    
    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Fit PCA and transform features
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Transformed features with PCA components
        """
        # Remove NaN rows
        features_clean = features.dropna()
        
        if len(features_clean) == 0:
            logger.warning("No valid features after dropping NaN")
            return features
        
        # Standardize if needed
        if self.standardize:
            features_scaled = self.scaler.fit_transform(features_clean)
        else:
            features_scaled = features_clean.values
        
        # Fit PCA
        pca_components = self.pca.fit_transform(features_scaled)
        
        # Create DataFrame with PCA components
        pca_df = pd.DataFrame(
            pca_components,
            index=features_clean.index,
            columns=[f'pca_{i+1}' for i in range(self.n_components)]
        )
        
        # Reindex to match original
        pca_df = pca_df.reindex(features.index)
        
        self.fitted = True
        
        explained_var = self.pca.explained_variance_ratio_.sum()
        logger.info(
            f"PCA fit complete. {self.n_components} components explain "
            f"{explained_var:.1%} of variance"
        )
        
        return pca_df
    
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new features using fitted PCA
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Transformed features
        """
        if not self.fitted:
            raise ValueError("Must call fit_transform before transform")
        
        features_clean = features.dropna()
        
        if self.standardize:
            features_scaled = self.scaler.transform(features_clean)
        else:
            features_scaled = features_clean.values
        
        pca_components = self.pca.transform(features_scaled)
        
        pca_df = pd.DataFrame(
            pca_components,
            index=features_clean.index,
            columns=[f'pca_{i+1}' for i in range(self.n_components)]
        )
        
        return pca_df.reindex(features.index)
