"""
Volatility estimators and calculations
Implements multiple volatility estimation methods with varying efficiency/bias tradeoffs
"""

from typing import Optional, Dict
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


class VolatilityEstimator:
    """
    Collection of volatility estimators for financial time series.
    
    All estimators return annualized volatility estimates.
    Assumes 252 trading days per year for annualization.
    """
    
    TRADING_DAYS_PER_YEAR = 252
    
    @staticmethod
    def close_to_close(prices: pd.Series, window: int = 20) -> pd.Series:
        """
        Classic close-to-close volatility estimator
        
        σ = sqrt(1/n * Σ(r_i - μ)²) * sqrt(252)
        
        Args:
            prices: Close prices
            window: Rolling window size
            
        Returns:
            Annualized volatility series
        """
        returns = np.log(prices / prices.shift(1))
        vol = returns.rolling(window=window).std() * np.sqrt(
            VolatilityEstimator.TRADING_DAYS_PER_YEAR
        )
        return vol
    
    @staticmethod
    def parkinson(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        """
        Parkinson volatility estimator (1980)
        Uses high-low range, more efficient than close-to-close
        
        σ² = 1/(4*n*ln(2)) * Σ(ln(H_i/L_i))²
        
        Args:
            high: High prices
            low: Low prices
            window: Rolling window size
            
        Returns:
            Annualized volatility series
        """
        hl = np.log(high / low)
        hl2 = hl ** 2
        
        vol = np.sqrt(
            hl2.rolling(window=window).sum() / (4 * window * np.log(2))
        ) * np.sqrt(VolatilityEstimator.TRADING_DAYS_PER_YEAR)
        
        return vol
    
    @staticmethod
    def garman_klass(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Garman-Klass volatility estimator (1980)
        Incorporates OHLC data, ~5x more efficient than close-to-close
        
        σ² = 1/n * Σ[0.5*(ln(H/L))² - (2ln(2)-1)*(ln(C/O))²]
        
        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window size
            
        Returns:
            Annualized volatility series
        """
        hl = np.log(high / low)
        co = np.log(close / open_)
        
        gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
        
        vol = np.sqrt(
            gk.rolling(window=window).mean()
        ) * np.sqrt(VolatilityEstimator.TRADING_DAYS_PER_YEAR)
        
        return vol
    
    @staticmethod
    def rogers_satchell(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Rogers-Satchell volatility estimator (1991)
        Allows for non-zero drift, useful for trending markets
        
        σ² = 1/n * Σ[ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)]
        
        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window size
            
        Returns:
            Annualized volatility series
        """
        hc = np.log(high / close)
        ho = np.log(high / open_)
        lc = np.log(low / close)
        lo = np.log(low / open_)
        
        rs = hc * ho + lc * lo
        
        vol = np.sqrt(
            rs.rolling(window=window).mean()
        ) * np.sqrt(VolatilityEstimator.TRADING_DAYS_PER_YEAR)
        
        return vol
    
    @staticmethod
    def yang_zhang(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Yang-Zhang volatility estimator (2000)
        Combines overnight and intraday volatility, handles drift
        One of the most efficient unbiased estimators
        
        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window size
            
        Returns:
            Annualized volatility series
        """
        # Overnight volatility (close to open)
        co = np.log(open_ / close.shift(1))
        ov = co.rolling(window=window).var()
        
        # Open to close volatility
        oc = np.log(close / open_)
        oc_var = oc.rolling(window=window).var()
        
        # Rogers-Satchell component
        rs_vol = VolatilityEstimator.rogers_satchell(open_, high, low, close, window)
        rs_var = (rs_vol / np.sqrt(VolatilityEstimator.TRADING_DAYS_PER_YEAR)) ** 2
        
        # Combine components
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        yz_var = ov + k * oc_var + (1 - k) * rs_var
        
        vol = np.sqrt(yz_var) * np.sqrt(VolatilityEstimator.TRADING_DAYS_PER_YEAR)
        
        return vol
    
    @staticmethod
    def realized_variance(returns: pd.Series, window: int = 20) -> pd.Series:
        """
        Realized variance: sum of squared returns
        
        RV = Σ r_i²
        
        Args:
            returns: Log returns series
            window: Rolling window size
            
        Returns:
            Annualized realized variance series
        """
        rv = (returns ** 2).rolling(window=window).sum()
        return rv * VolatilityEstimator.TRADING_DAYS_PER_YEAR
    
    @staticmethod
    def calculate_all_estimators(
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate all volatility estimators for a DataFrame with OHLC data
        
        Args:
            df: DataFrame with columns: open, high, low, close
            window: Rolling window size
            
        Returns:
            DataFrame with all volatility estimates
        """
        logger.info(f"Calculating volatility estimators with window={window}")
        
        required_cols = ['open', 'high', 'low', 'close']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        result = pd.DataFrame(index=df.index)
        
        # Close-to-close
        result['vol_close'] = VolatilityEstimator.close_to_close(
            df['close'], window
        )
        
        # Parkinson
        result['vol_parkinson'] = VolatilityEstimator.parkinson(
            df['high'], df['low'], window
        )
        
        # Garman-Klass
        result['vol_gk'] = VolatilityEstimator.garman_klass(
            df['open'], df['high'], df['low'], df['close'], window
        )
        
        # Rogers-Satchell
        result['vol_rs'] = VolatilityEstimator.rogers_satchell(
            df['open'], df['high'], df['low'], df['close'], window
        )
        
        # Yang-Zhang
        result['vol_yz'] = VolatilityEstimator.yang_zhang(
            df['open'], df['high'], df['low'], df['close'], window
        )
        
        # Calculate log returns for realized variance
        returns = np.log(df['close'] / df['close'].shift(1))
        result['realized_var'] = VolatilityEstimator.realized_variance(
            returns, window
        )
        result['realized_vol'] = np.sqrt(result['realized_var'])
        
        logger.info(f"Calculated {len(result.columns)} volatility estimators")
        
        return result


class ImpliedVolatility:
    """
    Implied volatility calculations and surface modeling
    """
    
    @staticmethod
    def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Black-Scholes call option price
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            
        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        return call
    
    @staticmethod
    def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Black-Scholes put option price
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            
        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        return put
    
    @staticmethod
    def implied_volatility_newton(
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            price: Observed option price
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility, or None if failed to converge
        """
        if T <= 0 or price <= 0:
            return None
        
        # Initial guess using Brenner-Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / T) * (price / S)
        
        for i in range(max_iterations):
            # Calculate option price and vega
            if option_type == 'call':
                calc_price = ImpliedVolatility.black_scholes_call(S, K, T, r, sigma)
            else:
                calc_price = ImpliedVolatility.black_scholes_put(S, K, T, r, sigma)
            
            # Vega (derivative of price w.r.t. sigma)
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            vega = S * stats.norm.pdf(d1) * np.sqrt(T)
            
            # Newton-Raphson update
            diff = calc_price - price
            
            if abs(diff) < tolerance:
                return sigma
            
            if vega < 1e-10:  # Avoid division by zero
                return None
            
            sigma = sigma - diff / vega
            
            # Keep sigma positive
            if sigma <= 0:
                sigma = 0.01
        
        logger.warning(f"IV calculation did not converge after {max_iterations} iterations")
        return None
    
    @staticmethod
    def calculate_iv_from_options(
        options_df: pd.DataFrame,
        spot_price: float,
        risk_free_rate: float = 0.04
    ) -> pd.DataFrame:
        """
        Calculate implied volatility for options chain
        
        Args:
            options_df: Options data with columns: strike, lastPrice, expiration, option_type
            spot_price: Current spot price
            risk_free_rate: Risk-free rate
            
        Returns:
            DataFrame with added 'implied_vol' column
        """
        result = options_df.copy()
        result['implied_vol'] = None
        
        for idx, row in result.iterrows():
            # Calculate time to expiration
            if isinstance(row['expiration'], str):
                exp_date = pd.to_datetime(row['expiration'])
            else:
                exp_date = row['expiration']
            
            T = (exp_date - pd.Timestamp.now()).days / 365.0
            
            if T <= 0:
                continue
            
            iv = ImpliedVolatility.implied_volatility_newton(
                price=row['lastPrice'],
                S=spot_price,
                K=row['strike'],
                T=T,
                r=risk_free_rate,
                option_type=row.get('option_type', 'call')
            )
            
            result.at[idx, 'implied_vol'] = iv
        
        return result
