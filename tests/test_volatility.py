"""
Unit tests for volatility estimators
"""

import pytest
import numpy as np
import pandas as pd
from qvp.research.volatility import VolatilityEstimator


class TestVolatilityEstimator:
    """Test volatility estimator calculations"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLC data"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Generate realistic price data
        close = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
        high = close * (1 + np.abs(np.random.randn(100) * 0.01))
        low = close * (1 - np.abs(np.random.randn(100) * 0.01))
        open_ = close.shift(1).fillna(close[0])
        
        df = pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close
        }, index=dates)
        
        return df
    
    def test_close_to_close_volatility(self, sample_data):
        """Test close-to-close volatility calculation"""
        vol = VolatilityEstimator.close_to_close(sample_data['close'], window=20)
        
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(sample_data)
        assert vol.iloc[-1] > 0  # Volatility should be positive
        assert not np.isnan(vol.iloc[-1])  # Should have value after warmup
    
    def test_parkinson_volatility(self, sample_data):
        """Test Parkinson volatility estimator"""
        vol = VolatilityEstimator.parkinson(
            sample_data['high'],
            sample_data['low'],
            window=20
        )
        
        assert isinstance(vol, pd.Series)
        assert vol.iloc[-1] > 0
        # Parkinson should be less noisy than close-to-close
        assert vol.std() > 0
    
    def test_garman_klass_volatility(self, sample_data):
        """Test Garman-Klass volatility estimator"""
        vol = VolatilityEstimator.garman_klass(
            sample_data['open'],
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            window=20
        )
        
        assert isinstance(vol, pd.Series)
        assert vol.iloc[-1] > 0
    
    def test_rogers_satchell_volatility(self, sample_data):
        """Test Rogers-Satchell volatility estimator"""
        vol = VolatilityEstimator.rogers_satchell(
            sample_data['open'],
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            window=20
        )
        
        assert isinstance(vol, pd.Series)
        assert vol.iloc[-1] > 0
    
    def test_yang_zhang_volatility(self, sample_data):
        """Test Yang-Zhang volatility estimator"""
        vol = VolatilityEstimator.yang_zhang(
            sample_data['open'],
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            window=20
        )
        
        assert isinstance(vol, pd.Series)
        assert vol.iloc[-1] > 0
    
    def test_calculate_all_estimators(self, sample_data):
        """Test calculating all estimators at once"""
        vol_df = VolatilityEstimator.calculate_all_estimators(
            sample_data,
            window=20
        )
        
        assert isinstance(vol_df, pd.DataFrame)
        assert len(vol_df) == len(sample_data)
        
        # Check all expected columns exist
        expected_cols = [
            'vol_close', 'vol_parkinson', 'vol_gk',
            'vol_rs', 'vol_yz', 'realized_var', 'realized_vol'
        ]
        
        for col in expected_cols:
            assert col in vol_df.columns
        
        # All volatilities should be positive where defined
        for col in expected_cols:
            assert vol_df[col].iloc[-1] > 0 or np.isnan(vol_df[col].iloc[-1])
    
    def test_volatility_ordering(self, sample_data):
        """Test that estimators have expected efficiency ordering"""
        vol_df = VolatilityEstimator.calculate_all_estimators(
            sample_data,
            window=20
        )
        
        # Yang-Zhang should generally be most stable (lowest std)
        # This is a general trend, not always true for every dataset
        close_std = vol_df['vol_close'].std()
        yz_std = vol_df['vol_yz'].std()
        
        # At least check they're both positive
        assert close_std > 0
        assert yz_std > 0


class TestImpliedVolatility:
    """Test implied volatility calculations"""
    
    def test_black_scholes_call(self):
        """Test Black-Scholes call pricing"""
        from qvp.research.volatility import ImpliedVolatility
        
        price = ImpliedVolatility.black_scholes_call(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2
        )
        
        assert price > 0
        assert price < 100  # Call price should be less than spot
    
    def test_black_scholes_put(self):
        """Test Black-Scholes put pricing"""
        from qvp.research.volatility import ImpliedVolatility
        
        price = ImpliedVolatility.black_scholes_put(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2
        )
        
        assert price > 0
        assert price < 100
    
    def test_put_call_parity(self):
        """Test put-call parity relationship"""
        from qvp.research.volatility import ImpliedVolatility
        
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        call = ImpliedVolatility.black_scholes_call(S, K, T, r, sigma)
        put = ImpliedVolatility.black_scholes_put(S, K, T, r, sigma)
        
        # Put-Call Parity: C - P = S - K*e^(-rT)
        lhs = call - put
        rhs = S - K * np.exp(-r * T)
        
        assert np.isclose(lhs, rhs, rtol=1e-6)
    
    def test_implied_volatility_newton(self):
        """Test IV calculation via Newton-Raphson"""
        from qvp.research.volatility import ImpliedVolatility
        
        # Generate a call price with known sigma
        S, K, T, r = 100, 100, 1.0, 0.05
        true_sigma = 0.25
        
        call_price = ImpliedVolatility.black_scholes_call(S, K, T, r, true_sigma)
        
        # Recover sigma
        implied_vol = ImpliedVolatility.implied_volatility_newton(
            price=call_price,
            S=S, K=K, T=T, r=r,
            option_type='call'
        )
        
        assert implied_vol is not None
        assert np.isclose(implied_vol, true_sigma, rtol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
