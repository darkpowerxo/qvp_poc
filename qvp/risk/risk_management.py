"""
Risk management framework
Includes VaR, CVaR, stress testing, and risk limits
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


class RiskMetrics:
    """
    Calculate various risk metrics
    """
    
    @staticmethod
    def value_at_risk(
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Return series
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            method: 'historical', 'parametric', or 'cornish_fisher'
            
        Returns:
            VaR (positive number representing potential loss)
        """
        if len(returns) == 0:
            return 0.0
        
        if method == 'historical':
            # Historical VaR: empirical quantile
            var = -np.percentile(returns, (1 - confidence) * 100)
            
        elif method == 'parametric':
            # Parametric VaR: assumes normal distribution
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence)
            var = -(mean + z_score * std)
            
        elif method == 'cornish_fisher':
            # Cornish-Fisher VaR: accounts for skewness and kurtosis
            z = stats.norm.ppf(1 - confidence)
            s = returns.skew()
            k = returns.kurtosis()
            
            z_cf = (z + (z**2 - 1) * s / 6 +
                    (z**3 - 3*z) * k / 24 -
                    (2*z**3 - 5*z) * s**2 / 36)
            
            var = -(returns.mean() + z_cf * returns.std())
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return max(var, 0)  # VaR should be non-negative
    
    @staticmethod
    def conditional_var(
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
        
        Args:
            returns: Return series
            confidence: Confidence level
            method: 'historical' or 'parametric'
            
        Returns:
            CVaR (expected loss beyond VaR)
        """
        if len(returns) == 0:
            return 0.0
        
        var = RiskMetrics.value_at_risk(returns, confidence, method)
        
        if method == 'historical':
            # Historical CVaR: mean of returns worse than VaR
            threshold = -var
            tail_returns = returns[returns <= threshold]
            cvar = -tail_returns.mean() if len(tail_returns) > 0 else var
            
        elif method == 'parametric':
            # Parametric CVaR: assumes normal distribution
            mean = returns.mean()
            std = returns.std()
            z = stats.norm.ppf(1 - confidence)
            cvar = -(mean - std * stats.norm.pdf(z) / (1 - confidence))
            
        else:
            cvar = var
        
        return max(cvar, 0)
    
    @staticmethod
    def downside_risk(
        returns: pd.Series,
        target_return: float = 0.0
    ) -> float:
        """
        Calculate downside risk (semi-deviation)
        
        Args:
            returns: Return series
            target_return: Target/threshold return
            
        Returns:
            Downside risk
        """
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        return np.sqrt(np.mean((downside_returns - target_return) ** 2))


class StressTesting:
    """
    Stress testing and scenario analysis
    """
    
    @staticmethod
    def historical_stress_scenarios() -> Dict[str, Dict[str, float]]:
        """
        Define historical stress scenarios
        
        Returns:
            Dictionary of scenarios with asset shocks
        """
        scenarios = {
            'market_crash': {
                'SPY': -0.20,  # 20% drop
                'VIX': 2.0,    # VIX doubles
                'description': 'Market crash similar to 2020 COVID'
            },
            'volatility_spike': {
                'SPY': -0.10,
                'VIX': 1.5,
                'description': 'Volatility spike without severe crash'
            },
            'gradual_decline': {
                'SPY': -0.15,
                'VIX': 0.5,
                'description': 'Gradual bear market'
            }
        }
        
        return scenarios
    
    @staticmethod
    def apply_stress_scenario(
        positions: Dict[str, float],
        prices: Dict[str, float],
        scenario: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply stress scenario to portfolio
        
        Args:
            positions: Dictionary of symbol -> quantity
            prices: Dictionary of symbol -> current price
            scenario: Dictionary of symbol -> price shock (% change)
            
        Returns:
            Dictionary with stress test results
        """
        # Calculate current portfolio value
        current_value = sum(
            positions.get(symbol, 0) * prices.get(symbol, 0)
            for symbol in positions
        )
        
        # Apply shocks
        stressed_value = 0
        for symbol, quantity in positions.items():
            current_price = prices.get(symbol, 0)
            shock = scenario.get(symbol, 0)
            stressed_price = current_price * (1 + shock)
            stressed_value += quantity * stressed_price
        
        loss = current_value - stressed_value
        loss_pct = loss / current_value if current_value > 0 else 0
        
        return {
            'current_value': current_value,
            'stressed_value': stressed_value,
            'loss': loss,
            'loss_pct': loss_pct
        }
    
    @staticmethod
    def run_all_scenarios(
        positions: Dict[str, float],
        prices: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Run all stress scenarios
        
        Args:
            positions: Portfolio positions
            prices: Current prices
            
        Returns:
            DataFrame with scenario results
        """
        scenarios = StressTesting.historical_stress_scenarios()
        results = []
        
        for scenario_name, scenario_data in scenarios.items():
            # Extract shocks (exclude 'description')
            shocks = {k: v for k, v in scenario_data.items() if k != 'description'}
            
            result = StressTesting.apply_stress_scenario(positions, prices, shocks)
            result['scenario'] = scenario_name
            result['description'] = scenario_data.get('description', '')
            
            results.append(result)
        
        return pd.DataFrame(results)


class RiskLimitMonitor:
    """
    Monitor and enforce risk limits
    """
    
    def __init__(
        self,
        max_portfolio_vol: float = 0.15,
        max_var_95: float = 0.05,
        max_drawdown: float = 0.15,
        max_position_size: float = 0.2
    ):
        """
        Initialize risk limit monitor
        
        Args:
            max_portfolio_vol: Maximum portfolio volatility (annualized)
            max_var_95: Maximum VaR at 95% confidence
            max_drawdown: Maximum drawdown allowed
            max_position_size: Maximum position size (fraction of portfolio)
        """
        self.max_portfolio_vol = max_portfolio_vol
        self.max_var_95 = max_var_95
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        
        self.violations: List[Dict] = []
        
        logger.info(
            f"Initialized risk limits: vol={max_portfolio_vol:.1%}, "
            f"VaR={max_var_95:.1%}, DD={max_drawdown:.1%}"
        )
    
    def check_limits(
        self,
        portfolio_returns: pd.Series,
        positions: Dict[str, float],
        portfolio_value: float,
        current_drawdown: float
    ) -> Dict[str, bool]:
        """
        Check all risk limits
        
        Args:
            portfolio_returns: Portfolio return series
            positions: Current positions
            portfolio_value: Total portfolio value
            current_drawdown: Current drawdown level
            
        Returns:
            Dictionary of limit checks (True = violated)
        """
        violations = {}
        
        # Portfolio volatility
        if len(portfolio_returns) > 0:
            vol = portfolio_returns.std() * np.sqrt(252)
            violations['volatility'] = vol > self.max_portfolio_vol
            
            # VaR
            var = RiskMetrics.value_at_risk(portfolio_returns, 0.95)
            violations['var'] = var > self.max_var_95
        
        # Drawdown
        violations['drawdown'] = abs(current_drawdown) > self.max_drawdown
        
        # Position sizes
        position_violations = []
        for symbol, quantity in positions.items():
            position_pct = abs(quantity) / portfolio_value if portfolio_value > 0 else 0
            if position_pct > self.max_position_size:
                position_violations.append((symbol, position_pct))
        
        violations['position_size'] = len(position_violations) > 0
        
        # Log violations
        for limit, violated in violations.items():
            if violated:
                logger.warning(f"Risk limit violated: {limit}")
                self.violations.append({
                    'timestamp': pd.Timestamp.now(),
                    'limit': limit,
                    'details': position_violations if limit == 'position_size' else None
                })
        
        return violations
    
    def get_violations_df(self) -> pd.DataFrame:
        """Get DataFrame of historical violations"""
        if not self.violations:
            return pd.DataFrame()
        
        return pd.DataFrame(self.violations)
