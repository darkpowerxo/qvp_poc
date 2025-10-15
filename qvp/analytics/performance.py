"""
Performance analytics and metrics calculation
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger


class PerformanceMetrics:
    """
    Calculate trading performance metrics
    """
    
    TRADING_DAYS_PER_YEAR = 252
    
    @staticmethod
    def calculate_returns(equity_curve: pd.Series) -> pd.Series:
        """
        Calculate returns from equity curve
        
        Args:
            equity_curve: Equity time series
            
        Returns:
            Returns series
        """
        return equity_curve.pct_change().fillna(0)
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sharpe ratio
        
        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sortino ratio (uses downside deviation)
        
        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_deviation = downside_returns.std()
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_deviation
    
    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown
        
        Args:
            equity_curve: Equity time series
            
        Returns:
            Dictionary with max_drawdown, max_drawdown_pct, duration
        """
        if len(equity_curve) == 0:
            return {'max_drawdown': 0.0, 'max_drawdown_pct': 0.0, 'duration': 0}
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = equity_curve - running_max
        drawdown_pct = drawdown / running_max
        
        max_dd = drawdown.min()
        max_dd_pct = drawdown_pct.min()
        
        # Find drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = in_drawdown.astype(int).groupby(
            (in_drawdown != in_drawdown.shift()).cumsum()
        ).sum()
        
        max_duration = drawdown_periods.max() if len(drawdown_periods) > 0 else 0
        
        return {
            'max_drawdown': abs(max_dd),
            'max_drawdown_pct': abs(max_dd_pct),
            'duration': max_duration
        }
    
    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        equity_curve: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown)
        
        Args:
            returns: Return series
            equity_curve: Equity curve
            periods_per_year: Trading periods per year
            
        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * periods_per_year
        max_dd = PerformanceMetrics.max_drawdown(equity_curve)['max_drawdown_pct']
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / max_dd
    
    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """
        Calculate win rate (percentage of positive returns)
        
        Args:
            returns: Return series
            
        Returns:
            Win rate (0-1)
        """
        if len(returns) == 0:
            return 0.0
        
        winning_days = (returns > 0).sum()
        return winning_days / len(returns)
    
    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        """
        Calculate profit factor (gross profits / gross losses)
        
        Args:
            returns: Return series
            
        Returns:
            Profit factor
        """
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return np.inf if gains > 0 else 0.0
        
        return gains / losses
    
    @staticmethod
    def calculate_all_metrics(
        equity_curve: pd.Series,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics
        
        Args:
            equity_curve: Equity time series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Dictionary of all metrics
        """
        logger.info("Calculating performance metrics")
        
        returns = PerformanceMetrics.calculate_returns(equity_curve)
        
        # Calculate metrics
        metrics = {}
        
        # Return metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annual_return = (1 + total_return) ** (periods_per_year / len(equity_curve)) - 1
        
        metrics['total_return'] = total_return
        metrics['annual_return'] = annual_return
        metrics['volatility'] = returns.std() * np.sqrt(periods_per_year)
        
        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = PerformanceMetrics.sharpe_ratio(
            returns, risk_free_rate, periods_per_year
        )
        metrics['sortino_ratio'] = PerformanceMetrics.sortino_ratio(
            returns, risk_free_rate, periods_per_year
        )
        
        # Drawdown metrics
        dd_metrics = PerformanceMetrics.max_drawdown(equity_curve)
        metrics.update(dd_metrics)
        
        metrics['calmar_ratio'] = PerformanceMetrics.calmar_ratio(
            returns, equity_curve, periods_per_year
        )
        
        # Win/loss metrics
        metrics['win_rate'] = PerformanceMetrics.win_rate(returns)
        metrics['profit_factor'] = PerformanceMetrics.profit_factor(returns)
        
        # Additional stats
        metrics['best_day'] = returns.max()
        metrics['worst_day'] = returns.min()
        metrics['avg_win'] = returns[returns > 0].mean() if (returns > 0).any() else 0.0
        metrics['avg_loss'] = returns[returns < 0].mean() if (returns < 0).any() else 0.0
        
        logger.info(f"Sharpe: {metrics['sharpe_ratio']:.2f}, Max DD: {metrics['max_drawdown_pct']:.2%}")
        
        return metrics


class RollingMetrics:
    """
    Calculate rolling performance metrics
    """
    
    @staticmethod
    def rolling_sharpe(
        returns: pd.Series,
        window: int = 60,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate rolling Sharpe ratio
        
        Args:
            returns: Return series
            window: Rolling window size
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Rolling Sharpe ratio series
        """
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        
        rolling_sharpe = np.sqrt(periods_per_year) * rolling_mean / rolling_std
        
        return rolling_sharpe
    
    @staticmethod
    def rolling_volatility(
        returns: pd.Series,
        window: int = 60,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate rolling volatility
        
        Args:
            returns: Return series
            window: Rolling window size
            periods_per_year: Trading periods per year
            
        Returns:
            Rolling volatility series (annualized)
        """
        return returns.rolling(window=window).std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def rolling_drawdown(equity_curve: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate rolling maximum drawdown
        
        Args:
            equity_curve: Equity time series
            window: Rolling window size
            
        Returns:
            Rolling drawdown series
        """
        def calc_dd(x):
            if len(x) == 0:
                return 0.0
            running_max = x.expanding().max()
            dd = (x - running_max) / running_max
            return dd.min()
        
        return equity_curve.rolling(window=window).apply(calc_dd, raw=False)


def generate_tearsheet(
    equity_curve: pd.DataFrame,
    strategy_name: str = "Strategy",
    risk_free_rate: float = 0.04
) -> pd.DataFrame:
    """
    Generate a performance tearsheet
    
    Args:
        equity_curve: DataFrame with equity curve data
        strategy_name: Name of strategy
        risk_free_rate: Annual risk-free rate
        
    Returns:
        DataFrame with formatted tearsheet
    """
    logger.info(f"Generating tearsheet for {strategy_name}")
    
    if 'equity' not in equity_curve.columns:
        logger.error("Equity curve must have 'equity' column")
        return pd.DataFrame()
    
    equity = equity_curve['equity']
    
    # Calculate all metrics
    metrics = PerformanceMetrics.calculate_all_metrics(equity, risk_free_rate)
    
    # Format as DataFrame
    tearsheet = pd.DataFrame({
        'Metric': metrics.keys(),
        'Value': metrics.values()
    })
    
    # Add strategy name
    tearsheet.insert(0, 'Strategy', strategy_name)
    
    return tearsheet
