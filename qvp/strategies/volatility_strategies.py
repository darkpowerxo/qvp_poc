"""
VIX mean reversion and volatility trading strategies
"""

from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
from loguru import logger

from qvp.backtest.engine import Strategy, Order, OrderType, OrderSide
from qvp.research.features import VolatilityFeatures


class VIXMeanReversionStrategy(Strategy):
    """
    VIX mean reversion strategy
    Goes long VIX when it's below mean, short when above
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        entry_zscore: float = 1.5,
        exit_zscore: float = 0.5,
        position_size: float = 0.1
    ):
        """
        Initialize VIX mean reversion strategy
        
        Args:
            lookback_period: Period for calculating mean and std
            entry_zscore: Z-score threshold for entry
            exit_zscore: Z-score threshold for exit
            position_size: Position size as fraction of capital
        """
        self.lookback_period = lookback_period
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.position_size = position_size
        
        self.capital = 0.0
        self.current_position = 0.0
        self.vix_history = pd.Series(dtype=float)
        
        logger.info(
            f"Initialized VIX Mean Reversion: lookback={lookback_period}, "
            f"entry_z={entry_zscore}, exit_z={exit_zscore}"
        )
    
    def initialize(self, initial_capital: float) -> None:
        """Initialize with capital"""
        self.capital = initial_capital
    
    def on_data(
        self,
        timestamp: datetime,
        data: Dict[str, pd.Series]
    ) -> List[Order]:
        """
        Generate trading signals based on VIX mean reversion
        
        Args:
            timestamp: Current timestamp
            data: Market data
            
        Returns:
            List of orders
        """
        orders = []
        
        # Get VIX data
        if 'VIX' not in data:
            return orders
        
        vix_close = data['VIX']['close']
        
        # Update history
        self.vix_history.loc[timestamp] = vix_close
        
        # Need enough history
        if len(self.vix_history) < self.lookback_period:
            return orders
        
        # Calculate z-score
        recent_vix = self.vix_history.iloc[-self.lookback_period:]
        mean_vix = recent_vix.mean()
        std_vix = recent_vix.std()
        
        if std_vix == 0:
            return orders
        
        zscore = (vix_close - mean_vix) / std_vix
        
        # Trading logic
        # Long VIX when below mean (expect reversion up)
        # Short VIX when above mean (expect reversion down)
        
        target_position = 0.0
        
        if zscore < -self.entry_zscore:
            # VIX very low, go long
            target_position = 1.0
        elif zscore > self.entry_zscore:
            # VIX very high, go short
            target_position = -1.0
        elif abs(zscore) < self.exit_zscore:
            # Near mean, exit position
            target_position = 0.0
        else:
            # Hold current position
            target_position = self.current_position
        
        # Generate order if position needs to change
        if target_position != self.current_position:
            # Calculate quantity
            position_value = self.capital * self.position_size
            quantity = abs(position_value / vix_close)
            
            if target_position > self.current_position:
                # Need to buy
                side = OrderSide.BUY
                qty = quantity * (target_position - self.current_position)
            elif target_position < self.current_position:
                # Need to sell
                side = OrderSide.SELL
                qty = quantity * (self.current_position - target_position)
            else:
                return orders
            
            order = Order(
                symbol='VIX',
                side=side,
                quantity=qty,
                order_type=OrderType.MARKET,
                timestamp=timestamp
            )
            
            orders.append(order)
            self.current_position = target_position
            
            logger.debug(
                f"{timestamp.date()}: VIX={vix_close:.2f}, z={zscore:.2f}, "
                f"target_pos={target_position}, order={side.value} {qty:.0f}"
            )
        
        return orders


class VolatilityRiskPremiumStrategy(Strategy):
    """
    Volatility risk premium strategy
    Sells volatility when IV > RV (collects premium)
    """
    
    def __init__(
        self,
        min_premium: float = 0.02,
        holding_period: int = 5,
        position_size: float = 0.15
    ):
        """
        Initialize volatility risk premium strategy
        
        Args:
            min_premium: Minimum IV - RV spread to enter
            holding_period: Days to hold position
            position_size: Position size as fraction of capital
        """
        self.min_premium = min_premium
        self.holding_period = holding_period
        self.position_size = position_size
        
        self.capital = 0.0
        self.entry_date = None
        self.days_held = 0
        
        logger.info(
            f"Initialized Vol Risk Premium: min_premium={min_premium}, "
            f"holding={holding_period}"
        )
    
    def initialize(self, initial_capital: float) -> None:
        """Initialize with capital"""
        self.capital = initial_capital
    
    def on_data(
        self,
        timestamp: datetime,
        data: Dict[str, pd.Series]
    ) -> List[Order]:
        """
        Generate signals based on vol risk premium
        
        Args:
            timestamp: Current timestamp
            data: Market data
            
        Returns:
            List of orders
        """
        orders = []
        
        # This is a simplified version - in practice, would use actual IV and RV
        # For demo, using VIX as proxy for IV
        
        if 'VIX' not in data or 'SPY' not in data:
            return orders
        
        vix = data['VIX']['close']
        
        # Simplified: assume RV is lower than VIX (normally need to calculate)
        # In production, would calculate realized vol from SPY returns
        implied_vol = vix / 100  # VIX to decimal
        realized_vol = implied_vol * 0.85  # Assume RV < IV
        
        premium = implied_vol - realized_vol
        
        # Track holding period
        if self.entry_date is not None:
            self.days_held += 1
        
        # Exit if held long enough
        if self.days_held >= self.holding_period:
            # Close position
            # Simplified: just reset
            self.entry_date = None
            self.days_held = 0
            logger.debug(f"{timestamp.date()}: Closing position after {self.holding_period} days")
        
        # Enter if premium is attractive and not in position
        elif premium > self.min_premium and self.entry_date is None:
            # Sell volatility (simplified as selling VIX)
            position_value = self.capital * self.position_size
            quantity = position_value / vix
            
            order = Order(
                symbol='VIX',
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.MARKET,
                timestamp=timestamp
            )
            
            orders.append(order)
            self.entry_date = timestamp
            self.days_held = 0
            
            logger.debug(
                f"{timestamp.date()}: Selling vol: IV={implied_vol:.3f}, "
                f"RV={realized_vol:.3f}, premium={premium:.3f}"
            )
        
        return orders


class SimpleVolatilityStrategy(Strategy):
    """
    Simple volatility strategy for demonstration
    Buys SPY when VIX is low, stays in cash when VIX is high
    """
    
    def __init__(
        self,
        vix_threshold: float = 20.0,
        position_size: float = 0.95
    ):
        """
        Initialize simple volatility strategy
        
        Args:
            vix_threshold: VIX level threshold
            position_size: Position size as fraction of capital
        """
        self.vix_threshold = vix_threshold
        self.position_size = position_size
        self.capital = 0.0
        self.in_position = False
        
        logger.info(f"Initialized Simple Vol Strategy: VIX threshold={vix_threshold}")
    
    def initialize(self, initial_capital: float) -> None:
        """Initialize with capital"""
        self.capital = initial_capital
    
    def on_data(
        self,
        timestamp: datetime,
        data: Dict[str, pd.Series]
    ) -> List[Order]:
        """
        Generate signals based on VIX level
        
        Args:
            timestamp: Current timestamp
            data: Market data
            
        Returns:
            List of orders
        """
        orders = []
        
        if 'VIX' not in data or 'SPY' not in data:
            return orders
        
        vix = data['VIX']['close']
        spy_price = data['SPY']['close']
        
        # Simple logic: buy SPY when VIX < threshold, sell when VIX > threshold
        if vix < self.vix_threshold and not self.in_position:
            # Enter long SPY
            position_value = self.capital * self.position_size
            quantity = position_value / spy_price
            
            order = Order(
                symbol='SPY',
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
                timestamp=timestamp
            )
            
            orders.append(order)
            self.in_position = True
            
            logger.debug(f"{timestamp.date()}: VIX={vix:.1f} < {self.vix_threshold}, buying SPY")
        
        elif vix > self.vix_threshold and self.in_position:
            # Exit SPY
            position_value = self.capital * self.position_size
            quantity = position_value / spy_price
            
            order = Order(
                symbol='SPY',
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.MARKET,
                timestamp=timestamp
            )
            
            orders.append(order)
            self.in_position = False
            
            logger.debug(f"{timestamp.date()}: VIX={vix:.1f} > {self.vix_threshold}, selling SPY")
        
        return orders
