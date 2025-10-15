"""
Async Strategy Base

Base class for async trading strategies in live simulation.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Optional
from datetime import datetime
from loguru import logger

from qvp.live.async_portfolio import AsyncPortfolio, PositionSide
from qvp.live.feeds import MarketData


class AsyncStrategy(ABC):
    """
    Base class for async trading strategies.
    
    Strategies receive real-time market data and make trading decisions
    asynchronously.
    """
    
    def __init__(
        self,
        name: str,
        portfolio: AsyncPortfolio,
        symbols: list
    ):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            portfolio: Async portfolio manager
            symbols: Symbols to trade
        """
        self.name = name
        self.portfolio = portfolio
        self.symbols = symbols
        self.market_data: Dict[str, MarketData] = {}
        self.running = False
    
    @abstractmethod
    async def on_market_data(self, tick: MarketData):
        """
        Handle incoming market data.
        
        Args:
            tick: Market data tick
        """
        pass
    
    async def update_positions(self):
        """Update all positions with current prices."""
        for symbol, tick in self.market_data.items():
            await self.portfolio.update_position_price(symbol, tick.price)
    
    async def start(self):
        """Start strategy."""
        self.running = True
        logger.info(f"Strategy '{self.name}' started")
    
    async def stop(self):
        """Stop strategy."""
        self.running = False
        logger.info(f"Strategy '{self.name}' stopped")


class SimpleVIXMeanReversion(AsyncStrategy):
    """
    Async VIX mean reversion strategy.
    
    Buys SPY when VIX is high (mean reversion signal).
    """
    
    def __init__(
        self,
        portfolio: AsyncPortfolio,
        vix_threshold_high: float = 25.0,
        vix_threshold_low: float = 15.0,
        position_size: float = 50000.0
    ):
        """
        Initialize strategy.
        
        Args:
            portfolio: Portfolio manager
            vix_threshold_high: VIX level to enter (buy signal)
            vix_threshold_low: VIX level to exit
            position_size: Dollar amount per position
        """
        super().__init__("VIX Mean Reversion", portfolio, ["SPY", "^VIX"])
        self.vix_threshold_high = vix_threshold_high
        self.vix_threshold_low = vix_threshold_low
        self.position_size = position_size
    
    async def on_market_data(self, tick: MarketData):
        """Handle market data and generate signals."""
        # Store tick
        self.market_data[tick.symbol] = tick
        
        # Need both SPY and VIX data
        if "SPY" not in self.market_data or "^VIX" not in self.market_data:
            return
        
        spy_price = self.market_data["SPY"].price
        vix_level = self.market_data["^VIX"].price
        
        # Update position prices
        await self.update_positions()
        
        # Check for signals
        current_position = await self.portfolio.get_position("SPY")
        
        # Entry signal: VIX high, no position
        if current_position is None and vix_level > self.vix_threshold_high:
            quantity = int(self.position_size / spy_price)
            if quantity > 0:
                success = await self.portfolio.open_position(
                    "SPY",
                    PositionSide.LONG,
                    quantity,
                    spy_price
                )
                if success:
                    logger.info(f"SIGNAL: VIX {vix_level:.2f} > {self.vix_threshold_high} - BUY SPY")
        
        # Exit signal: VIX low, has position
        elif current_position is not None and vix_level < self.vix_threshold_low:
            success = await self.portfolio.close_position("SPY", spy_price)
            if success:
                logger.info(f"SIGNAL: VIX {vix_level:.2f} < {self.vix_threshold_low} - SELL SPY")
