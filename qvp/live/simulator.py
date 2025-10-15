"""
Live Trading Simulator

Main simulator for live trading with async data feeds and strategies.
"""

import asyncio
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
from loguru import logger

from qvp.live.async_portfolio import AsyncPortfolio
from qvp.live.feeds import SimulatedDataFeed, MarketData
from qvp.live.async_strategy import AsyncStrategy, SimpleVIXMeanReversion


class LiveSimulator:
    """
    Live trading simulator.
    
    Orchestrates async data feeds, strategies, and portfolio management.
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        symbols: List[str] = None,
        initial_prices: dict = None
    ):
        """
        Initialize simulator.
        
        Args:
            initial_capital: Starting capital
            symbols: Symbols to trade
            initial_prices: Initial prices for simulation
        """
        self.initial_capital = initial_capital
        self.symbols = symbols or ["SPY", "^VIX"]
        self.initial_prices = initial_prices or {"SPY": 450.0, "^VIX": 18.0}
        
        # Components
        self.portfolio = AsyncPortfolio(initial_capital=initial_capital)
        self.data_feed = SimulatedDataFeed(
            symbols=self.symbols,
            initial_prices=self.initial_prices,
            volatility=0.15,  # 15% annualized vol
            tick_interval=1.0  # 1 second ticks
        )
        self.strategies: List[AsyncStrategy] = []
        
        # State
        self.running = False
        self._tasks: List[asyncio.Task] = []
        self.start_time: Optional[datetime] = None
    
    def add_strategy(self, strategy: AsyncStrategy):
        """Add a trading strategy."""
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.name}")
    
    async def _on_market_data(self, tick: MarketData):
        """Handle incoming market data."""
        # Update portfolio positions
        await self.portfolio.update_position_price(tick.symbol, tick.price)
        
        # Pass to strategies
        for strategy in self.strategies:
            if strategy.running:
                try:
                    await strategy.on_market_data(tick)
                except Exception as e:
                    logger.error(f"Error in strategy {strategy.name}: {e}")
    
    async def _record_equity_loop(self, interval: float = 5.0):
        """Periodically record portfolio equity."""
        try:
            while self.running:
                await self.portfolio.record_equity()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Equity recording stopped")
    
    async def _status_loop(self, interval: float = 10.0):
        """Periodically print status."""
        try:
            while self.running:
                stats = await self.portfolio.get_stats()
                logger.info(
                    f"Portfolio - Equity: ${stats['equity']:,.2f} | "
                    f"Positions: {stats['num_positions']} | "
                    f"P&L: ${stats['total_pnl']:,.2f} ({stats['return_pct']:.2f}%)"
                )
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Status logging stopped")
    
    async def start(self, duration: Optional[float] = None):
        """
        Start live simulation.
        
        Args:
            duration: Simulation duration in seconds (None for infinite)
        """
        if self.running:
            logger.warning("Simulator already running")
            return
        
        self.running = True
        self.start_time = datetime.now()
        logger.info("="*80)
        logger.info("STARTING LIVE TRADING SIMULATION")
        logger.info("="*80)
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Strategies: {len(self.strategies)}")
        
        # Subscribe to data feed
        self.data_feed.subscribe(self._on_market_data)
        
        # Start data feed
        await self.data_feed.start()
        
        # Start strategies
        for strategy in self.strategies:
            await strategy.start()
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._record_equity_loop()),
            asyncio.create_task(self._status_loop())
        ]
        
        # Run for duration or until stopped
        if duration:
            logger.info(f"Running simulation for {duration} seconds...")
            await asyncio.sleep(duration)
            await self.stop()
        else:
            logger.info("Running simulation until stopped...")
            # Keep running
            try:
                while self.running:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                await self.stop()
    
    async def stop(self):
        """Stop simulation."""
        if not self.running:
            return
        
        logger.info("Stopping simulation...")
        self.running = False
        
        # Stop strategies
        for strategy in self.strategies:
            await strategy.stop()
        
        # Stop data feed
        await self.data_feed.stop()
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Final stats
        stats = await self.portfolio.get_stats()
        logger.info("="*80)
        logger.info("SIMULATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Final Equity:     ${stats['equity']:,.2f}")
        logger.info(f"Total P&L:        ${stats['total_pnl']:,.2f}")
        logger.info(f"Return:           {stats['return_pct']:.2f}%")
        logger.info(f"Trades Executed:  {stats['num_trades']}")
        logger.info(f"Final Positions:  {stats['num_positions']}")
        
        # Save results
        self._save_results()
    
    def _save_results(self):
        """Save simulation results to files."""
        try:
            output_dir = Path("data/live_sim")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save equity history
            equity_df = self.portfolio.get_equity_dataframe()
            if not equity_df.empty:
                equity_file = output_dir / "equity_history.csv"
                equity_df.to_csv(equity_file)
                logger.info(f"Saved equity history to {equity_file}")
            
            # Save trade history
            trades_df = self.portfolio.get_trades_dataframe()
            if not trades_df.empty:
                trades_file = output_dir / "trade_history.csv"
                trades_df.to_csv(trades_file)
                logger.info(f"Saved trade history to {trades_file}")
        
        except Exception as e:
            logger.error(f"Error saving results: {e}")


async def main():
    """Run live simulation demo."""
    # Create simulator
    sim = LiveSimulator(
        initial_capital=1_000_000,
        symbols=["SPY", "^VIX"],
        initial_prices={"SPY": 450.0, "^VIX": 18.0}
    )
    
    # Add VIX mean reversion strategy
    strategy = SimpleVIXMeanReversion(
        portfolio=sim.portfolio,
        vix_threshold_high=22.0,  # Lower threshold for more action in simulation
        vix_threshold_low=16.0,
        position_size=100_000
    )
    sim.add_strategy(strategy)
    
    # Run for 60 seconds
    try:
        await sim.start(duration=60)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        await sim.stop()


if __name__ == "__main__":
    asyncio.run(main())
