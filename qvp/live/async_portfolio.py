"""
Async Portfolio Manager

Asynchronous portfolio management for live trading simulation.
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd
from loguru import logger


class PositionSide(Enum):
    """Position side."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_price(self, new_price: float):
        """Update current price and unrealized P&L."""
        self.current_price = new_price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.entry_price - new_price) * self.quantity
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'entry_time': self.entry_time.isoformat()
        }


@dataclass
class AsyncPortfolio:
    """
    Async Portfolio Manager for live trading.
    
    Manages positions, cash, and P&L in real-time with async operations.
    """
    initial_capital: float
    cash: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    equity_history: List[Dict] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    def __post_init__(self):
        """Initialize cash."""
        self.cash = self.initial_capital
    
    async def get_equity(self) -> float:
        """Get current portfolio equity."""
        async with self._lock:
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            return self.cash + total_unrealized_pnl
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        async with self._lock:
            return self.positions.get(symbol)
    
    async def open_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        price: float
    ) -> bool:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            side: LONG or SHORT
            quantity: Number of shares
            price: Entry price
            
        Returns:
            True if position opened successfully
        """
        async with self._lock:
            # Check if position already exists
            if symbol in self.positions:
                logger.warning(f"Position already exists for {symbol}")
                return False
            
            # Calculate cost
            cost = quantity * price
            
            # Check cash available
            if cost > self.cash:
                logger.warning(f"Insufficient cash for {symbol}: need {cost}, have {self.cash}")
                return False
            
            # Create position
            position = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                entry_time=datetime.now()
            )
            
            self.positions[symbol] = position
            self.cash -= cost
            
            # Record trade
            self.trade_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side.value,
                'quantity': quantity,
                'price': price,
                'action': 'OPEN'
            })
            
            logger.info(f"Opened {side.value} position: {quantity} {symbol} @ ${price:.2f}")
            return True
    
    async def close_position(self, symbol: str, price: float) -> bool:
        """
        Close an existing position.
        
        Args:
            symbol: Trading symbol
            price: Exit price
            
        Returns:
            True if position closed successfully
        """
        async with self._lock:
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return False
            
            position = self.positions[symbol]
            
            # Calculate realized P&L
            if position.side == PositionSide.LONG:
                realized_pnl = (price - position.entry_price) * position.quantity
            else:
                realized_pnl = (position.entry_price - price) * position.quantity
            
            # Update cash
            proceeds = position.quantity * price
            self.cash += proceeds
            
            # Record trade
            self.trade_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': position.side.value,
                'quantity': position.quantity,
                'price': price,
                'action': 'CLOSE',
                'pnl': realized_pnl
            })
            
            # Remove position
            del self.positions[symbol]
            
            logger.info(f"Closed position: {symbol} @ ${price:.2f}, P&L: ${realized_pnl:.2f}")
            return True
    
    async def update_position_price(self, symbol: str, price: float):
        """Update position with new market price."""
        async with self._lock:
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
    
    async def record_equity(self):
        """Record current equity to history."""
        equity = await self.get_equity()
        async with self._lock:
            self.equity_history.append({
                'timestamp': datetime.now(),
                'equity': equity,
                'cash': self.cash,
                'num_positions': len(self.positions),
                'total_unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values())
            })
    
    async def get_positions_summary(self) -> List[Dict]:
        """Get summary of all positions."""
        async with self._lock:
            return [pos.to_dict() for pos in self.positions.values()]
    
    async def get_stats(self) -> Dict:
        """Get portfolio statistics."""
        equity = await self.get_equity()
        async with self._lock:
            total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized = sum(
                trade['pnl'] 
                for trade in self.trade_history 
                if 'pnl' in trade
            )
            
            return {
                'equity': equity,
                'cash': self.cash,
                'num_positions': len(self.positions),
                'total_unrealized_pnl': total_unrealized,
                'total_realized_pnl': total_realized,
                'total_pnl': total_unrealized + total_realized,
                'return_pct': (equity - self.initial_capital) / self.initial_capital * 100,
                'num_trades': len([t for t in self.trade_history if t['action'] == 'CLOSE'])
            }
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity history as DataFrame."""
        if not self.equity_history:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_history).set_index('timestamp')
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_history).set_index('timestamp')
