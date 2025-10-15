"""
Event-driven backtesting engine
Supports realistic simulation with proper handling of lookahead bias
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger

from qvp.config import config


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """
    Represents a trading order
    """
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    timestamp: datetime
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    order_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate order ID if not provided"""
        if self.order_id is None:
            self.order_id = f"{self.symbol}_{self.timestamp.strftime('%Y%m%d%H%M%S')}"


@dataclass
class Fill:
    """
    Represents an order fill/execution
    """
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """
    Represents a position in a security
    """
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Current market value (without current price)"""
        return self.quantity * self.avg_price
    
    @property
    def is_long(self) -> bool:
        """Is this a long position"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Is this a short position"""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """Is position closed"""
        return abs(self.quantity) < 1e-6


class Portfolio:
    """
    Portfolio manager tracking positions, cash, and P&L
    """
    
    def __init__(self, initial_capital: float):
        """
        Initialize portfolio
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Fill] = []
        self.equity_curve: List[Dict] = []
        
        logger.info(f"Initialized portfolio with capital: ${initial_capital:,.2f}")
    
    def process_fill(self, fill: Fill, current_price: float) -> None:
        """
        Process an order fill and update position
        
        Args:
            fill: Fill object
            current_price: Current market price for P&L calculation
        """
        symbol = fill.symbol
        
        # Initialize position if new
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        position = self.positions[symbol]
        
        # Calculate trade value
        trade_value = fill.quantity * fill.price
        
        # Update position based on side
        if fill.side == OrderSide.BUY:
            # Buying increases position
            if position.quantity >= 0:
                # Adding to long or opening long
                new_quantity = position.quantity + fill.quantity
                position.avg_price = (
                    (position.quantity * position.avg_price + trade_value) / new_quantity
                    if new_quantity > 0 else fill.price
                )
                position.quantity = new_quantity
            else:
                # Covering short
                if abs(position.quantity) >= fill.quantity:
                    # Partial or complete cover
                    pnl = fill.quantity * (position.avg_price - fill.price)
                    position.realized_pnl += pnl
                    position.quantity += fill.quantity
                else:
                    # Cover short and go long
                    cover_qty = abs(position.quantity)
                    pnl = cover_qty * (position.avg_price - fill.price)
                    position.realized_pnl += pnl
                    
                    long_qty = fill.quantity - cover_qty
                    position.quantity = long_qty
                    position.avg_price = fill.price
            
            # Decrease cash
            self.cash -= (trade_value + fill.commission)
            
        else:  # SELL
            # Selling decreases position
            if position.quantity <= 0:
                # Adding to short or opening short
                new_quantity = position.quantity - fill.quantity
                position.avg_price = (
                    (abs(position.quantity) * position.avg_price + trade_value) / abs(new_quantity)
                    if new_quantity < 0 else fill.price
                )
                position.quantity = new_quantity
            else:
                # Closing long
                if position.quantity >= fill.quantity:
                    # Partial or complete close
                    pnl = fill.quantity * (fill.price - position.avg_price)
                    position.realized_pnl += pnl
                    position.quantity -= fill.quantity
                else:
                    # Close long and go short
                    close_qty = position.quantity
                    pnl = close_qty * (fill.price - position.avg_price)
                    position.realized_pnl += pnl
                    
                    short_qty = fill.quantity - close_qty
                    position.quantity = -short_qty
                    position.avg_price = fill.price
            
            # Increase cash
            self.cash += (trade_value - fill.commission)
        
        # Record trade
        self.trades.append(fill)
    
    def get_market_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total market value of positions
        
        Args:
            prices: Dictionary of symbol -> current price
            
        Returns:
            Total market value
        """
        total = 0.0
        for symbol, position in self.positions.items():
            if not position.is_flat and symbol in prices:
                total += position.quantity * prices[symbol]
        return total
    
    def get_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """
        Calculate unrealized P&L
        
        Args:
            prices: Dictionary of symbol -> current price
            
        Returns:
            Total unrealized P&L
        """
        total = 0.0
        for symbol, position in self.positions.items():
            if not position.is_flat and symbol in prices:
                current_value = position.quantity * prices[symbol]
                cost_basis = position.quantity * position.avg_price
                total += (current_value - cost_basis)
        return total
    
    def get_total_pnl(self, prices: Dict[str, float]) -> float:
        """
        Calculate total P&L (realized + unrealized)
        
        Args:
            prices: Dictionary of symbol -> current price
            
        Returns:
            Total P&L
        """
        realized = sum(pos.realized_pnl for pos in self.positions.values())
        unrealized = self.get_unrealized_pnl(prices)
        return realized + unrealized
    
    def get_equity(self, prices: Dict[str, float]) -> float:
        """
        Calculate total equity (cash + market value)
        
        Args:
            prices: Dictionary of symbol -> current price
            
        Returns:
            Total equity
        """
        return self.cash + self.get_market_value(prices)
    
    def record_equity(self, timestamp: datetime, prices: Dict[str, float]) -> None:
        """
        Record equity curve point
        
        Args:
            timestamp: Current timestamp
            prices: Current prices
        """
        equity = self.get_equity(prices)
        market_value = self.get_market_value(prices)
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'cash': self.cash,
            'market_value': market_value,
            'pnl': self.get_total_pnl(prices)
        })
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        """
        Get equity curve as DataFrame
        
        Returns:
            DataFrame with equity curve data
        """
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        return df


class TransactionCostModel:
    """
    Models transaction costs including commissions and slippage
    """
    
    def __init__(
        self,
        commission_pct: float = 0.0005,
        slippage_bps: float = 2.0,
        min_commission: float = 1.0
    ):
        """
        Initialize cost model
        
        Args:
            commission_pct: Commission as percentage of trade value
            slippage_bps: Slippage in basis points
            min_commission: Minimum commission per trade
        """
        self.commission_pct = commission_pct
        self.slippage_bps = slippage_bps
        self.min_commission = min_commission
    
    def calculate_commission(self, quantity: float, price: float) -> float:
        """
        Calculate commission for a trade
        
        Args:
            quantity: Trade quantity
            price: Trade price
            
        Returns:
            Commission amount
        """
        trade_value = abs(quantity * price)
        commission = trade_value * self.commission_pct
        return max(commission, self.min_commission)
    
    def calculate_slippage(
        self,
        side: OrderSide,
        quantity: float,
        price: float,
        volume: Optional[float] = None
    ) -> float:
        """
        Calculate slippage for a trade
        
        Args:
            side: Order side
            quantity: Trade quantity
            price: Reference price
            volume: Market volume (optional, for volume-based slippage)
            
        Returns:
            Slippage-adjusted price
        """
        # Simple slippage model: fixed bps
        slippage_factor = self.slippage_bps / 10000.0
        
        if side == OrderSide.BUY:
            # Pay up when buying
            slipped_price = price * (1 + slippage_factor)
        else:
            # Get hit when selling
            slipped_price = price * (1 - slippage_factor)
        
        return slipped_price


class Strategy(ABC):
    """
    Abstract base class for trading strategies
    """
    
    @abstractmethod
    def on_data(
        self,
        timestamp: datetime,
        data: Dict[str, pd.Series]
    ) -> List[Order]:
        """
        Called on each bar of data
        
        Args:
            timestamp: Current timestamp
            data: Dictionary of symbol -> price data
            
        Returns:
            List of orders to submit
        """
        pass
    
    @abstractmethod
    def initialize(self, initial_capital: float) -> None:
        """
        Initialize strategy with capital
        
        Args:
            initial_capital: Starting capital
        """
        pass


class BacktestEngine:
    """
    Event-driven backtesting engine
    """
    
    def __init__(
        self,
        initial_capital: float,
        start_date: str,
        end_date: str,
        commission_pct: float = 0.0005,
        slippage_bps: float = 2.0
    ):
        """
        Initialize backtest engine
        
        Args:
            initial_capital: Starting capital
            start_date: Backtest start date
            end_date: Backtest end date
            commission_pct: Commission percentage
            slippage_bps: Slippage in basis points
        """
        self.initial_capital = initial_capital
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        self.portfolio = Portfolio(initial_capital)
        self.cost_model = TransactionCostModel(commission_pct, slippage_bps)
        
        self.current_timestamp: Optional[datetime] = None
        self.data: Dict[str, pd.DataFrame] = {}
        
        logger.info(
            f"Initialized backtest engine: ${initial_capital:,.2f} "
            f"from {start_date} to {end_date}"
        )
    
    def add_data(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Add market data for backtesting
        
        Args:
            symbol: Symbol identifier
            df: DataFrame with OHLCV data
        """
        # Filter to backtest date range
        df_filtered = df[
            (df.index >= self.start_date) & (df.index <= self.end_date)
        ].copy()
        
        self.data[symbol] = df_filtered
        logger.info(f"Added {len(df_filtered)} bars for {symbol}")
    
    def run(self, strategy: Strategy) -> pd.DataFrame:
        """
        Run backtest with given strategy
        
        Args:
            strategy: Strategy instance
            
        Returns:
            DataFrame with backtest results
        """
        logger.info("Starting backtest")
        
        # Initialize strategy
        strategy.initialize(self.initial_capital)
        
        # Get all unique timestamps across all data
        all_timestamps = sorted(set(
            ts for df in self.data.values() for ts in df.index
        ))
        
        logger.info(f"Running backtest over {len(all_timestamps)} time steps")
        
        for i, timestamp in enumerate(all_timestamps):
            self.current_timestamp = timestamp
            
            # Get current data for all symbols
            current_data = {}
            current_prices = {}
            
            for symbol, df in self.data.items():
                if timestamp in df.index:
                    current_data[symbol] = df.loc[timestamp]
                    current_prices[symbol] = df.loc[timestamp, 'close']
            
            # Call strategy
            orders = strategy.on_data(timestamp, current_data)
            
            # Process orders
            for order in orders:
                self._process_order(order, current_prices)
            
            # Record equity
            self.portfolio.record_equity(timestamp, current_prices)
            
            # Progress logging
            if i % 100 == 0:
                equity = self.portfolio.get_equity(current_prices)
                pnl_pct = (equity / self.initial_capital - 1) * 100
                logger.debug(
                    f"Progress: {i}/{len(all_timestamps)} | "
                    f"Equity: ${equity:,.2f} ({pnl_pct:+.2f}%)"
                )
        
        logger.info("Backtest complete")
        
        return self.portfolio.get_equity_curve_df()
    
    def _process_order(
        self,
        order: Order,
        current_prices: Dict[str, float]
    ) -> None:
        """
        Process an order (simulate execution)
        
        Args:
            order: Order to process
            current_prices: Current market prices
        """
        if order.symbol not in current_prices:
            logger.warning(f"No price data for {order.symbol}, rejecting order")
            order.status = OrderStatus.REJECTED
            return
        
        # For simplicity, fill market orders immediately
        if order.order_type == OrderType.MARKET:
            fill_price = current_prices[order.symbol]
            
            # Apply slippage
            fill_price = self.cost_model.calculate_slippage(
                order.side,
                order.quantity,
                fill_price
            )
            
            # Calculate commission
            commission = self.cost_model.calculate_commission(
                order.quantity,
                fill_price
            )
            
            # Create fill
            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                timestamp=self.current_timestamp,
                commission=commission,
                slippage=abs(fill_price - current_prices[order.symbol])
            )
            
            # Update portfolio
            self.portfolio.process_fill(fill, current_prices[order.symbol])
            
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = fill_price
            order.commission = commission
