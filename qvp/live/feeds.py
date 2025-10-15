"""
WebSocket Data Feeds

Real-time market data feeds using WebSocket connections (simulated and real).
"""

import asyncio
import json
from typing import Dict, List, Callable, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class MarketData:
    """Market data tick."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask
        }


class SimulatedDataFeed:
    """
    Simulated market data feed for testing.
    
    Generates realistic price movements with configurable parameters.
    """
    
    def __init__(
        self,
        symbols: List[str],
        initial_prices: Dict[str, float],
        volatility: float = 0.02,
        tick_interval: float = 1.0
    ):
        """
        Initialize simulated data feed.
        
        Args:
            symbols: List of symbols to simulate
            initial_prices: Initial prices for each symbol
            volatility: Daily volatility (annualized)
            tick_interval: Seconds between ticks
        """
        self.symbols = symbols
        self.prices = initial_prices.copy()
        self.volatility = volatility
        self.tick_interval = tick_interval
        self.subscribers: List[Callable] = []
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    def subscribe(self, callback: Callable):
        """Subscribe to market data updates."""
        self.subscribers.append(callback)
        logger.info(f"Subscriber added. Total subscribers: {len(self.subscribers)}")
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from market data updates."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"Subscriber removed. Total subscribers: {len(self.subscribers)}")
    
    async def _generate_tick(self, symbol: str) -> MarketData:
        """Generate a simulated price tick."""
        # Geometric Brownian Motion
        current_price = self.prices[symbol]
        dt = self.tick_interval / (252 * 6.5 * 3600)  # Convert to trading year fraction
        drift = 0.0
        shock = np.random.normal(0, 1)
        price_change = current_price * (drift * dt + self.volatility * np.sqrt(dt) * shock)
        
        new_price = current_price + price_change
        new_price = max(new_price, 0.01)  # Prevent negative prices
        
        self.prices[symbol] = new_price
        
        # Generate bid/ask spread (10 bps)
        spread = new_price * 0.001
        bid = new_price - spread / 2
        ask = new_price + spread / 2
        
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            price=new_price,
            volume=int(np.random.exponential(1000)),
            bid=bid,
            ask=ask
        )
    
    async def _publish_tick(self, tick: MarketData):
        """Publish tick to all subscribers."""
        if self.subscribers:
            await asyncio.gather(
                *[subscriber(tick) for subscriber in self.subscribers],
                return_exceptions=True
            )
    
    async def _run(self):
        """Main feed loop."""
        logger.info(f"Starting simulated data feed for {self.symbols}")
        
        try:
            while self.running:
                # Generate tick for each symbol
                for symbol in self.symbols:
                    if not self.running:
                        break
                    
                    tick = await self._generate_tick(symbol)
                    await self._publish_tick(tick)
                
                # Wait for next tick
                await asyncio.sleep(self.tick_interval)
        
        except asyncio.CancelledError:
            logger.info("Data feed cancelled")
        except Exception as e:
            logger.error(f"Error in data feed: {e}")
        finally:
            logger.info("Data feed stopped")
    
    async def start(self):
        """Start the data feed."""
        if self.running:
            logger.warning("Data feed already running")
            return
        
        self.running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Data feed started")
    
    async def stop(self):
        """Stop the data feed."""
        if not self.running:
            return
        
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Data feed stopped")
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        return self.prices.copy()


class WebSocketDataFeed:
    """
    Real WebSocket data feed client.
    
    Connects to real-time data providers (e.g., Polygon, Alpaca, IEX).
    """
    
    def __init__(
        self,
        url: str,
        symbols: List[str],
        api_key: Optional[str] = None
    ):
        """
        Initialize WebSocket client.
        
        Args:
            url: WebSocket URL
            symbols: Symbols to subscribe
            api_key: API key for authentication
        """
        self.url = url
        self.symbols = symbols
        self.api_key = api_key
        self.subscribers: List[Callable] = []
        self.running = False
        self._websocket = None
        self._task: Optional[asyncio.Task] = None
        self.reconnect_delay = 5
        self.max_reconnect_delay = 60
    
    def subscribe(self, callback: Callable):
        """Subscribe to market data updates."""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from market data updates."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def _connect(self):
        """Establish WebSocket connection."""
        try:
            import websockets
            
            logger.info(f"Connecting to {self.url}")
            self._websocket = await websockets.connect(self.url)
            
            # Authenticate if needed
            if self.api_key:
                auth_msg = json.dumps({
                    'action': 'authenticate',
                    'key': self.api_key
                })
                await self._websocket.send(auth_msg)
            
            # Subscribe to symbols
            subscribe_msg = json.dumps({
                'action': 'subscribe',
                'symbols': self.symbols
            })
            await self._websocket.send(subscribe_msg)
            
            logger.info(f"Subscribed to {len(self.symbols)} symbols")
            return True
        
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            # Parse message into MarketData
            # Format depends on data provider
            tick = MarketData(
                symbol=data.get('symbol', ''),
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                price=float(data.get('price', 0)),
                volume=int(data.get('volume', 0)),
                bid=data.get('bid'),
                ask=data.get('ask')
            )
            
            # Publish to subscribers
            if self.subscribers:
                await asyncio.gather(
                    *[subscriber(tick) for subscriber in self.subscribers],
                    return_exceptions=True
                )
        
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _run(self):
        """Main WebSocket loop with reconnection logic."""
        current_delay = self.reconnect_delay
        
        while self.running:
            try:
                # Connect
                connected = await self._connect()
                if not connected:
                    logger.warning(f"Connection failed, retrying in {current_delay}s")
                    await asyncio.sleep(current_delay)
                    current_delay = min(current_delay * 2, self.max_reconnect_delay)
                    continue
                
                # Reset delay on successful connection
                current_delay = self.reconnect_delay
                
                # Receive messages
                import websockets
                async for message in self._websocket:
                    if not self.running:
                        break
                    await self._handle_message(message)
            
            except asyncio.CancelledError:
                logger.info("WebSocket feed cancelled")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                logger.info(f"Reconnecting in {current_delay}s")
                await asyncio.sleep(current_delay)
                current_delay = min(current_delay * 2, self.max_reconnect_delay)
    
    async def start(self):
        """Start WebSocket feed."""
        if self.running:
            logger.warning("WebSocket feed already running")
            return
        
        self.running = True
        self._task = asyncio.create_task(self._run())
        logger.info("WebSocket feed started")
    
    async def stop(self):
        """Stop WebSocket feed."""
        if not self.running:
            return
        
        self.running = False
        
        if self._websocket:
            await self._websocket.close()
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("WebSocket feed stopped")
