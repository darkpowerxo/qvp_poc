"""
Unit tests for backtesting engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from qvp.backtest import (
    BacktestEngine,
    Strategy,
    Order,
    OrderType,
    OrderSide,
    Portfolio
)


class SimpleTestStrategy(Strategy):
    """Simple buy-and-hold strategy for testing"""
    
    def __init__(self):
        self.initialized = False
        self.capital = 0
    
    def initialize(self, initial_capital: float):
        self.capital = initial_capital
        self.initialized = True
    
    def on_data(self, timestamp, data):
        orders = []
        
        # Buy on first bar
        if not self.initialized:
            return orders
        
        if 'TEST' in data:
            # Simple: buy 100 shares on first day
            if timestamp == list(data['TEST'].index)[0]:
                orders.append(Order(
                    symbol='TEST',
                    side=OrderSide.BUY,
                    quantity=100,
                    order_type=OrderType.MARKET,
                    timestamp=timestamp
                ))
        
        return orders


class TestPortfolio:
    """Test portfolio management"""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization"""
        portfolio = Portfolio(initial_capital=100000)
        
        assert portfolio.cash == 100000
        assert portfolio.initial_capital == 100000
        assert len(portfolio.positions) == 0
    
    def test_portfolio_long_fill(self):
        """Test processing a long fill"""
        from qvp.backtest.engine import Fill
        
        portfolio = Portfolio(initial_capital=100000)
        
        fill = Fill(
            order_id='test_1',
            symbol='SPY',
            side=OrderSide.BUY,
            quantity=100,
            price=400,
            timestamp=datetime.now(),
            commission=10
        )
        
        portfolio.process_fill(fill, current_price=400)
        
        assert 'SPY' in portfolio.positions
        assert portfolio.positions['SPY'].quantity == 100
        assert portfolio.positions['SPY'].avg_price == 400
        assert portfolio.cash < 100000  # Cash should decrease
    
    def test_portfolio_round_trip(self):
        """Test complete round trip trade"""
        from qvp.backtest.engine import Fill
        
        portfolio = Portfolio(initial_capital=100000)
        
        # Buy
        buy_fill = Fill(
            order_id='buy_1',
            symbol='SPY',
            side=OrderSide.BUY,
            quantity=100,
            price=400,
            timestamp=datetime.now(),
            commission=10
        )
        portfolio.process_fill(buy_fill, current_price=400)
        
        # Sell at profit
        sell_fill = Fill(
            order_id='sell_1',
            symbol='SPY',
            side=OrderSide.SELL,
            quantity=100,
            price=410,
            timestamp=datetime.now(),
            commission=10
        )
        portfolio.process_fill(sell_fill, current_price=410)
        
        # Position should be flat
        assert portfolio.positions['SPY'].is_flat
        
        # Should have profit (minus commissions)
        realized_pnl = portfolio.positions['SPY'].realized_pnl
        assert realized_pnl > 0
        assert np.isclose(realized_pnl, 1000, rtol=0.1)  # 100 * (410 - 400)


class TestBacktestEngine:
    """Test backtesting engine"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample price data"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'open': 100 + np.arange(10),
            'high': 102 + np.arange(10),
            'low': 98 + np.arange(10),
            'close': 100 + np.arange(10),
            'volume': 1000000
        }, index=dates)
        return df
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        engine = BacktestEngine(
            initial_capital=100000,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        assert engine.initial_capital == 100000
        assert engine.portfolio.cash == 100000
    
    def test_add_data(self, sample_data):
        """Test adding data to engine"""
        engine = BacktestEngine(
            initial_capital=100000,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        engine.add_data('TEST', sample_data)
        
        assert 'TEST' in engine.data
        assert len(engine.data['TEST']) > 0
    
    def test_run_backtest(self, sample_data):
        """Test running a backtest"""
        engine = BacktestEngine(
            initial_capital=100000,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        engine.add_data('TEST', sample_data)
        
        strategy = SimpleTestStrategy()
        results = engine.run(strategy)
        
        assert isinstance(results, pd.DataFrame)
        assert 'equity' in results.columns
        assert len(results) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
