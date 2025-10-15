"""
Main script to run QVP backtest demonstration
Shows complete workflow from data download to performance analysis
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qvp.data import DataIngester, DataStorage
from qvp.research import VolatilityEstimator
from qvp.backtest import BacktestEngine
from qvp.strategies.volatility_strategies import (
    VIXMeanReversionStrategy,
    SimpleVolatilityStrategy
)
from qvp.analytics.performance import PerformanceMetrics, generate_tearsheet
from qvp.config import config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run QVP backtest demonstration')
    parser.add_argument(
        '--native-tls',
        action='store_true',
        help='Use native TLS (disable SSL verification)'
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download of data even if cached'
    )
    return parser.parse_args()


def main():
    """
    Run complete QVP demonstration
    """
    # Parse command line arguments
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("QVP - Quantitative Volatility Platform Demo")
    logger.info("=" * 80)
    
    # Step 1: Download data
    logger.info("\n[Step 1] Downloading market data...")
    ingester = DataIngester(verify_ssl=not args.native_tls)
    
    # Download SPY and VIX data
    equity_data = ingester.download_equity_data(
        symbols=['SPY'],
        start_date='2021-01-01',
        end_date='2024-12-31',
        force_download=args.force_download
    )
    
    vix_data = ingester.download_vix_data(
        start_date='2021-01-01',
        end_date='2024-12-31',
        force_download=args.force_download
    )
    
    if 'SPY' not in equity_data or vix_data.empty:
        logger.error("Failed to download data")
        return
    
    spy_df = equity_data['SPY']
    logger.info(f"Downloaded {len(spy_df)} days of SPY data")
    logger.info(f"Downloaded {len(vix_data)} days of VIX data")
    
    # Step 2: Calculate volatility metrics
    logger.info("\n[Step 2] Calculating volatility metrics...")
    vol_metrics = VolatilityEstimator.calculate_all_estimators(spy_df, window=20)
    logger.info(f"Calculated {len(vol_metrics.columns)} volatility estimators")
    
    # Step 3: Run backtest - VIX Mean Reversion Strategy
    logger.info("\n[Step 3] Running VIX Mean Reversion Strategy backtest...")
    
    engine = BacktestEngine(
        initial_capital=1_000_000,
        start_date='2021-01-01',
        end_date='2024-12-31',
        commission_pct=0.0005,
        slippage_bps=2.0
    )
    
    # Add data to backtest
    engine.add_data('SPY', spy_df)
    engine.add_data('VIX', vix_data)
    
    # Create and run strategy
    strategy = VIXMeanReversionStrategy(
        lookback_period=20,
        entry_zscore=1.5,
        exit_zscore=0.5,
        position_size=0.1
    )
    
    results = engine.run(strategy)
    
    if results.empty:
        logger.error("Backtest produced no results")
        return
    
    logger.info(f"Backtest complete: {len(results)} data points")
    
    # Step 4: Calculate performance metrics
    logger.info("\n[Step 4] Calculating performance metrics...")
    
    equity_series = results['equity']
    metrics = PerformanceMetrics.calculate_all_metrics(
        equity_curve=equity_series,
        risk_free_rate=0.04
    )
    
    # Print key metrics
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE SUMMARY - VIX Mean Reversion Strategy")
    logger.info("=" * 80)
    logger.info(f"Total Return:        {metrics['total_return']:>10.2%}")
    logger.info(f"Annual Return:       {metrics['annual_return']:>10.2%}")
    logger.info(f"Volatility:          {metrics['volatility']:>10.2%}")
    logger.info(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
    logger.info(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
    logger.info(f"Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")
    logger.info(f"Max Drawdown:        {metrics['max_drawdown_pct']:>10.2%}")
    logger.info(f"Win Rate:            {metrics['win_rate']:>10.2%}")
    logger.info(f"Profit Factor:       {metrics['profit_factor']:>10.2f}")
    logger.info("=" * 80)
    
    # Step 5: Run Simple Volatility Strategy for comparison
    logger.info("\n[Step 5] Running Simple Volatility Filter Strategy...")
    
    engine2 = BacktestEngine(
        initial_capital=1_000_000,
        start_date='2021-01-01',
        end_date='2024-12-31',
        commission_pct=0.0005,
        slippage_bps=2.0
    )
    
    engine2.add_data('SPY', spy_df)
    engine2.add_data('VIX', vix_data)
    
    strategy2 = SimpleVolatilityStrategy(
        vix_threshold=20.0,
        position_size=0.95
    )
    
    results2 = engine2.run(strategy2)
    
    equity_series2 = results2['equity']
    metrics2 = PerformanceMetrics.calculate_all_metrics(equity_series2, 0.04)
    
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE SUMMARY - Simple Volatility Filter Strategy")
    logger.info("=" * 80)
    logger.info(f"Total Return:        {metrics2['total_return']:>10.2%}")
    logger.info(f"Annual Return:       {metrics2['annual_return']:>10.2%}")
    logger.info(f"Volatility:          {metrics2['volatility']:>10.2%}")
    logger.info(f"Sharpe Ratio:        {metrics2['sharpe_ratio']:>10.2f}")
    logger.info(f"Max Drawdown:        {metrics2['max_drawdown_pct']:>10.2%}")
    logger.info("=" * 80)
    
    # Step 6: Generate tearsheets
    logger.info("\n[Step 6] Generating performance tearsheets...")
    
    tearsheet1 = generate_tearsheet(results, "VIX Mean Reversion", 0.04)
    tearsheet2 = generate_tearsheet(results2, "Simple Vol Filter", 0.04)
    
    # Save results
    output_dir = Path('data/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results.to_csv(output_dir / 'vix_mean_reversion_equity.csv')
    results2.to_csv(output_dir / 'simple_vol_filter_equity.csv')
    tearsheet1.to_csv(output_dir / 'tearsheet_vix_mr.csv')
    tearsheet2.to_csv(output_dir / 'tearsheet_simple.csv')
    
    logger.info(f"\nResults saved to {output_dir}/")
    
    # Step 7: Summary statistics
    logger.info("\n[Step 7] Comparison Summary")
    logger.info("=" * 80)
    logger.info(f"{'Metric':<25} {'VIX Mean Rev':>15} {'Simple Filter':>15}")
    logger.info("-" * 80)
    logger.info(f"{'Sharpe Ratio':<25} {metrics['sharpe_ratio']:>15.2f} {metrics2['sharpe_ratio']:>15.2f}")
    logger.info(f"{'Annual Return':<25} {metrics['annual_return']:>14.2%} {metrics2['annual_return']:>14.2%}")
    logger.info(f"{'Max Drawdown':<25} {metrics['max_drawdown_pct']:>14.2%} {metrics2['max_drawdown_pct']:>14.2%}")
    logger.info(f"{'Win Rate':<25} {metrics['win_rate']:>14.2%} {metrics2['win_rate']:>14.2%}")
    logger.info("=" * 80)
    
    logger.info("\n✅ QVP Demo completed successfully!")
    logger.info("\nNext steps:")
    logger.info("  1. Explore the Jupyter notebooks in notebooks/")
    logger.info("  2. Customize strategies in qvp/strategies/")
    logger.info("  3. Modify config in config/config.yaml")
    logger.info("  4. Run tests with: pytest tests/ -v")


if __name__ == '__main__':
    main()
