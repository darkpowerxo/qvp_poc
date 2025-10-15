"""
Example script demonstrating volatility calculations
Simple standalone example showing volatility estimators
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Simulated price data generation
def generate_sample_data(n_days=252, initial_price=100, volatility=0.20):
    """
    Generate sample OHLC data with realistic characteristics
    
    Args:
        n_days: Number of trading days
        initial_price: Starting price
        volatility: Annual volatility
        
    Returns:
        DataFrame with OHLC data
    """
    np.random.seed(42)
    
    # Generate daily returns
    daily_vol = volatility / np.sqrt(252)
    returns = np.random.normal(0.0005, daily_vol, n_days)
    
    # Create price series
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Create OHLC data with realistic intraday range
    df = pd.DataFrame(index=pd.date_range('2023-01-01', periods=n_days, freq='D'))
    df['close'] = prices
    
    # Open is previous close with small gap
    df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.001, n_days))
    df['open'].iloc[0] = initial_price
    
    # High/Low based on close with realistic range
    intraday_range = np.abs(np.random.normal(0.01, 0.005, n_days))
    df['high'] = df['close'] * (1 + intraday_range)
    df['low'] = df['close'] * (1 - intraday_range)
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    df['volume'] = np.random.randint(1000000, 5000000, n_days)
    
    return df


# Volatility calculation functions
def close_to_close_vol(prices, window=20):
    """Calculate close-to-close volatility"""
    returns = np.log(prices / prices.shift(1))
    return returns.rolling(window).std() * np.sqrt(252)


def parkinson_vol(high, low, window=20):
    """Calculate Parkinson volatility"""
    hl = np.log(high / low)
    hl2 = hl ** 2
    return np.sqrt(hl2.rolling(window).sum() / (4 * window * np.log(2))) * np.sqrt(252)


def garman_klass_vol(open_, high, low, close, window=20):
    """Calculate Garman-Klass volatility"""
    hl = np.log(high / low)
    co = np.log(close / open_)
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    return np.sqrt(gk.rolling(window).mean()) * np.sqrt(252)


def main():
    """Main demo function"""
    print("=" * 80)
    print("QVP - Volatility Estimator Demo")
    print("=" * 80)
    
    # Generate sample data
    print("\n[1] Generating sample price data...")
    df = generate_sample_data(n_days=252, volatility=0.25)
    print(f"Generated {len(df)} days of OHLC data")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Calculate volatility estimates
    print("\n[2] Calculating volatility estimates...")
    
    vol_close = close_to_close_vol(df['close'])
    vol_parkinson = parkinson_vol(df['high'], df['low'])
    vol_gk = garman_klass_vol(df['open'], df['high'], df['low'], df['close'])
    
    # Create results DataFrame
    vol_df = pd.DataFrame({
        'Close-to-Close': vol_close,
        'Parkinson': vol_parkinson,
        'Garman-Klass': vol_gk
    })
    
    # Print statistics
    print("\n[3] Volatility Statistics (annualized):")
    print("-" * 80)
    print(f"{'Estimator':<20} {'Mean':<12} {'Std Dev':<12} {'Current':<12}")
    print("-" * 80)
    
    for col in vol_df.columns:
        mean_vol = vol_df[col].mean()
        std_vol = vol_df[col].std()
        current_vol = vol_df[col].iloc[-1]
        print(f"{col:<20} {mean_vol:<12.2%} {std_vol:<12.2%} {current_vol:<12.2%}")
    
    print("-" * 80)
    
    # Compare efficiency
    print("\n[4] Estimator Characteristics:")
    print("-" * 80)
    print(f"Close-to-Close:  Baseline estimator (1.0x efficiency)")
    print(f"Parkinson:       Uses High/Low (5.0x more efficient)")
    print(f"Garman-Klass:    Uses OHLC (7.7x more efficient)")
    print("-" * 80)
    
    # Plot results
    print("\n[5] Generating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Price chart
    ax1.plot(df.index, df['close'], label='Close Price', color='blue', linewidth=1)
    ax1.fill_between(df.index, df['low'], df['high'], alpha=0.2, color='gray', label='Daily Range')
    ax1.set_title('Price Series', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volatility chart
    vol_df.plot(ax=ax2, linewidth=1.5)
    ax2.set_title('Volatility Estimates (20-day rolling)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Annualized Volatility')
    ax2.set_xlabel('Date')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'data/volatility_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Show plot (comment out if running headless)
    # plt.show()
    
    print("\n✅ Demo completed successfully!")
    print("\nKey Insights:")
    print("  • Parkinson and GK estimators are smoother (less noisy)")
    print("  • All estimators track similar trends")
    print("  • Higher-efficiency estimators converge faster")
    print("\nNext Steps:")
    print("  • Try with real market data using qvp.data.DataIngester")
    print("  • Explore GARCH models for volatility forecasting")
    print("  • Run full backtest with scripts/run_demo.py")


if __name__ == '__main__':
    main()
