"""
Fourier Series Analysis Demo

Demonstrates advanced spectral analysis capabilities for volatility research.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from qvp.data.ingestion import DataIngester
from qvp.research.volatility import VolatilityEstimator
from qvp.research.fourier import (
    FourierSeriesAnalyzer,
    VolatilityFourierAnalyzer,
    fourier_smooth,
    extract_cycles,
    compare_frequency_domains
)


def main():
    """Run comprehensive Fourier analysis demonstration."""
    logger.info("="*80)
    logger.info("FOURIER SERIES ANALYSIS FOR VOLATILITY RESEARCH")
    logger.info("="*80)
    
    # 1. Load data
    logger.info("\n1. Loading market data...")
    ingester = DataIngester()
    spy_data = ingester.download_equity_data(['SPY'], start_date='2020-01-01', end_date='2024-12-31')
    vix_data = ingester.download_vix_data(start_date='2020-01-01', end_date='2024-12-31')
    
    logger.info(f"   Loaded {len(spy_data)} days of SPY data")
    logger.info(f"   Loaded {len(vix_data)} days of VIX data")
    
    # 2. Calculate realized volatility
    logger.info("\n2. Calculating realized volatility...")
    vol_estimator = VolatilityEstimator()
    
    # Use close-to-close volatility if we have SPY data, otherwise use VIX as proxy
    if len(spy_data) > 20 and 'close' in spy_data.columns:
        realized_vol = vol_estimator.close_to_close(spy_data['close'], window=20) * np.sqrt(252)
        realized_vol = realized_vol.dropna()
    else:
        # Use VIX as realized volatility proxy if SPY download failed
        logger.info("   Using VIX as volatility proxy (SPY data incomplete)")
        realized_vol = vix_data['close'] / 100  # VIX is in percentage points
        realized_vol.name = 'volatility'
    
    logger.info(f"   Mean volatility: {realized_vol.mean():.2%}")
    logger.info(f"   Vol of vol: {realized_vol.std():.2%}")
    
    # 3. Basic Fourier Analysis
    logger.info("\n3. Performing Fourier decomposition...")
    fourier = FourierSeriesAnalyzer(n_harmonics=15)
    fourier.fit(realized_vol)
    
    components_df = fourier.get_summary()
    logger.info(f"\n   Top 5 Fourier Components:")
    logger.info(f"\n{components_df.head().to_string(index=False)}")
    
    # 4. Dominant Frequencies
    logger.info("\n4. Detecting dominant cycles...")
    dominant_freqs = fourier.get_dominant_frequencies(realized_vol, n_peaks=5)
    logger.info(f"\n   Dominant Cycles:")
    logger.info(f"\n{dominant_freqs.to_string(index=False)}")
    
    # 5. Volatility-Specific Analysis
    logger.info("\n5. Advanced volatility cycle analysis...")
    vol_fourier = VolatilityFourierAnalyzer(n_harmonics=20)
    analysis = vol_fourier.analyze_volatility_cycles(realized_vol)
    
    logger.info(f"\n   Spectral Entropy: {analysis['spectral_entropy']:.4f}")
    logger.info(f"   (Lower = more periodic, Higher = more random)")
    
    # 6. Seasonal Decomposition
    logger.info("\n6. Seasonal decomposition...")
    decomposition = analysis['decomposition']
    for key in decomposition.keys():
        if key.startswith('seasonal'):
            period = key.split('_')[1]
            logger.info(f"   {period} component std: {decomposition[key].std():.4f}")
    
    # 7. Forecasting
    logger.info("\n7. Forecasting volatility (30 days)...")
    forecast = vol_fourier.forecast_volatility(realized_vol, n_steps=30)
    logger.info(f"   Forecast mean: {forecast.mean():.2%}")
    logger.info(f"   Forecast range: {forecast.min():.2%} - {forecast.max():.2%}")
    
    # 8. Smoothing
    logger.info("\n8. Fourier smoothing...")
    smoothed_vol = fourier_smooth(realized_vol, n_harmonics=10)
    smoothing_error = np.mean(np.abs(realized_vol - smoothed_vol))
    logger.info(f"   Mean absolute error: {smoothing_error:.4f}")
    
    # 9. Cycle Extraction
    logger.info("\n9. Extracting specific cycles...")
    monthly_cycle = extract_cycles(realized_vol, period=21)
    quarterly_cycle = extract_cycles(realized_vol, period=63)
    logger.info(f"   Monthly cycle amplitude: {monthly_cycle.std():.4f}")
    logger.info(f"   Quarterly cycle amplitude: {quarterly_cycle.std():.4f}")
    
    # 10. Coherence Analysis
    logger.info("\n10. Analyzing VIX vs Realized Vol coherence...")
    vix_aligned = vix_data['close'].reindex(realized_vol.index, method='ffill')
    coherence_results = compare_frequency_domains(realized_vol, vix_aligned)
    logger.info(f"   Mean coherence: {coherence_results['mean_coherence']:.3f}")
    logger.info(f"   Max coherence: {coherence_results['max_coherence']:.3f}")
    logger.info(f"   Max coherence frequency: {coherence_results['max_coherence_freq']:.4f}")
    
    # 11. Visualizations
    logger.info("\n11. Creating visualizations...")
    create_visualizations(
        realized_vol,
        fourier,
        analysis,
        forecast,
        smoothed_vol,
        decomposition,
        coherence_results
    )
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info("\nKey Insights:")
    logger.info(f"  • {len(components_df)} significant harmonic components detected")
    logger.info(f"  • Strongest cycle period: {dominant_freqs.iloc[0]['period_days']:.1f} days")
    logger.info(f"  • Spectral complexity: {analysis['spectral_entropy']:.2f}")
    logger.info(f"  • VIX-RealizedVol coherence: {coherence_results['mean_coherence']:.1%}")
    logger.info("\nVisualizations saved to: fourier_analysis.html")


def create_visualizations(
    realized_vol,
    fourier,
    analysis,
    forecast,
    smoothed_vol,
    decomposition,
    coherence_results
):
    """Create comprehensive visualization dashboard."""
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Realized Volatility & Fourier Reconstruction',
            'Power Spectral Density',
            'Dominant Frequency Components',
            'Seasonal Decomposition',
            'Volatility Forecast',
            'Smoothed vs Original',
            'Cycle Extraction (Monthly)',
            'VIX-RealizedVol Coherence'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.12
    )
    
    # 1. Original vs Reconstruction
    reconstruction = fourier.reconstruct(n_points=len(realized_vol))
    fig.add_trace(
        go.Scatter(x=realized_vol.index, y=realized_vol, name='Realized Vol', 
                   line=dict(color='blue', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=realized_vol.index, y=reconstruction, name='Fourier Reconstruction',
                   line=dict(color='red', width=2, dash='dash')),
        row=1, col=1
    )
    
    # 2. Power Spectrum
    frequencies, psd = analysis['power_spectrum']['frequencies'], analysis['power_spectrum']['power']
    fig.add_trace(
        go.Scatter(x=frequencies, y=psd, name='Power Spectrum',
                   line=dict(color='purple', width=2), fill='tozeroy'),
        row=1, col=2
    )
    
    # 3. Dominant Components
    components = analysis['components'].head(10)
    fig.add_trace(
        go.Bar(x=components['component'], y=components['amplitude'],
               name='Component Amplitude', marker_color='orange'),
        row=2, col=1
    )
    
    # 4. Seasonal Decomposition
    if 'seasonal_21d' in decomposition:
        fig.add_trace(
            go.Scatter(x=decomposition['seasonal_21d'].index,
                      y=decomposition['seasonal_21d'],
                      name='Monthly Seasonal', line=dict(color='green', width=1)),
            row=2, col=2
        )
    if 'seasonal_252d' in decomposition:
        fig.add_trace(
            go.Scatter(x=decomposition['seasonal_252d'].index,
                      y=decomposition['seasonal_252d'],
                      name='Yearly Seasonal', line=dict(color='red', width=1)),
            row=2, col=2
        )
    
    # 5. Forecast
    fig.add_trace(
        go.Scatter(x=realized_vol.index[-60:], y=realized_vol.values[-60:],
                   name='Historical', line=dict(color='blue', width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=forecast.index, y=forecast.values,
                   name='Forecast', line=dict(color='red', width=2, dash='dash')),
        row=3, col=1
    )
    
    # 6. Smoothed vs Original
    fig.add_trace(
        go.Scatter(x=realized_vol.index, y=realized_vol, name='Original',
                   line=dict(color='lightblue', width=1), opacity=0.5),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=smoothed_vol.index, y=smoothed_vol, name='Smoothed',
                   line=dict(color='darkblue', width=2)),
        row=3, col=2
    )
    
    # 7. Monthly Cycle
    monthly_cycle = extract_cycles(realized_vol, period=21)
    fig.add_trace(
        go.Scatter(x=monthly_cycle.index, y=monthly_cycle, name='Monthly Cycle',
                   line=dict(color='green', width=2)),
        row=4, col=1
    )
    
    # 8. Coherence
    fig.add_trace(
        go.Scatter(x=coherence_results['frequencies'], y=coherence_results['coherence'],
                   name='Coherence', line=dict(color='purple', width=2), fill='tozeroy'),
        row=4, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_xaxes(title_text="Component #", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=2)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=4, col=2)
    
    fig.update_yaxes(title_text="Volatility", row=1, col=1)
    fig.update_yaxes(title_text="Power", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_yaxes(title_text="Seasonal Component", row=2, col=2)
    fig.update_yaxes(title_text="Volatility", row=3, col=1)
    fig.update_yaxes(title_text="Volatility", row=3, col=2)
    fig.update_yaxes(title_text="Cycle Component", row=4, col=1)
    fig.update_yaxes(title_text="Coherence", row=4, col=2)
    
    fig.update_layout(
        title='Fourier Series Analysis - Volatility Research',
        height=1600,
        showlegend=True,
        template='plotly_white'
    )
    
    # Save
    fig.write_html('fourier_analysis.html')
    logger.info("   Visualization saved to fourier_analysis.html")


if __name__ == "__main__":
    main()
