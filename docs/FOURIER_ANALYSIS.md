# Fourier Series Analysis for Volatility Research

## Overview

The QVP platform now includes a comprehensive Fourier analysis module for advanced volatility research. This implementation provides state-of-the-art spectral analysis capabilities for understanding cyclical patterns, seasonal effects, and frequency-domain characteristics of volatility time series.

## Features

### 1. FourierSeriesAnalyzer Class

Core Fourier analysis functionality:

```python
from qvp.research import FourierSeriesAnalyzer

# Initialize with desired harmonics
analyzer = FourierSeriesAnalyzer(n_harmonics=15)

# Fit to time series
analyzer.fit(volatility_series)

# Get Fourier components
components = analyzer.components
```

**Key Methods:**

- `fit(series)` - Decompose time series into harmonic components
- `reconstruct(t)` - Reconstruct signal from harmonics
- `forecast(steps)` - Extrapolate using harmonic components
- `get_power_spectrum()` - Compute Power Spectral Density
- `denoise(threshold)` - Remove noise via frequency filtering
- `seasonal_decompose(periods)` - Extract seasonal cycles
- `filter_frequency_band(low, high)` - Bandpass filtering
- `compute_coherence(other_series)` - Cross-spectral coherence

### 2. VolatilityFourierAnalyzer Class

Volatility-specific analysis wrapper:

```python
from qvp.research import VolatilityFourierAnalyzer

# Initialize with configuration
vol_analyzer = VolatilityFourierAnalyzer(
    n_harmonics=20,
    smoothing_window=5
)

# Comprehensive analysis
results = vol_analyzer.analyze_volatility_cycles(volatility_data)

# Forecast future volatility
forecast = vol_analyzer.forecast_volatility(
    volatility_data,
    forecast_days=30
)
```

**Analysis Results Include:**

- Dominant frequency components
- Spectral entropy (measure of randomness)
- Power spectral density
- Seasonal decomposition
- Cycle extraction
- Regime change detection

### 3. Utility Functions

Convenient helper functions:

```python
from qvp.research import (
    fourier_smooth,
    extract_cycles,
    compare_frequency_domains
)

# Smooth volatility using Fourier filtering
smoothed = fourier_smooth(volatility, n_harmonics=10)

# Extract specific cyclical components
monthly_cycle = extract_cycles(volatility, periods=[21])

# Compare two series in frequency domain
coherence = compare_frequency_domains(vix, realized_vol)
```

## Mathematical Background

### Fourier Series Decomposition

Any periodic signal can be represented as a sum of sinusoids:

$$
f(t) = a_0 + \sum_{n=1}^{N} [a_n \cos(2\pi f_n t) + b_n \sin(2\pi f_n t)]
$$

Where:
- $a_0$ = DC component (mean)
- $a_n, b_n$ = Fourier coefficients
- $f_n$ = harmonic frequencies
- $N$ = number of harmonics

### Power Spectral Density

Measures energy distribution across frequencies:

$$
PSD(f) = \lim_{T \to \infty} \frac{1}{T} |X(f)|^2
$$

Computed using Welch's method for robust estimation.

### Spectral Entropy

Quantifies frequency domain complexity:

$$
H = -\sum_{i} p_i \log_2(p_i)
$$

Where $p_i$ is normalized power at frequency $i$.

- Lower entropy → More periodic/predictable
- Higher entropy → More random/unpredictable

## Usage Examples

### Basic FFT Decomposition

```python
from qvp.research import FourierSeriesAnalyzer
import numpy as np

# Create analyzer
fourier = FourierSeriesAnalyzer(n_harmonics=15)

# Fit to volatility data
fourier.fit(volatility_series)

# Examine components
print(f"Number of components: {len(fourier.components)}")
for comp in fourier.components[:5]:
    print(f"Period: {comp.period:.1f} days, Amplitude: {comp.amplitude:.4f}")

# Reconstruct signal
reconstructed = fourier.reconstruct(np.arange(len(volatility_series)))

# Calculate reconstruction error
error = np.mean(np.abs(volatility_series - reconstructed))
print(f"Mean absolute error: {error:.4f}")
```

### Volatility Forecasting

```python
from qvp.research import VolatilityFourierAnalyzer

# Initialize analyzer
vol_analyzer = VolatilityFourierAnalyzer(n_harmonics=20)

# Forecast 30 days ahead
forecast = vol_analyzer.forecast_volatility(
    historical_volatility,
    forecast_days=30,
    confidence_level=0.95
)

# Access forecast values
print(f"Mean forecast: {forecast['mean'].mean():.2%}")
print(f"Forecast range: {forecast['lower_bound'].min():.2%} - {forecast['upper_bound'].max():.2%}")
```

### Cycle Detection

```python
from qvp.research import extract_cycles

# Extract weekly cycle (5 trading days)
weekly = extract_cycles(volatility, periods=[5], bandwidth=0.5)

# Extract monthly cycle (21 trading days)
monthly = extract_cycles(volatility, periods=[21], bandwidth=0.5)

# Extract quarterly cycle (63 trading days)
quarterly = extract_cycles(volatility, periods=[63], bandwidth=0.5)

print(f"Weekly cycle strength: {weekly.std():.4f}")
print(f"Monthly cycle strength: {monthly.std():.4f}")
print(f"Quarterly cycle strength: {quarterly.std():.4f}")
```

### Coherence Analysis

```python
from qvp.research import compare_frequency_domains

# Compare VIX and realized volatility
coherence_results = compare_frequency_domains(
    vix_series,
    realized_vol_series
)

print(f"Mean coherence: {coherence_results['mean_coherence']:.3f}")
print(f"Max coherence: {coherence_results['max_coherence']:.3f}")
print(f"Frequency of max coherence: {coherence_results['max_coherence_freq']:.4f}")

# High coherence (>0.8) indicates strong relationship at that frequency
```

### Denoising

```python
from qvp.research import FourierSeriesAnalyzer

# Create analyzer
fourier = FourierSeriesAnalyzer(n_harmonics=50)
fourier.fit(noisy_volatility)

# Denoise by removing low-power components
# Keep only components with >5% of max power
denoised = fourier.denoise(threshold=0.05)

# Calculate noise reduction
noise_removed = np.std(noisy_volatility - denoised)
print(f"Noise level removed: {noise_removed:.4f}")
```

### Seasonal Decomposition

```python
from qvp.research import FourierSeriesAnalyzer

# Decompose into seasonal components
fourier = FourierSeriesAnalyzer(n_harmonics=100)
fourier.fit(volatility)

seasonal = fourier.seasonal_decompose(
    periods=[5, 21, 63, 252]  # weekly, monthly, quarterly, annual
)

# Access individual components
weekly = seasonal[5]
monthly = seasonal[21]
quarterly = seasonal[63]
annual = seasonal[252]

print(f"Weekly seasonality: {weekly.std():.4f}")
print(f"Monthly seasonality: {monthly.std():.4f}")
```

## Complete Analysis Pipeline

Run the comprehensive demonstration:

```bash
uv run python scripts/example_fourier.py
```

This executes an 11-step analysis pipeline:

1. **Load market data** - SPY and VIX historical data
2. **Calculate realized volatility** - Yang-Zhang estimator
3. **Fourier decomposition** - Extract top 15 harmonics
4. **Detect dominant cycles** - Identify strongest periodic components
5. **Advanced cycle analysis** - Spectral entropy and regime detection
6. **Seasonal decomposition** - Weekly, monthly, quarterly, annual cycles
7. **Volatility forecasting** - 30-day ahead prediction
8. **Fourier smoothing** - Denoise via frequency filtering
9. **Extract specific cycles** - Monthly and quarterly components
10. **Coherence analysis** - VIX vs realized volatility relationship
11. **Visualizations** - 8-panel interactive dashboard

**Output:** `fourier_analysis.html` - Interactive Plotly dashboard with:
- Time series plots
- Frequency spectrum
- Reconstruction comparison
- Seasonal decomposition
- Forecast visualization
- Coherence heatmap
- Power spectral density
- Cycle extraction

## Interpretation Guide

### Spectral Entropy

- **Low (< 3.0)**: Highly periodic, predictable patterns
- **Medium (3.0-4.5)**: Mixed periodic and stochastic behavior
- **High (> 4.5)**: Mostly random, unpredictable

### Dominant Frequencies

- **High frequency (short periods)**: Intraday/daily patterns
- **Medium frequency (5-21 days)**: Weekly/monthly cycles
- **Low frequency (>63 days)**: Quarterly/seasonal effects

### Reconstruction Error

- **Low (<0.01)**: Excellent fit, strong periodic structure
- **Medium (0.01-0.05)**: Good fit, some noise present
- **High (>0.05)**: Poor fit, mostly stochastic behavior

### Coherence

- **>0.8**: Strong frequency-domain relationship
- **0.5-0.8**: Moderate relationship
- **<0.5**: Weak/no relationship at that frequency

## Integration with QVP

The Fourier analysis module integrates seamlessly with existing QVP components:

### With Volatility Estimators

```python
from qvp.research import VolatilityEstimator, FourierSeriesAnalyzer

# Calculate realized volatility
vol_est = VolatilityEstimator()
realized_vol = vol_est.yang_zhang(
    open=data['open'],
    high=data['high'],
    low=data['low'],
    close=data['close'],
    window=20
) * np.sqrt(252)

# Analyze frequency components
fourier = FourierSeriesAnalyzer(n_harmonics=15)
fourier.fit(realized_vol)
```

### With GARCH Models

```python
from qvp.research import GARCHModeler, VolatilityFourierAnalyzer

# Fit GARCH model
garch = GARCHModeler()
garch.fit(returns, model_type='GJR-GARCH')

# Extract residuals for cycle analysis
residuals = garch.get_standardized_residuals()

# Analyze cyclical patterns in residuals
vol_fourier = VolatilityFourierAnalyzer()
cycle_analysis = vol_fourier.analyze_volatility_cycles(residuals)
```

### With ML Features

```python
from qvp.research import MLFeatureEngine, FourierSeriesAnalyzer

# Generate ML features
feature_engine = MLFeatureEngine()
features = feature_engine.generate_features(data)

# Add Fourier features
fourier = FourierSeriesAnalyzer(n_harmonics=10)
fourier.fit(features['volatility'])

# Add dominant cycle features
features['dominant_cycle'] = fourier.components[0].period
features['spectral_entropy'] = fourier.compute_spectral_entropy()
```

## Performance Considerations

### Computational Complexity

- **FFT**: O(N log N) - Very efficient for large datasets
- **DFT**: O(N²) - Used only for small datasets
- **Welch's method**: O(N log N) with overlapping segments

### Best Practices

1. **Choose appropriate harmonics**: 
   - Too few → Underfitting
   - Too many → Overfitting, noise amplification
   - Rule of thumb: N/10 to N/5 harmonics

2. **Detrend before FFT**:
   - Removes low-frequency trend
   - Improves harmonic extraction
   - Already handled in `fit()` method

3. **Windowing for PSD**:
   - Welch's method reduces variance
   - Segment length = N/8 is a good default
   - More segments → smoother but lower resolution

4. **Seasonal decomposition**:
   - Use exact frequencies when known (e.g., 252 trading days/year)
   - Bandwidth parameter controls frequency range
   - Wider bandwidth → smoother extraction

## References

### Algorithms Implemented

1. **Fast Fourier Transform (FFT)**
   - Cooley-Tukey algorithm
   - Implementation: `scipy.fft.fft`

2. **Welch's Method**
   - Overlapped segment averaging
   - Implementation: `scipy.signal.welch`

3. **Coherence Analysis**
   - Cross-spectral density estimation
   - Implementation: `scipy.signal.coherence`

4. **Bandpass Filtering**
   - Butterworth filter design
   - Implementation: `scipy.signal.butter`, `scipy.signal.filtfilt`

### Literature

- Cooley, J. W., & Tukey, J. W. (1965). "An algorithm for the machine calculation of complex Fourier series"
- Welch, P. (1967). "The use of fast Fourier transform for the estimation of power spectra"
- Brillinger, D. R. (2001). "Time Series: Data Analysis and Theory"
- Percival, D. B., & Walden, A. T. (1993). "Spectral Analysis for Physical Applications"

## API Reference

### FourierSeriesAnalyzer

```python
class FourierSeriesAnalyzer:
    """
    Complete Fourier series analysis for time series data.
    
    Parameters
    ----------
    n_harmonics : int, default=10
        Number of harmonic components to extract
    detrend : bool, default=True
        Whether to remove linear trend before FFT
    """
    
    def fit(self, series: pd.Series) -> 'FourierSeriesAnalyzer':
        """Fit Fourier components to time series."""
        
    def reconstruct(self, t: np.ndarray) -> np.ndarray:
        """Reconstruct signal from harmonics."""
        
    def forecast(self, steps: int) -> pd.Series:
        """Forecast future values using harmonics."""
        
    def get_power_spectrum(self) -> pd.DataFrame:
        """Compute power spectral density."""
        
    def denoise(self, threshold: float = 0.1) -> pd.Series:
        """Denoise by removing low-power components."""
        
    def seasonal_decompose(self, periods: List[int]) -> Dict[int, pd.Series]:
        """Extract seasonal components at specific periods."""
```

### VolatilityFourierAnalyzer

```python
class VolatilityFourierAnalyzer:
    """
    Volatility-specific Fourier analysis.
    
    Parameters
    ----------
    n_harmonics : int, default=20
        Number of harmonics for decomposition
    smoothing_window : int, default=5
        Window for pre-smoothing volatility
    """
    
    def analyze_volatility_cycles(self, volatility: pd.Series) -> Dict:
        """Comprehensive volatility cycle analysis."""
        
    def forecast_volatility(
        self,
        volatility: pd.Series,
        forecast_days: int = 30,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """Forecast future volatility with confidence bounds."""
        
    def detect_regime_changes(
        self,
        volatility: pd.Series,
        window: int = 63
    ) -> pd.DataFrame:
        """Detect volatility regime changes via spectral analysis."""
```

### Utility Functions

```python
def fourier_smooth(
    series: pd.Series,
    n_harmonics: int = 10
) -> pd.Series:
    """Smooth time series using Fourier filtering."""

def extract_cycles(
    series: pd.Series,
    periods: List[int],
    bandwidth: float = 0.1
) -> pd.Series:
    """Extract specific cyclical components."""

def compare_frequency_domains(
    series1: pd.Series,
    series2: pd.Series
) -> Dict:
    """Compare two series in frequency domain using coherence."""
```

## Troubleshooting

### Common Issues

**Issue**: High reconstruction error
- **Solution**: Increase `n_harmonics` or check for non-stationarity

**Issue**: Noisy power spectrum
- **Solution**: Use Welch's method with appropriate segment length

**Issue**: Spurious seasonal components
- **Solution**: Increase `bandwidth` parameter in `extract_cycles()`

**Issue**: Forecast diverges
- **Solution**: Ensure series is stationary, or use shorter forecast horizon

### Getting Help

For questions or issues:
1. Check this documentation
2. Run `scripts/example_fourier.py` for working examples
3. Review test cases in `tests/test_fourier.py`
4. See visualization output in `fourier_analysis.html`

## Future Enhancements

Planned features for future versions:

- [ ] Wavelet analysis integration
- [ ] Hilbert-Huang Transform
- [ ] Adaptive harmonic selection
- [ ] Real-time streaming FFT
- [ ] Multi-variate spectral analysis
- [ ] Regime-dependent forecasting
- [ ] GPU-accelerated FFT for large datasets
- [ ] Automatic cycle detection via peak finding

---

**Version**: 1.0  
**Last Updated**: January 2025  
**Author**: QVP Development Team
