"""
Fourier Series Analysis for Volatility Research

Advanced spectral analysis using Fourier series decomposition for:
- Cyclical pattern detection in volatility
- Frequency domain analysis
- Seasonal decomposition
- Volatility forecasting using harmonic components
- Spectral density estimation
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from scipy import signal, fft
from scipy.optimize import minimize
from loguru import logger


@dataclass
class FourierComponent:
    """Represents a single Fourier component."""
    frequency: float
    amplitude: float
    phase: float
    period: float
    
    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the component at given time points."""
        return self.amplitude * np.cos(2 * np.pi * self.frequency * t + self.phase)


class FourierSeriesAnalyzer:
    """
    Complete Fourier Series implementation for time series analysis.
    
    Features:
    - Discrete Fourier Transform (DFT) and Fast Fourier Transform (FFT)
    - Power Spectral Density (PSD) estimation
    - Dominant frequency detection
    - Harmonic decomposition
    - Series reconstruction
    - Denoising and smoothing
    """
    
    def __init__(
        self,
        n_harmonics: int = 10,
        sampling_freq: Optional[float] = None
    ):
        """
        Initialize Fourier analyzer.
        
        Args:
            n_harmonics: Number of harmonics to use in decomposition
            sampling_freq: Sampling frequency (defaults to 252 for daily trading data)
        """
        self.n_harmonics = n_harmonics
        self.sampling_freq = sampling_freq or 252.0
        self.components: List[FourierComponent] = []
        self.mean_value: float = 0.0
    
    def fit(self, series: pd.Series, detrend: bool = True) -> 'FourierSeriesAnalyzer':
        """
        Fit Fourier series to time series data.
        
        Args:
            series: Time series data
            detrend: Whether to remove linear trend first
            
        Returns:
            Self for method chaining
        """
        # Convert to numpy array
        data = series.values
        n = len(data)
        
        # Store mean
        self.mean_value = np.mean(data)
        
        # Detrend if requested
        if detrend:
            data = signal.detrend(data, type='linear')
        else:
            data = data - self.mean_value
        
        # Compute FFT
        fft_values = fft.fft(data)
        fft_freq = fft.fftfreq(n, d=1/self.sampling_freq)
        
        # Get positive frequencies only
        positive_freq_idx = fft_freq > 0
        fft_values = fft_values[positive_freq_idx]
        fft_freq = fft_freq[positive_freq_idx]
        
        # Calculate amplitudes and phases
        amplitudes = 2 * np.abs(fft_values) / n
        phases = np.angle(fft_values)
        
        # Sort by amplitude (descending)
        sorted_indices = np.argsort(amplitudes)[::-1]
        
        # Extract top N harmonics
        self.components = []
        for i in range(min(self.n_harmonics, len(sorted_indices))):
            idx = sorted_indices[i]
            component = FourierComponent(
                frequency=fft_freq[idx],
                amplitude=amplitudes[idx],
                phase=phases[idx],
                period=1 / fft_freq[idx] if fft_freq[idx] > 0 else np.inf
            )
            self.components.append(component)
        
        logger.info(f"Fitted {len(self.components)} Fourier components")
        return self
    
    def reconstruct(self, n_points: Optional[int] = None) -> np.ndarray:
        """
        Reconstruct time series from Fourier components.
        
        Args:
            n_points: Number of points to generate (None uses fitted length)
            
        Returns:
            Reconstructed series
        """
        if not self.components:
            raise ValueError("Must fit model before reconstruction")
        
        if n_points is None:
            n_points = 252  # Default to 1 year of trading days
        
        t = np.arange(n_points) / self.sampling_freq
        reconstruction = np.full(n_points, self.mean_value)
        
        for component in self.components:
            reconstruction += component.evaluate(t)
        
        return reconstruction
    
    def forecast(self, n_steps: int) -> np.ndarray:
        """
        Forecast future values using fitted Fourier series.
        
        Args:
            n_steps: Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        return self.reconstruct(n_points=n_steps)
    
    def get_power_spectrum(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate power spectral density.
        
        Args:
            series: Time series data
            
        Returns:
            Tuple of (frequencies, power spectrum)
        """
        data = series.values - np.mean(series.values)
        
        # Use Welch's method for better spectrum estimation
        frequencies, psd = signal.welch(
            data,
            fs=self.sampling_freq,
            nperseg=min(256, len(data)),
            scaling='density'
        )
        
        return frequencies, psd
    
    def get_dominant_frequencies(
        self,
        series: pd.Series,
        n_peaks: int = 5
    ) -> pd.DataFrame:
        """
        Find dominant frequencies in the time series.
        
        Args:
            series: Time series data
            n_peaks: Number of dominant frequencies to return
            
        Returns:
            DataFrame with frequency, period, and power information
        """
        frequencies, psd = self.get_power_spectrum(series)
        
        # Find peaks in power spectrum
        peaks, properties = signal.find_peaks(psd, height=0)
        
        # Sort by height (power)
        sorted_peaks = peaks[np.argsort(properties['peak_heights'])[::-1]]
        
        # Get top N peaks
        top_peaks = sorted_peaks[:n_peaks]
        
        results = []
        for peak in top_peaks:
            freq = frequencies[peak]
            power = psd[peak]
            period = 1 / freq if freq > 0 else np.inf
            
            results.append({
                'frequency': freq,
                'period_days': period,
                'power': power,
                'normalized_power': power / np.max(psd)
            })
        
        return pd.DataFrame(results)
    
    def denoise(
        self,
        series: pd.Series,
        threshold_percentile: float = 90
    ) -> pd.Series:
        """
        Denoise series by removing low-power frequency components.
        
        Args:
            series: Time series data
            threshold_percentile: Keep components above this power percentile
            
        Returns:
            Denoised series
        """
        data = series.values
        n = len(data)
        
        # FFT
        fft_values = fft.fft(data - np.mean(data))
        
        # Calculate power
        power = np.abs(fft_values) ** 2
        
        # Threshold
        threshold = np.percentile(power, threshold_percentile)
        
        # Zero out low-power components
        fft_filtered = fft_values.copy()
        fft_filtered[power < threshold] = 0
        
        # Inverse FFT
        denoised = fft.ifft(fft_filtered).real + np.mean(data)
        
        return pd.Series(denoised, index=series.index)
    
    def seasonal_decompose(
        self,
        series: pd.Series,
        periods: List[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Decompose series into seasonal components using Fourier analysis.
        
        Args:
            series: Time series data
            periods: List of periods to extract (e.g., [5, 21, 252] for weekly, monthly, yearly)
            
        Returns:
            Dictionary with trend, seasonal components, and residual
        """
        if periods is None:
            periods = [5, 21, 63, 252]  # Weekly, monthly, quarterly, yearly
        
        data = series.values
        n = len(data)
        t = np.arange(n)
        
        # Extract trend (mean)
        trend = np.full(n, np.mean(data))
        
        # Extract seasonal components
        seasonal_components = {}
        residual = data - trend
        
        for period in periods:
            if period > n / 2:
                continue
            
            # Target frequency
            target_freq = 1 / period
            
            # FFT
            fft_values = fft.fft(residual)
            fft_freq = fft.fftfreq(n, d=1/self.sampling_freq)
            
            # Create filter for this period
            freq_filter = np.abs(fft_freq - target_freq) < (0.5 / period)
            
            # Extract component
            fft_component = np.zeros_like(fft_values)
            fft_component[freq_filter] = fft_values[freq_filter]
            
            component = fft.ifft(fft_component).real
            seasonal_components[f'seasonal_{period}d'] = pd.Series(component, index=series.index)
            
            # Remove from residual
            residual -= component
        
        return {
            'trend': pd.Series(trend, index=series.index),
            **seasonal_components,
            'residual': pd.Series(residual, index=series.index)
        }
    
    def get_spectral_entropy(self, series: pd.Series) -> float:
        """
        Calculate spectral entropy as measure of complexity.
        
        Lower entropy = more regular/periodic
        Higher entropy = more random/complex
        
        Args:
            series: Time series data
            
        Returns:
            Spectral entropy value
        """
        _, psd = self.get_power_spectrum(series)
        
        # Normalize to probability distribution
        psd_norm = psd / np.sum(psd)
        
        # Calculate entropy
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        return entropy
    
    def filter_frequency_band(
        self,
        series: pd.Series,
        low_freq: float,
        high_freq: float
    ) -> pd.Series:
        """
        Bandpass filter to extract specific frequency range.
        
        Args:
            series: Time series data
            low_freq: Low frequency cutoff
            high_freq: High frequency cutoff
            
        Returns:
            Filtered series
        """
        data = series.values
        n = len(data)
        
        # FFT
        fft_values = fft.fft(data - np.mean(data))
        fft_freq = fft.fftfreq(n, d=1/self.sampling_freq)
        
        # Bandpass filter
        freq_filter = (np.abs(fft_freq) >= low_freq) & (np.abs(fft_freq) <= high_freq)
        fft_filtered = np.zeros_like(fft_values)
        fft_filtered[freq_filter] = fft_values[freq_filter]
        
        # Inverse FFT
        filtered = fft.ifft(fft_filtered).real + np.mean(data)
        
        return pd.Series(filtered, index=series.index)
    
    def compute_coherence(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute coherence between two time series.
        
        Coherence measures correlation in frequency domain.
        
        Args:
            series1: First time series
            series2: Second time series
            
        Returns:
            Tuple of (frequencies, coherence values)
        """
        data1 = series1.values - np.mean(series1.values)
        data2 = series2.values - np.mean(series2.values)
        
        # Ensure same length
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        frequencies, coherence = signal.coherence(
            data1,
            data2,
            fs=self.sampling_freq,
            nperseg=min(256, min_len)
        )
        
        return frequencies, coherence
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of fitted Fourier components.
        
        Returns:
            DataFrame with component information
        """
        if not self.components:
            return pd.DataFrame()
        
        summary = []
        for i, comp in enumerate(self.components):
            summary.append({
                'component': i + 1,
                'frequency': comp.frequency,
                'period_days': comp.period,
                'amplitude': comp.amplitude,
                'phase': comp.phase,
                'phase_degrees': np.degrees(comp.phase)
            })
        
        return pd.DataFrame(summary)


class VolatilityFourierAnalyzer:
    """
    Specialized Fourier analysis for volatility time series.
    
    Includes:
    - Intraday volatility patterns
    - Day-of-week effects
    - Monthly seasonality
    - Volatility clustering detection
    """
    
    def __init__(self, n_harmonics: int = 20):
        """
        Initialize volatility-specific Fourier analyzer.
        
        Args:
            n_harmonics: Number of harmonics to use
        """
        self.fourier = FourierSeriesAnalyzer(n_harmonics=n_harmonics)
    
    def analyze_volatility_cycles(
        self,
        volatility: pd.Series
    ) -> Dict:
        """
        Comprehensive cyclical analysis of volatility.
        
        Args:
            volatility: Volatility time series
            
        Returns:
            Dictionary with various cycle analyses
        """
        logger.info("Analyzing volatility cycles...")
        
        # Fit Fourier series
        self.fourier.fit(volatility)
        
        # Get dominant frequencies
        dominant_freqs = self.fourier.get_dominant_frequencies(volatility, n_peaks=10)
        
        # Seasonal decomposition
        decomposition = self.fourier.seasonal_decompose(
            volatility,
            periods=[5, 21, 63, 252]  # Weekly, monthly, quarterly, yearly
        )
        
        # Spectral entropy
        entropy = self.fourier.get_spectral_entropy(volatility)
        
        # Power spectrum
        frequencies, psd = self.fourier.get_power_spectrum(volatility)
        
        return {
            'dominant_frequencies': dominant_freqs,
            'decomposition': decomposition,
            'spectral_entropy': entropy,
            'power_spectrum': {
                'frequencies': frequencies,
                'power': psd
            },
            'components': self.fourier.get_summary()
        }
    
    def forecast_volatility(
        self,
        volatility: pd.Series,
        n_steps: int
    ) -> pd.Series:
        """
        Forecast volatility using Fourier extrapolation.
        
        Args:
            volatility: Historical volatility
            n_steps: Steps to forecast
            
        Returns:
            Forecasted volatility
        """
        self.fourier.fit(volatility)
        forecast = self.fourier.forecast(n_steps)
        
        # Create date index
        last_date = volatility.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=n_steps,
            freq='D'
        )
        
        return pd.Series(forecast, index=forecast_index)
    
    def detect_regime_changes(
        self,
        volatility: pd.Series,
        window: int = 63
    ) -> pd.Series:
        """
        Detect volatility regime changes using spectral analysis.
        
        Args:
            volatility: Volatility time series
            window: Rolling window size
            
        Returns:
            Series indicating regime change probability
        """
        # Rolling spectral entropy
        entropies = []
        
        for i in range(window, len(volatility)):
            window_data = volatility.iloc[i-window:i]
            entropy = self.fourier.get_spectral_entropy(window_data)
            entropies.append(entropy)
        
        entropy_series = pd.Series(
            entropies,
            index=volatility.index[window:]
        )
        
        # Normalize to [0, 1]
        normalized = (entropy_series - entropy_series.min()) / (entropy_series.max() - entropy_series.min())
        
        return normalized


# Convenience functions
def fourier_smooth(
    series: pd.Series,
    n_harmonics: int = 10
) -> pd.Series:
    """
    Smooth time series using Fourier reconstruction.
    
    Args:
        series: Time series to smooth
        n_harmonics: Number of harmonics to keep
        
    Returns:
        Smoothed series
    """
    analyzer = FourierSeriesAnalyzer(n_harmonics=n_harmonics)
    analyzer.fit(series)
    smoothed = analyzer.reconstruct(n_points=len(series))
    
    return pd.Series(smoothed, index=series.index)


def extract_cycles(
    series: pd.Series,
    period: int
) -> pd.Series:
    """
    Extract specific cyclical component from series.
    
    Args:
        series: Time series
        period: Period to extract (in days)
        
    Returns:
        Cyclical component
    """
    analyzer = FourierSeriesAnalyzer()
    low_freq = 1 / (period * 1.2)
    high_freq = 1 / (period * 0.8)
    
    return analyzer.filter_frequency_band(series, low_freq, high_freq)


def compare_frequency_domains(
    series1: pd.Series,
    series2: pd.Series
) -> Dict:
    """
    Compare two time series in frequency domain.
    
    Args:
        series1: First time series
        series2: Second time series
        
    Returns:
        Dictionary with comparison metrics
    """
    analyzer = FourierSeriesAnalyzer()
    
    frequencies, coherence = analyzer.compute_coherence(series1, series2)
    
    return {
        'frequencies': frequencies,
        'coherence': coherence,
        'mean_coherence': np.mean(coherence),
        'max_coherence': np.max(coherence),
        'max_coherence_freq': frequencies[np.argmax(coherence)]
    }
