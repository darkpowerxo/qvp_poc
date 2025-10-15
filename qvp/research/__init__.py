"""
Research module initialization
"""

from qvp.research.volatility import VolatilityEstimator, ImpliedVolatility
from qvp.research.garch import GARCHModeler, compare_garch_models
from qvp.research.features import (
    TechnicalIndicators,
    VolatilityFeatures,
    RollingStatistics,
    RegimeDetection,
    MLFeatureEngine
)
from qvp.research.fourier import (
    FourierSeriesAnalyzer,
    VolatilityFourierAnalyzer,
    FourierComponent,
    fourier_smooth,
    extract_cycles,
    compare_frequency_domains
)

__all__ = [
    'VolatilityEstimator',
    'ImpliedVolatility',
    'GARCHModeler',
    'compare_garch_models',
    'TechnicalIndicators',
    'VolatilityFeatures',
    'RollingStatistics',
    'RegimeDetection',
    'MLFeatureEngine',
    'FourierSeriesAnalyzer',
    'VolatilityFourierAnalyzer',
    'FourierComponent',
    'fourier_smooth',
    'extract_cycles',
    'compare_frequency_domains',
]
