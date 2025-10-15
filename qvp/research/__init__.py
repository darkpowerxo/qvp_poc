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
]
