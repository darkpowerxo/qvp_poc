"""
Strategy module initialization
"""

from qvp.strategies.volatility_strategies import (
    VIXMeanReversionStrategy,
    VolatilityRiskPremiumStrategy,
    SimpleVolatilityStrategy
)

__all__ = [
    'VIXMeanReversionStrategy',
    'VolatilityRiskPremiumStrategy',
    'SimpleVolatilityStrategy',
]
