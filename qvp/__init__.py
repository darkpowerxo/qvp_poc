"""
QVP - Quantitative Volatility Platform
Main package initialization
"""

__version__ = "0.1.0"
__author__ = "Sam Abtahi"

from loguru import logger
import sys

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
)

# Export main components
__all__ = [
    "__version__",
    "__author__",
]
