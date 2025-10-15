"""
Data module initialization
"""

from qvp.data.ingestion import DataIngester, DataStorage

try:
    from qvp.data.kdb_connector import KDBConnector, load_kdb_config
    KDB_AVAILABLE = True
    __all__ = ['DataIngester', 'DataStorage', 'KDBConnector', 'load_kdb_config']
except ImportError:
    KDB_AVAILABLE = False
    __all__ = ['DataIngester', 'DataStorage']
