"""
Configuration management for QVP platform
Handles loading and validation of configuration from YAML and environment variables
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv
from loguru import logger


class Config:
    """
    Configuration manager for QVP platform.
    
    Loads configuration from:
    1. config/config.yaml (default settings)
    2. .env file (environment-specific overrides)
    3. Environment variables (highest priority)
    """
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        """Singleton pattern to ensure single config instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration if not already loaded"""
        if not self._config:
            self.load_config()
    
    def load_config(self, config_path: Optional[Path] = None) -> None:
        """
        Load configuration from YAML file and environment variables
        
        Args:
            config_path: Path to config YAML file. Defaults to config/config.yaml
        """
        # Load .env file
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment from {env_path}")
        
        # Load YAML config
        if config_path is None:
            config_path = Path("config/config.yaml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            self._config = self._get_default_config()
        
        # Override with environment variables
        self._apply_env_overrides()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if no config file exists"""
        return {
            'data': {
                'symbols': {'equity': ['SPY'], 'volatility': ['^VIX']},
                'date_range': {'start': '2021-01-01', 'end': '2024-12-31'},
                'storage': {'format': 'parquet', 'path': './data'},
            },
            'backtest': {
                'initial_capital': 1000000,
                'transaction_costs': {'commission_pct': 0.0005, 'slippage_bps': 2.0},
            },
            'logging': {'level': 'INFO'},
        }
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration"""
        # Data directory
        if data_dir := os.getenv('DATA_DIR'):
            self._config.setdefault('data', {})['storage'] = \
                self._config['data'].get('storage', {})
            self._config['data']['storage']['path'] = data_dir
        
        # Symbols
        if symbols := os.getenv('SYMBOLS'):
            self._config.setdefault('data', {})['symbols'] = \
                self._config['data'].get('symbols', {})
            self._config['data']['symbols']['equity'] = symbols.split(',')
        
        # Initial capital
        if capital := os.getenv('INITIAL_CAPITAL'):
            self._config.setdefault('backtest', {})['initial_capital'] = float(capital)
        
        # Logging level
        if log_level := os.getenv('LOG_LEVEL'):
            self._config.setdefault('logging', {})['level'] = log_level
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key
        
        Args:
            key: Configuration key in dot notation (e.g., 'data.symbols.equity')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Examples:
            >>> config = Config()
            >>> config.get('backtest.initial_capital')
            1000000
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-notation key
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        
        config[keys[-1]] = value
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path"""
        return Path(self.get('data.storage.path', './data'))
    
    @property
    def initial_capital(self) -> float:
        """Get initial capital for backtesting"""
        return float(self.get('backtest.initial_capital', 1000000))
    
    @property
    def symbols(self) -> list:
        """Get list of equity symbols"""
        return self.get('data.symbols.equity', ['SPY'])
    
    @property
    def vix_symbols(self) -> list:
        """Get list of volatility index symbols"""
        return self.get('data.symbols.volatility', ['^VIX'])
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary"""
        return self._config.copy()


# Global config instance
config = Config()
