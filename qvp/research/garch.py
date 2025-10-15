"""
GARCH and time series volatility models
"""

from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from arch import arch_model
from loguru import logger


class GARCHModeler:
    """
    GARCH family models for volatility forecasting
    Supports GARCH, EGARCH, and GJR-GARCH specifications
    """
    
    def __init__(
        self,
        model_type: str = 'GARCH',
        p: int = 1,
        q: int = 1,
        distribution: str = 'normal'
    ):
        """
        Initialize GARCH modeler
        
        Args:
            model_type: 'GARCH', 'EGARCH', or 'GJR-GARCH'
            p: GARCH order
            q: ARCH order
            distribution: Error distribution ('normal', 't', 'skewt')
        """
        self.model_type = model_type
        self.p = p
        self.q = q
        self.distribution = distribution
        self.fitted_model = None
        
        logger.info(
            f"Initialized {model_type}({p},{q}) with {distribution} distribution"
        )
    
    def fit(
        self,
        returns: pd.Series,
        rescale: bool = True
    ) -> 'GARCHModeler':
        """
        Fit GARCH model to returns data
        
        Args:
            returns: Return series (should be percentage returns, not log returns)
            rescale: Whether to rescale returns to percentage (multiply by 100)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.model_type} model to {len(returns)} observations")
        
        # Remove NaN values
        returns_clean = returns.dropna()
        
        # Rescale if needed (ARCH library works better with percentage returns)
        if rescale:
            returns_clean = returns_clean * 100
        
        # Map model types to arch library parameters
        vol_map = {
            'GARCH': 'Garch',
            'EGARCH': 'EGARCH',
            'GJR-GARCH': 'GARCH'
        }
        
        power_map = {
            'GARCH': 2.0,
            'EGARCH': 1.0,
            'GJR-GARCH': 2.0
        }
        
        # Create and fit model
        try:
            model = arch_model(
                returns_clean,
                vol=vol_map[self.model_type],
                p=self.p,
                q=self.q,
                power=power_map[self.model_type],
                dist=self.distribution
            )
            
            self.fitted_model = model.fit(disp='off', show_warning=False)
            
            logger.info(f"Model fitted successfully. AIC: {self.fitted_model.aic:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to fit GARCH model: {e}")
            raise
        
        return self
    
    def forecast(
        self,
        horizon: int = 1,
        method: str = 'analytic'
    ) -> pd.DataFrame:
        """
        Generate volatility forecasts
        
        Args:
            horizon: Forecast horizon (number of steps ahead)
            method: Forecast method ('analytic', 'simulation', or 'bootstrap')
            
        Returns:
            DataFrame with forecasted variance and volatility
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        logger.info(f"Generating {horizon}-step ahead forecast")
        
        forecast = self.fitted_model.forecast(horizon=horizon, method=method)
        
        # Extract forecasted variance
        variance_forecast = forecast.variance.iloc[-1]
        
        result = pd.DataFrame({
            'variance': variance_forecast,
            'volatility': np.sqrt(variance_forecast)
        })
        
        return result
    
    def get_conditional_volatility(self) -> pd.Series:
        """
        Get fitted conditional volatility series
        
        Returns:
            Series of conditional volatility
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        # Return annualized volatility
        # ARCH library returns are in percentage, so divide by 100 and annualize
        cond_vol = self.fitted_model.conditional_volatility / 100
        cond_vol_annual = cond_vol * np.sqrt(252)
        
        return cond_vol_annual
    
    def get_model_summary(self) -> str:
        """Get model summary statistics"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return str(self.fitted_model.summary())
    
    def get_information_criteria(self) -> Dict[str, float]:
        """
        Get model information criteria
        
        Returns:
            Dictionary with AIC, BIC, and log-likelihood
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'loglikelihood': self.fitted_model.loglikelihood
        }


def compare_garch_models(
    returns: pd.Series,
    models: Optional[list] = None
) -> pd.DataFrame:
    """
    Compare different GARCH model specifications
    
    Args:
        returns: Return series
        models: List of (model_type, p, q) tuples. If None, uses default set.
        
    Returns:
        DataFrame comparing model fit statistics
    """
    if models is None:
        models = [
            ('GARCH', 1, 1),
            ('GARCH', 1, 2),
            ('GARCH', 2, 1),
            ('EGARCH', 1, 1),
            ('GJR-GARCH', 1, 1),
        ]
    
    logger.info(f"Comparing {len(models)} GARCH specifications")
    
    results = []
    
    for model_type, p, q in models:
        try:
            modeler = GARCHModeler(model_type=model_type, p=p, q=q)
            modeler.fit(returns)
            
            ic = modeler.get_information_criteria()
            
            results.append({
                'model': f"{model_type}({p},{q})",
                'aic': ic['aic'],
                'bic': ic['bic'],
                'loglikelihood': ic['loglikelihood']
            })
            
        except Exception as e:
            logger.warning(f"Failed to fit {model_type}({p},{q}): {e}")
            continue
    
    df = pd.DataFrame(results)
    df = df.sort_values('aic')  # Lower AIC is better
    
    logger.info(f"Best model by AIC: {df.iloc[0]['model']}")
    
    return df


class VolatilityForecaster:
    """
    Ensemble volatility forecasting combining multiple methods
    """
    
    def __init__(self):
        """Initialize forecaster"""
        self.models = {}
        self.weights = {}
    
    def add_model(
        self,
        name: str,
        model: GARCHModeler,
        weight: float = 1.0
    ) -> 'VolatilityForecaster':
        """
        Add a model to the ensemble
        
        Args:
            name: Model identifier
            model: Fitted GARCH model
            weight: Weight in ensemble (will be normalized)
            
        Returns:
            Self for method chaining
        """
        self.models[name] = model
        self.weights[name] = weight
        return self
    
    def forecast(self, horizon: int = 1) -> pd.DataFrame:
        """
        Generate ensemble forecast
        
        Args:
            horizon: Forecast horizon
            
        Returns:
            DataFrame with weighted average forecast
        """
        if not self.models:
            raise ValueError("No models added to ensemble")
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        normalized_weights = {k: v / total_weight for k, v in self.weights.items()}
        
        forecasts = {}
        
        for name, model in self.models.items():
            fc = model.forecast(horizon=horizon)
            forecasts[name] = fc
        
        # Weighted average
        ensemble_variance = sum(
            forecasts[name]['variance'] * normalized_weights[name]
            for name in self.models
        )
        
        result = pd.DataFrame({
            'variance': ensemble_variance,
            'volatility': np.sqrt(ensemble_variance)
        })
        
        return result
