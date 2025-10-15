"""
Portfolio optimization using convex optimization
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import cvxpy as cp
from loguru import logger


class PortfolioOptimizer:
    """
    Portfolio optimization using CVXPY
    Supports mean-variance optimization with constraints
    """
    
    def __init__(
        self,
        max_position: float = 0.2,
        max_leverage: float = 2.0,
        min_position: float = 0.01
    ):
        """
        Initialize portfolio optimizer
        
        Args:
            max_position: Maximum position size (fraction)
            max_leverage: Maximum leverage
            min_position: Minimum position size (fraction)
        """
        self.max_position = max_position
        self.max_leverage = max_leverage
        self.min_position = min_position
        
        logger.info(
            f"Initialized optimizer: max_pos={max_position}, "
            f"max_lev={max_leverage}"
        )
    
    def mean_variance_optimization(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        long_only: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Mean-variance portfolio optimization
        
        Maximize: expected_return - risk_aversion * variance
        
        Args:
            expected_returns: Expected returns vector
            cov_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter (higher = more conservative)
            long_only: Whether to constrain to long-only positions
            
        Returns:
            Tuple of (optimal weights, optimization info)
        """
        n_assets = len(expected_returns)
        
        # Decision variables: portfolio weights
        w = cp.Variable(n_assets)
        
        # Objective: maximize return - risk_aversion * variance
        portfolio_return = expected_returns @ w
        portfolio_variance = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
        ]
        
        # Position limits
        if long_only:
            constraints.append(w >= 0)
            constraints.append(w <= self.max_position)
        else:
            constraints.append(w >= -self.max_position)
            constraints.append(w <= self.max_position)
            constraints.append(cp.norm(w, 1) <= self.max_leverage)
        
        # Minimum position (avoid tiny positions)
        # If position > 0, it must be >= min_position
        # This is a non-convex constraint, so we'll handle in post-processing
        
        # Solve problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            logger.warning(f"Optimization status: {problem.status}")
            # Return equal weight as fallback
            return np.ones(n_assets) / n_assets, {'status': problem.status}
        
        weights = w.value
        
        # Post-process: eliminate tiny positions
        weights[np.abs(weights) < self.min_position] = 0
        
        # Renormalize
        if np.sum(np.abs(weights)) > 0:
            weights = weights / np.sum(np.abs(weights))
        
        info = {
            'status': problem.status,
            'expected_return': expected_returns @ weights,
            'volatility': np.sqrt(weights @ cov_matrix @ weights),
            'objective_value': problem.value
        }
        
        logger.info(
            f"Optimization complete: E[R]={info['expected_return']:.4f}, "
            f"Vol={info['volatility']:.4f}"
        )
        
        return weights, info
    
    def minimum_variance(
        self,
        cov_matrix: np.ndarray,
        long_only: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Minimum variance portfolio
        
        Args:
            cov_matrix: Covariance matrix
            long_only: Whether to constrain to long-only positions
            
        Returns:
            Tuple of (optimal weights, optimization info)
        """
        n_assets = cov_matrix.shape[0]
        
        # Decision variables
        w = cp.Variable(n_assets)
        
        # Objective: minimize variance
        portfolio_variance = cp.quad_form(w, cov_matrix)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints = [cp.sum(w) == 1]
        
        if long_only:
            constraints.append(w >= 0)
            constraints.append(w <= self.max_position)
        else:
            constraints.append(w >= -self.max_position)
            constraints.append(w <= self.max_position)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            logger.warning(f"Optimization status: {problem.status}")
            return np.ones(n_assets) / n_assets, {'status': problem.status}
        
        weights = w.value
        weights[np.abs(weights) < self.min_position] = 0
        
        if np.sum(np.abs(weights)) > 0:
            weights = weights / np.sum(np.abs(weights))
        
        info = {
            'status': problem.status,
            'volatility': np.sqrt(weights @ cov_matrix @ weights)
        }
        
        return weights, info
    
    def maximum_sharpe(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.0,
        long_only: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Maximum Sharpe ratio portfolio
        
        Args:
            expected_returns: Expected returns vector
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate
            long_only: Whether to constrain to long-only positions
            
        Returns:
            Tuple of (optimal weights, optimization info)
        """
        # Sharpe ratio maximization is non-convex, but can be reformulated
        # We'll use a grid search over risk aversion parameters
        
        best_sharpe = -np.inf
        best_weights = None
        best_info = None
        
        for risk_aversion in np.linspace(0.1, 10, 50):
            weights, info = self.mean_variance_optimization(
                expected_returns,
                cov_matrix,
                risk_aversion,
                long_only
            )
            
            if info['status'] == cp.OPTIMAL:
                excess_return = info['expected_return'] - risk_free_rate
                sharpe = excess_return / info['volatility'] if info['volatility'] > 0 else 0
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = weights
                    best_info = info
                    best_info['sharpe_ratio'] = sharpe
        
        if best_weights is None:
            logger.warning("Failed to find optimal Sharpe portfolio")
            n_assets = len(expected_returns)
            return np.ones(n_assets) / n_assets, {'status': 'FAILED'}
        
        logger.info(f"Max Sharpe optimization: Sharpe={best_sharpe:.4f}")
        
        return best_weights, best_info


class RiskParityOptimizer:
    """
    Risk parity portfolio optimization
    Equalizes risk contribution across assets
    """
    
    @staticmethod
    def risk_parity_weights(
        cov_matrix: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        Calculate risk parity weights using iterative algorithm
        
        Args:
            cov_matrix: Covariance matrix
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Risk parity weights
        """
        n_assets = cov_matrix.shape[0]
        
        # Initialize with equal weights
        weights = np.ones(n_assets) / n_assets
        
        for iteration in range(max_iter):
            # Calculate portfolio volatility
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            
            # Calculate marginal risk contribution
            marginal_contrib = (cov_matrix @ weights) / port_vol
            
            # Calculate risk contribution
            risk_contrib = weights * marginal_contrib
            
            # Target risk contribution (equal for all assets)
            target_risk = port_vol / n_assets
            
            # Update weights
            weights_new = weights * (target_risk / risk_contrib)
            
            # Normalize
            weights_new = weights_new / np.sum(weights_new)
            
            # Check convergence
            if np.max(np.abs(weights_new - weights)) < tol:
                logger.info(f"Risk parity converged in {iteration} iterations")
                return weights_new
            
            weights = weights_new
        
        logger.warning(f"Risk parity did not converge after {max_iter} iterations")
        return weights


def calculate_expected_returns(
    returns: pd.DataFrame,
    method: str = 'mean'
) -> np.ndarray:
    """
    Calculate expected returns from historical data
    
    Args:
        returns: DataFrame of asset returns
        method: Method to estimate ('mean', 'ewma')
        
    Returns:
        Expected returns vector
    """
    if method == 'mean':
        return returns.mean().values
    elif method == 'ewma':
        return returns.ewm(span=60).mean().iloc[-1].values
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_covariance_matrix(
    returns: pd.DataFrame,
    method: str = 'sample'
) -> np.ndarray:
    """
    Calculate covariance matrix from returns
    
    Args:
        returns: DataFrame of asset returns
        method: Method to estimate ('sample', 'ewma', 'shrinkage')
        
    Returns:
        Covariance matrix
    """
    if method == 'sample':
        return returns.cov().values
    elif method == 'ewma':
        return returns.ewm(span=60).cov().iloc[-len(returns.columns):].values
    elif method == 'shrinkage':
        # Simple shrinkage towards diagonal
        sample_cov = returns.cov().values
        prior = np.diag(np.diag(sample_cov))
        shrinkage = 0.2
        return shrinkage * prior + (1 - shrinkage) * sample_cov
    else:
        raise ValueError(f"Unknown method: {method}")
