"""
Option Greeks Calculation Utilities using Black-Scholes Model.

This module provides the OptionPricing class for calculating implied volatility
and option Greeks (Delta, Gamma, Theta, Vega, Rho) using the Black-Scholes analytical formula.
"""

import logging
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

logger = logging.getLogger(__name__)

class OptionPricing:
    """
    Black-Scholes Option Pricing and Greeks Calculator.
    """
    
    def __init__(self, S: float, K: float, r: float, T: float):
        """
        Initialize OptionPricing calculator.
        
        Args:
            S: Spot price of the underlying asset
            K: Strike price of the option
            r: Risk-free interest rate (annualized, e.g., 0.05 for 5%)
            T: Time to expiration in years
        """
        self.S = float(S)
        self.K = float(K)
        self.r = float(r)
        self.T = float(T)
        
        # Avoid division by zero or log of zero/negative
        # NOTE: Removed artificial TTE floor - validation should be done in transform script
        self.S = max(self.S, 1e-5)
        
        self.IV_LOWER_BOUND = 0.0001
        self.IV_UPPER_BOUND = 5.0  # 500% volatility cap for sanity

    def BS_d1(self, sigma: float) -> float:
        """Calculate d1 term of Black-Scholes."""
        if sigma < self.IV_LOWER_BOUND:
            return np.inf if self.S > self.K else -np.inf
            
        sigma_sqrt_t = sigma * np.sqrt(self.T)
        if sigma_sqrt_t == 0:
             return np.inf if self.S > self.K else -np.inf
             
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / sigma_sqrt_t
        return d1

    def BS_d2(self, sigma: float) -> float:
        """Calculate d2 term of Black-Scholes."""
        if sigma < self.IV_LOWER_BOUND:
             return np.inf if self.S > self.K else -np.inf
             
        return self.BS_d1(sigma) - sigma * np.sqrt(self.T)

    def BS_CallPricing(self, sigma: float) -> float:
        """Calculate theoretical Call price."""
        if sigma <= 0:
            return max(0.0, self.S - self.K * np.exp(-self.r * self.T))
            
        d1 = self.BS_d1(sigma)
        d2 = self.BS_d2(sigma)
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def BS_PutPricing(self, sigma: float) -> float:
        """Calculate theoretical Put price."""
        if sigma <= 0:
            return max(0.0, self.K * np.exp(-self.r * self.T) - self.S)
            
        d1 = self.BS_d1(sigma)
        d2 = self.BS_d2(sigma)
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def ImplVolWithBrent(self, option_price: float, option_type: str) -> float:
        """
        Calculate Implied Volatility using Brent's method.
        
        Args:
            option_price: Market price of the option
            option_type: 'CE' (Call) or 'PE' (Put)
            
        Returns:
            Implied Volatility (decimal)
        """
        pricing_func = self.BS_CallPricing if option_type == 'CE' else self.BS_PutPricing
        
        # Intrinsic value check
        intrinsic = 0.0
        if option_type == 'CE':
            intrinsic = max(0.0, self.S - self.K * np.exp(-self.r * self.T))
        else:
            intrinsic = max(0.0, self.K * np.exp(-self.r * self.T) - self.S)
            
        if option_price <= intrinsic + 0.001:
             return self.IV_LOWER_BOUND

        try:
            def objective(sigma):
                return pricing_func(sigma) - option_price

            # Check bounds first
            y_min = objective(self.IV_LOWER_BOUND)
            y_max = objective(self.IV_UPPER_BOUND)

            if y_min * y_max > 0:
                # Root not bracketed - cannot solve for IV
                # This indicates bad data (price too high or too low relative to theoretical bounds)
                logger.debug(
                    f"IV solver: root not bracketed. S={self.S:.2f}, K={self.K:.2f}, "
                    f"price={option_price:.2f}, type={option_type}, "
                    f"bounds=[{y_min:.4f}, {y_max:.4f}], T={self.T:.6f}y"
                )
                if abs(y_min) < abs(y_max):
                    return self.IV_LOWER_BOUND
                else:
                    # Price is too high - return None to signal failure (caller should use analytic fallback)
                    logger.debug(
                        f"IV solver: price too high, cannot solve. "
                        f"S={self.S:.2f}, K={self.K:.2f}, price={option_price:.2f}, "
                        f"theoretical_max_at_upper_bound={pricing_func(self.IV_UPPER_BOUND):.2f}"
                    )
                    return None  # Signal failure - caller should use analytic fallback

            iv = brentq(objective, self.IV_LOWER_BOUND, self.IV_UPPER_BOUND, xtol=1e-4)
            return max(iv, self.IV_LOWER_BOUND)
            
        except Exception:
            return self.IV_LOWER_BOUND

    def Delta(self, sigma: float, option_type: str) -> float:
        """Calculate Delta."""
        d1 = self.BS_d1(sigma)
        if option_type == 'CE':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0

    def Gamma(self, sigma: float) -> float:
        """Calculate Gamma (same for Call and Put)."""
        if sigma <= self.IV_LOWER_BOUND or self.S <= 0 or self.T <= 0:
            return 0.0
        d1 = self.BS_d1(sigma)
        return norm.pdf(d1) / (self.S * sigma * np.sqrt(self.T))

    def Theta(self, sigma: float, option_type: str) -> float:
        """Calculate Theta (annualized)."""
        if sigma <= self.IV_LOWER_BOUND:
             return 0.0
             
        d1 = self.BS_d1(sigma)
        d2 = self.BS_d2(sigma)
        
        term1 = -(self.S * norm.pdf(d1) * sigma) / (2 * np.sqrt(self.T))
        
        if option_type == 'CE':
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            return term1 + term2
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            return term1 + term2

    def Vega(self, sigma: float) -> float:
        """Calculate Vega (same for Call and Put)."""
        if sigma <= self.IV_LOWER_BOUND:
             return 0.0
        d1 = self.BS_d1(sigma)
        # Vega is typically reported as change per 1% vol change, so divide by 100
        return self.S * np.sqrt(self.T) * norm.pdf(d1) / 100.0

    def Rho(self, sigma: float, option_type: str) -> float:
        """Calculate Rho."""
        if sigma <= self.IV_LOWER_BOUND:
             return 0.0
             
        d2 = self.BS_d2(sigma)
        if option_type == 'CE':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100.0
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100.0

