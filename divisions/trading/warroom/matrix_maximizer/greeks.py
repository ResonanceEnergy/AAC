"""
MATRIX MAXIMIZER — Black-Scholes Greeks Engine
================================================
Full analytical Black-Scholes-Merton pricing for European puts.

Computes:
    Price  — BSM put price
    Delta  — dP/dS (rate of change vs spot)
    Gamma  — d2P/dS2 (rate of change of delta)
    Vega   — dP/dsigma (sensitivity to IV, per 1% move)
    Theta  — dP/dt (time decay per day)
    Rho    — dP/dr (sensitivity to interest rates)

Formulas:
    d1 = [ln(S/K) + (r - q + sigma^2/2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    Put  = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)
    Delta_put = -e^(-qT)*N(-d1)
    Gamma = e^(-qT)*n(d1) / (S*sigma*sqrt(T))
    Vega  = S*e^(-qT)*n(d1)*sqrt(T) / 100
    Theta = [-S*sigma*e^(-qT)*n(d1)/(2*sqrt(T)) + r*K*e^(-rT)*N(-d2)
             - q*S*e^(-qT)*N(-d1)] / 365
    Rho   = -K*T*e^(-rT)*N(-d2) / 100
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class GreeksResult:
    """Complete Black-Scholes Greeks for a single put option."""
    # Inputs
    spot: float          # S — current underlying price
    strike: float        # K — option strike
    time_years: float    # T — time to expiry in years
    sigma: float         # IV — implied volatility (annualised)
    rate: float          # r — risk-free rate
    div_yield: float     # q — continuous dividend yield

    # Outputs
    price: float         # BS put price
    delta: float         # dP/dS
    gamma: float         # d2P/dS2
    vega: float          # dP/dsigma (per 1% IV move)
    theta: float         # dP/dt (per calendar day)
    rho: float           # dP/dr (per 1% rate move)

    # Derived
    d1: float
    d2: float
    otm_pct: float       # How far OTM: (S - K) / S
    intrinsic: float     # max(K - S, 0) for put
    extrinsic: float     # price - intrinsic

    @property
    def moneyness(self) -> str:
        if self.spot < self.strike * 0.98:
            return "ITM"
        elif self.spot > self.strike * 1.02:
            return "OTM"
        return "ATM"

    @property
    def dte(self) -> int:
        """Days to expiry."""
        return max(1, int(self.time_years * 365))

    def print_card(self) -> str:
        """Compact display card for a single option."""
        return (
            f"PUT {self.strike:.1f} | {self.dte}d | IV={self.sigma:.0%}\n"
            f"  Price: ${self.price:.2f} | Delta: {self.delta:.3f} | "
            f"Gamma: {self.gamma:.4f}\n"
            f"  Vega: ${self.vega:.2f} | Theta: ${self.theta:.3f}/day | "
            f"Rho: ${self.rho:.3f}\n"
            f"  {self.moneyness} | OTM: {self.otm_pct:.1%} | "
            f"Intrinsic: ${self.intrinsic:.2f} | Extrinsic: ${self.extrinsic:.2f}"
        )


class BlackScholesEngine:
    """Full Black-Scholes-Merton put pricing engine with all six Greeks.

    Usage:
        bs = BlackScholesEngine(rate=0.037, div_yield=0.0109)
        greeks = bs.price_put(S=667, K=600, T_days=30, sigma=0.22)
        print(greeks.print_card())
    """

    def __init__(
        self,
        rate: float = 0.037,          # 3-month Treasury yield (March 2026)
        div_yield: float = 0.0109,    # SPY dividend yield
    ) -> None:
        self.rate = rate
        self.div_yield = div_yield

    def price_put(
        self,
        S: float,
        K: float,
        T_days: int,
        sigma: float,
        rate: Optional[float] = None,
        div_yield: Optional[float] = None,
    ) -> GreeksResult:
        """Price a European put and compute all Greeks.

        Args:
            S: Spot price of underlying
            K: Strike price
            T_days: Days to expiry
            sigma: Implied volatility (annualised, e.g., 0.22 = 22%)
            rate: Override risk-free rate
            div_yield: Override dividend yield

        Returns:
            GreeksResult with price and all six Greeks
        """
        r = rate if rate is not None else self.rate
        q = div_yield if div_yield is not None else self.div_yield
        T = max(T_days, 1) / 365.0

        # Core BSM calculations
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Standard normal PDF and CDF
        n_d1 = norm.pdf(d1)
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)

        # Discount factors
        exp_qT = math.exp(-q * T)
        exp_rT = math.exp(-r * T)

        # Put price
        price = K * exp_rT * N_neg_d2 - S * exp_qT * N_neg_d1
        price = max(price, 0.0)

        # Delta (put)
        delta = -exp_qT * N_neg_d1

        # Gamma (same for put and call)
        gamma = exp_qT * n_d1 / (S * sigma * sqrt_T)

        # Vega (per 1% IV move = per 0.01 sigma)
        vega = S * exp_qT * n_d1 * sqrt_T / 100.0

        # Theta (per calendar day)
        theta_term1 = -S * sigma * exp_qT * n_d1 / (2 * sqrt_T)
        theta_term2 = r * K * exp_rT * N_neg_d2
        theta_term3 = -q * S * exp_qT * N_neg_d1
        theta = (theta_term1 + theta_term2 + theta_term3) / 365.0

        # Rho (per 1% rate move = per 0.01)
        rho = -K * T * exp_rT * N_neg_d2 / 100.0

        # OTM percentage
        otm_pct = (S - K) / S if S > 0 else 0.0

        # Intrinsic/extrinsic
        intrinsic = max(K - S, 0.0)
        extrinsic = price - intrinsic

        return GreeksResult(
            spot=S,
            strike=K,
            time_years=T,
            sigma=sigma,
            rate=r,
            div_yield=q,
            price=price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            d1=d1,
            d2=d2,
            otm_pct=otm_pct,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
        )

    def price_put_spread(
        self,
        S: float,
        K_long: float,
        K_short: float,
        T_days: int,
        sigma: float,
    ) -> dict:
        """Price a bear put spread (buy higher K, sell lower K).

        Returns:
            dict with net_debit, max_profit, max_loss, breakeven, greeks for each leg
        """
        long_leg = self.price_put(S, K_long, T_days, sigma)
        short_leg = self.price_put(S, K_short, T_days, sigma)

        net_debit = long_leg.price - short_leg.price
        max_profit = (K_long - K_short) - net_debit
        max_loss = net_debit
        breakeven = K_long - net_debit

        return {
            "net_debit": net_debit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": breakeven,
            "reward_risk_ratio": max_profit / max_loss if max_loss > 0 else 0.0,
            "long_leg": long_leg,
            "short_leg": short_leg,
            "net_delta": long_leg.delta - short_leg.delta,
            "net_gamma": long_leg.gamma - short_leg.gamma,
            "net_vega": long_leg.vega - short_leg.vega,
            "net_theta": long_leg.theta - short_leg.theta,
        }

    def delta_decay_check(
        self,
        entry_delta: float,
        current_greeks: GreeksResult,
        threshold: float = 0.20,
    ) -> bool:
        """Check if delta has decayed beyond threshold (triggers auto-roll).

        Returns True if |current_delta| / |entry_delta| < (1 - threshold),
        meaning the position has lost ≥ threshold of its original delta.
        """
        if abs(entry_delta) < 1e-6:
            return False
        decay_ratio = abs(current_greeks.delta) / abs(entry_delta)
        return decay_ratio < (1.0 - threshold)

    def find_strike_for_delta(
        self,
        S: float,
        target_delta: float,
        T_days: int,
        sigma: float,
        tolerance: float = 0.005,
    ) -> float:
        """Binary search for strike price that gives target put delta.

        Args:
            S: Spot price
            target_delta: Target delta (negative, e.g., -0.35)
            T_days: Days to expiry
            sigma: IV
            tolerance: Delta tolerance for convergence

        Returns:
            Strike price with desired delta
        """
        K_low = S * 0.70
        K_high = S * 1.05

        for _ in range(50):
            K_mid = (K_low + K_high) / 2.0
            greeks = self.price_put(S, K_mid, T_days, sigma)
            if abs(greeks.delta - target_delta) < tolerance:
                return K_mid
            # Put delta becomes more negative (larger magnitude) as K rises
            if greeks.delta > target_delta:
                K_low = K_mid
            else:
                K_high = K_mid

        return (K_low + K_high) / 2.0
