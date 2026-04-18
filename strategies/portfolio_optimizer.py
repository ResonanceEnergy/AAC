"""Portfolio Optimization — mean-variance, risk parity, HRP, Black-Litterman.

Wraps PyPortfolioOpt and Riskfolio-Lib to provide production-ready
portfolio allocation for AAC's multi-asset universe.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class OptMethod(Enum):
    MEAN_VARIANCE = "mean_variance"
    MIN_VOLATILITY = "min_volatility"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    HRP = "hrp"
    BLACK_LITTERMAN = "black_litterman"


@dataclass
class OptimizationResult:
    """Output of any portfolio optimization run."""

    method: str
    weights: dict[str, float]
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    diversification_ratio: float = 0.0
    max_drawdown_estimate: float = 0.0
    risk_contributions: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    # convenience -----------------------------------------------------------
    @property
    def top_holdings(self) -> list[tuple[str, float]]:
        return sorted(self.weights.items(), key=lambda x: x[1], reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "weights": self.weights,
            "expected_return": round(self.expected_return, 6),
            "expected_volatility": round(self.expected_volatility, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "diversification_ratio": round(self.diversification_ratio, 4),
            "risk_contributions": {
                k: round(v, 6) for k, v in self.risk_contributions.items()
            },
        }


# ---------------------------------------------------------------------------
# Portfolio Optimizer
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    """Unified facade over PyPortfolioOpt + Riskfolio-Lib.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate for Sharpe calculations.
    frequency : int
        Trading days per year (252 default).
    weight_bounds : tuple[float, float]
        Lower/upper bound for each asset weight in MV/Max-Sharpe.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        frequency: int = 252,
        weight_bounds: tuple[float, float] = (0.0, 0.40),
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        self.weight_bounds = weight_bounds

    # ── public API ─────────────────────────────────────────────────────────

    def optimize(
        self,
        prices: pd.DataFrame,
        method: OptMethod = OptMethod.MAX_SHARPE,
        views: Optional[Dict[str, float]] = None,
        market_caps: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """Run the requested optimization.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily close prices, columns = tickers, index = dates.
        method : OptMethod
            Which optimization to run.
        views : dict
            Analyst views for Black-Litterman (ticker → expected excess return).
        market_caps : dict
            Market capitalizations for Black-Litterman equilibrium.
        """
        if prices.shape[0] < 30:
            raise ValueError("Need ≥30 price observations for reliable optimization")
        if prices.shape[1] < 2:
            raise ValueError("Need ≥2 assets to optimize")

        dispatch = {
            OptMethod.MAX_SHARPE: self._max_sharpe,
            OptMethod.MIN_VOLATILITY: self._min_vol,
            OptMethod.MEAN_VARIANCE: self._max_sharpe,  # alias
            OptMethod.RISK_PARITY: self._risk_parity,
            OptMethod.HRP: self._hrp,
            OptMethod.BLACK_LITTERMAN: self._black_litterman,
        }

        handler = dispatch.get(method)
        if handler is None:
            raise ValueError(f"Unknown optimization method: {method}")

        if method == OptMethod.BLACK_LITTERMAN:
            return handler(prices, views=views, market_caps=market_caps)
        return handler(prices)

    # ── Mean-Variance / Max-Sharpe ─────────────────────────────────────────

    def _max_sharpe(self, prices: pd.DataFrame) -> OptimizationResult:
        from pypfopt import expected_returns, risk_models
        from pypfopt.efficient_frontier import EfficientFrontier

        mu = expected_returns.mean_historical_return(
            prices, frequency=self.frequency,
        )
        S = risk_models.CovarianceShrinkage(prices, frequency=self.frequency).ledoit_wolf()

        ef = EfficientFrontier(mu, S, weight_bounds=self.weight_bounds)
        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        weights = ef.clean_weights()

        perf = ef.portfolio_performance(
            verbose=False, risk_free_rate=self.risk_free_rate,
        )

        risk_contribs = self._compute_risk_contributions(weights, S)

        return OptimizationResult(
            method=OptMethod.MAX_SHARPE.value,
            weights=dict(weights),
            expected_return=perf[0],
            expected_volatility=perf[1],
            sharpe_ratio=perf[2],
            risk_contributions=risk_contribs,
            diversification_ratio=self._diversification_ratio(weights, S),
        )

    def _min_vol(self, prices: pd.DataFrame) -> OptimizationResult:
        from pypfopt import expected_returns, risk_models
        from pypfopt.efficient_frontier import EfficientFrontier

        mu = expected_returns.mean_historical_return(
            prices, frequency=self.frequency,
        )
        S = risk_models.CovarianceShrinkage(prices, frequency=self.frequency).ledoit_wolf()

        ef = EfficientFrontier(mu, S, weight_bounds=self.weight_bounds)
        ef.min_volatility()
        weights = ef.clean_weights()

        perf = ef.portfolio_performance(
            verbose=False, risk_free_rate=self.risk_free_rate,
        )

        risk_contribs = self._compute_risk_contributions(weights, S)

        return OptimizationResult(
            method=OptMethod.MIN_VOLATILITY.value,
            weights=dict(weights),
            expected_return=perf[0],
            expected_volatility=perf[1],
            sharpe_ratio=perf[2],
            risk_contributions=risk_contribs,
            diversification_ratio=self._diversification_ratio(weights, S),
        )

    # ── Risk Parity ────────────────────────────────────────────────────────

    def _risk_parity(self, prices: pd.DataFrame) -> OptimizationResult:
        returns = prices.pct_change().dropna()
        cov = returns.cov() * self.frequency
        tickers = list(prices.columns)
        n = len(tickers)

        # Naive risk parity: w_i ∝ 1/σ_i, then normalize
        vols = np.sqrt(np.diag(cov.values))
        raw = 1.0 / np.where(vols > 1e-10, vols, 1e-10)
        raw /= raw.sum()

        weights = {t: float(raw[i]) for i, t in enumerate(tickers)}

        # Portfolio stats
        w = np.array([weights[t] for t in tickers])
        port_var = float(w @ cov.values @ w)
        port_vol = np.sqrt(port_var)
        mu = returns.mean() * self.frequency
        port_ret = float(w @ mu.values)
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        risk_contribs = self._compute_risk_contributions(weights, cov)

        return OptimizationResult(
            method=OptMethod.RISK_PARITY.value,
            weights=weights,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            risk_contributions=risk_contribs,
            diversification_ratio=self._diversification_ratio(weights, cov),
        )

    # ── Hierarchical Risk Parity ───────────────────────────────────────────

    def _hrp(self, prices: pd.DataFrame) -> OptimizationResult:
        from pypfopt import HRPOpt, expected_returns

        returns = prices.pct_change().dropna()
        hrp = HRPOpt(returns)
        weights = hrp.optimize()
        weights = dict(weights)

        # Post-optimization stats
        cov = returns.cov() * self.frequency
        mu = expected_returns.mean_historical_return(
            prices, frequency=self.frequency,
        )
        tickers = list(prices.columns)
        w = np.array([weights.get(t, 0.0) for t in tickers])
        port_var = float(w @ cov.values @ w)
        port_vol = np.sqrt(port_var)
        port_ret = float(w @ mu.values)
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        risk_contribs = self._compute_risk_contributions(weights, cov)

        return OptimizationResult(
            method=OptMethod.HRP.value,
            weights=weights,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            risk_contributions=risk_contribs,
            diversification_ratio=self._diversification_ratio(weights, cov),
        )

    # ── Black-Litterman ────────────────────────────────────────────────────

    def _black_litterman(
        self,
        prices: pd.DataFrame,
        views: Optional[Dict[str, float]] = None,
        market_caps: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        from pypfopt import BlackLittermanModel, expected_returns, risk_models
        from pypfopt.efficient_frontier import EfficientFrontier

        S = risk_models.CovarianceShrinkage(prices, frequency=self.frequency).ledoit_wolf()

        if market_caps is None:
            # Equal-weight equilibrium as fallback
            n = prices.shape[1]
            market_caps = {t: 1.0 / n for t in prices.columns}

        bl = BlackLittermanModel(
            S,
            pi="market",
            market_caps=market_caps,
            risk_aversion=2.5,
            absolute_views=views if views else {},
        )
        bl_returns = bl.bl_returns()
        bl_cov = bl.bl_cov()

        ef = EfficientFrontier(bl_returns, bl_cov, weight_bounds=self.weight_bounds)
        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        weights = ef.clean_weights()

        perf = ef.portfolio_performance(
            verbose=False, risk_free_rate=self.risk_free_rate,
        )
        risk_contribs = self._compute_risk_contributions(weights, S)

        return OptimizationResult(
            method=OptMethod.BLACK_LITTERMAN.value,
            weights=dict(weights),
            expected_return=perf[0],
            expected_volatility=perf[1],
            sharpe_ratio=perf[2],
            risk_contributions=risk_contribs,
            diversification_ratio=self._diversification_ratio(weights, S),
            details={
                "bl_expected_returns": {
                    k: float(v) for k, v in bl_returns.items()
                },
            },
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_risk_contributions(
        weights: dict[str, float],
        cov: pd.DataFrame,
    ) -> dict[str, float]:
        tickers = list(cov.columns)
        w = np.array([weights.get(t, 0.0) for t in tickers])
        port_var = float(w @ cov.values @ w)
        if port_var < 1e-12:
            return {t: 0.0 for t in tickers}
        marginal = cov.values @ w
        rc = w * marginal / port_var
        return {t: float(rc[i]) for i, t in enumerate(tickers)}

    @staticmethod
    def _diversification_ratio(
        weights: dict[str, float],
        cov: pd.DataFrame,
    ) -> float:
        tickers = list(cov.columns)
        w = np.array([weights.get(t, 0.0) for t in tickers])
        vols = np.sqrt(np.diag(cov.values))
        weighted_avg_vol = float(w @ vols)
        port_vol = np.sqrt(float(w @ cov.values @ w))
        if port_vol < 1e-12:
            return 1.0
        return weighted_avg_vol / port_vol

    # ── Convenience ────────────────────────────────────────────────────────

    def compare_methods(
        self,
        prices: pd.DataFrame,
        methods: Optional[List[OptMethod]] = None,
    ) -> list[OptimizationResult]:
        """Run multiple optimizations and return sorted by Sharpe."""
        if methods is None:
            methods = [
                OptMethod.MAX_SHARPE,
                OptMethod.MIN_VOLATILITY,
                OptMethod.RISK_PARITY,
                OptMethod.HRP,
            ]
        results = []
        for m in methods:
            try:
                results.append(self.optimize(prices, method=m))
            except Exception as exc:  # noqa: BLE001
                _log.warning("optimization_failed method=%s error=%s", m.value, str(exc))
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        return results
