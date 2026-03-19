"""
MATRIX MAXIMIZER — Monte Carlo Simulation Engine
==================================================
10,000-path correlated Geometric Brownian Motion (GBM).

Oil simulated FIRST as the primary driver (Hormuz supply disruption bias).
All other assets correlated to oil via negative betas plus their own noise.

Mathematical framework:
    S_T = S_0 * exp((mu - sigma^2/2)*T + sigma*sqrt(T)*Z)
    where Z is a vector of correlated standard normals.

Multivariate correlation via Cholesky decomposition of the correlation matrix.
Scenario-weighted drifts from the geopolitical model (BASE/BEAR/BULL).

Outputs:
    - Per-asset: mean, median, percentiles, return distribution
    - Downside probabilities: P(10% drop), P(15% drop), P(20% drop)
    - VaR/CVaR at 95% confidence
    - Dynamic mandate based on probability distribution
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from strategies.matrix_maximizer.core import (
    Asset,
    AssetForecast,
    ASSET_OIL_BETAS,
    ASSET_VOLATILITIES,
    CORRELATION_MATRIX,
    DEFAULT_PRICES,
    MatrixConfig,
    MandateLevel,
    PortfolioForecast,
    Scenario,
    SCENARIO_DRIFTS,
    ScenarioWeights,
    SystemMandate,
)

logger = logging.getLogger(__name__)

# Asset order matching CORRELATION_MATRIX rows/columns
_ASSET_ORDER: List[Asset] = [
    Asset.SPY, Asset.QQQ, Asset.USO, Asset.BITO, Asset.TLT,
    Asset.JETS, Asset.KRE, Asset.HYG, Asset.XLY, Asset.ZIM, Asset.XLE,
]


class MonteCarloEngine:
    """Oil-integrated correlated Monte Carlo simulation engine.

    Runs N simulations of all tracked assets over T days using
    scenario-weighted GBM with Cholesky-decomposed correlation.
    """

    def __init__(self, config: MatrixConfig) -> None:
        self.config = config
        self._n = config.n_simulations
        self._T = config.horizon_days / 252.0  # Trading days → years
        self._T_days = config.horizon_days
        self._rng = np.random.default_rng(seed=42)

        # Decompose correlation matrix once
        corr = np.array(CORRELATION_MATRIX, dtype=np.float64)
        # Ensure positive semi-definite (numerical safety)
        eigvals = np.linalg.eigvalsh(corr)
        if np.any(eigvals < -1e-8):
            logger.warning("Correlation matrix not PSD, applying nearest PSD fix")
            corr = self._nearest_psd(corr)
        self._cholesky = np.linalg.cholesky(corr)

    @staticmethod
    def _nearest_psd(mat: np.ndarray) -> np.ndarray:
        """Project matrix to nearest positive semi-definite."""
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def simulate(
        self,
        prices: Optional[Dict[Asset, float]] = None,
        scenario_weights: Optional[ScenarioWeights] = None,
        oil_price_override: Optional[float] = None,
        vix_override: Optional[float] = None,
    ) -> PortfolioForecast:
        """Run full Monte Carlo simulation.

        Args:
            prices: Current asset prices (defaults to March 2026 baselines)
            scenario_weights: Scenario probabilities (auto-adjusted if None)
            oil_price_override: Override oil price for scenario adjustment
            vix_override: Override VIX for scenario adjustment

        Returns:
            PortfolioForecast with all asset forecasts and system mandate
        """
        prices = prices or dict(DEFAULT_PRICES)
        weights = scenario_weights or ScenarioWeights()

        # Auto-adjust weights based on oil and VIX
        oil_price = oil_price_override or prices.get(Asset.USO, 96.5)
        vix = vix_override or 22.0

        weights = weights.adjust_for_oil(oil_price)
        weights = weights.adjust_for_vix(vix)
        weights.validate()

        logger.info(
            "MATRIX MAXIMIZER MC: %d sims, %dd horizon, weights BASE=%.0f%% BEAR=%.0f%% BULL=%.0f%%",
            self._n, self._T_days,
            weights.base * 100, weights.bear * 100, weights.bull * 100,
        )

        # Compute scenario-weighted drifts for each asset
        n_assets = len(_ASSET_ORDER)
        mu = np.zeros(n_assets)
        sigma = np.zeros(n_assets)

        for i, asset in enumerate(_ASSET_ORDER):
            # Weighted drift across scenarios
            mu[i] = (
                weights.base * SCENARIO_DRIFTS[Scenario.BASE].get(asset, 0.0)
                + weights.bear * SCENARIO_DRIFTS[Scenario.BEAR].get(asset, 0.0)
                + weights.bull * SCENARIO_DRIFTS[Scenario.BULL].get(asset, 0.0)
            )
            sigma[i] = ASSET_VOLATILITIES.get(asset, 0.20)

        # Generate correlated random normals: (n_sims, n_assets)
        Z_independent = self._rng.standard_normal((self._n, n_assets))
        Z_correlated = Z_independent @ self._cholesky.T

        # GBM terminal prices: S_T = S_0 * exp((mu - sigma^2/2)*T + sigma*sqrt(T)*Z)
        drift = (mu - 0.5 * sigma ** 2) * self._T
        diffusion = sigma * np.sqrt(self._T)

        # (n_sims, n_assets) matrix of terminal prices
        log_returns = drift[np.newaxis, :] + diffusion[np.newaxis, :] * Z_correlated
        S0 = np.array([prices.get(a, DEFAULT_PRICES[a]) for a in _ASSET_ORDER])
        S_T = S0[np.newaxis, :] * np.exp(log_returns)

        # Build per-asset forecasts
        asset_forecasts: Dict[Asset, AssetForecast] = {}
        for i, asset in enumerate(_ASSET_ORDER):
            terminal = S_T[:, i]
            current = S0[i]
            returns = (terminal - current) / current

            # 1-day VaR approximation: scale from T-day to 1-day
            daily_returns = returns / np.sqrt(self._T_days)
            sorted_daily = np.sort(daily_returns)
            var_idx_95 = int(0.05 * self._n)

            # 90-day VaR
            sorted_90d = np.sort(returns)
            var_90d_idx = int(0.05 * self._n)

            forecast = AssetForecast(
                asset=asset,
                current_price=current,
                mean_price=float(np.mean(terminal)),
                median_price=float(np.median(terminal)),
                pct_5=float(np.percentile(terminal, 5)),
                pct_25=float(np.percentile(terminal, 25)),
                pct_75=float(np.percentile(terminal, 75)),
                pct_95=float(np.percentile(terminal, 95)),
                expected_return_pct=float(np.mean(returns) * 100),
                prob_down_10=float(np.mean(returns <= -0.10)),
                prob_down_15=float(np.mean(returns <= -0.15)),
                prob_down_20=float(np.mean(returns <= -0.20)),
                prob_up_5=float(np.mean(returns >= 0.05)),
                var_95_1d=float(-sorted_daily[var_idx_95]),
                cvar_95_1d=float(-np.mean(sorted_daily[:var_idx_95])),
                var_95_90d=float(-sorted_90d[var_90d_idx]),
                cvar_95_90d=float(-np.mean(sorted_90d[:var_90d_idx])),
                simulated_paths=self._n,
            )
            asset_forecasts[asset] = forecast

        # Portfolio-level metrics (equal-weight for VaR estimation)
        portfolio_returns = np.mean(log_returns, axis=1)  # Average across assets
        sorted_port = np.sort(portfolio_returns)
        p_var_idx = int(0.05 * self._n)

        # SPY downside probability drives mandate
        spy_prob_10 = asset_forecasts[Asset.SPY].prob_down_10 if Asset.SPY in asset_forecasts else 0.3

        mandate = SystemMandate.from_probabilities(
            prob_10_down=spy_prob_10,
            oil_price=oil_price,
            vix=vix,
            config=self.config,
        )

        forecast = PortfolioForecast(
            timestamp=datetime.utcnow(),
            scenario_weights=weights,
            asset_forecasts=asset_forecasts,
            weighted_return=float(np.mean(portfolio_returns) * 100),
            portfolio_var_95=float(-sorted_port[p_var_idx]),
            portfolio_cvar_95=float(-np.mean(sorted_port[:p_var_idx])),
            mandate=mandate,
            horizon_days=self._T_days,
            n_simulations=self._n,
        )

        logger.info(
            "MC complete: SPY mean=%.1f (%.1f%%), P(10%%down)=%.0f%%, mandate=%s",
            asset_forecasts[Asset.SPY].mean_price,
            asset_forecasts[Asset.SPY].expected_return_pct,
            spy_prob_10 * 100,
            mandate.level.value,
        )

        return forecast

    def simulate_put_strategy(
        self,
        put_strike_pct: float = 0.10,
        put_premium: float = 8.0,
        forecast: Optional[PortfolioForecast] = None,
        prices: Optional[Dict[Asset, float]] = None,
    ) -> Dict[str, float]:
        """Simulate the put-buying strategy return across all MC paths.

        Buys a put at (1 - put_strike_pct) * S0, pays put_premium.
        Calculates intrinsic payoff at expiry across all simulated SPY paths.

        Returns:
            Dict with expected_return, win_rate, avg_win, avg_loss, max_payoff
        """
        if forecast is None:
            forecast = self.simulate(prices)

        spy_forecast = forecast.asset_forecasts.get(Asset.SPY)
        if spy_forecast is None:
            return {"error": "No SPY forecast available"}

        S0 = spy_forecast.current_price
        K = S0 * (1.0 - put_strike_pct)

        # Re-simulate to get raw terminal prices for SPY
        n_assets = len(_ASSET_ORDER)
        Z = self._rng.standard_normal((self._n, n_assets))
        Z_corr = Z @ self._cholesky.T

        weights = forecast.scenario_weights
        mu_spy = (
            weights.base * SCENARIO_DRIFTS[Scenario.BASE][Asset.SPY]
            + weights.bear * SCENARIO_DRIFTS[Scenario.BEAR][Asset.SPY]
            + weights.bull * SCENARIO_DRIFTS[Scenario.BULL][Asset.SPY]
        )
        sigma_spy = ASSET_VOLATILITIES[Asset.SPY]
        spy_idx = _ASSET_ORDER.index(Asset.SPY)

        drift = (mu_spy - 0.5 * sigma_spy ** 2) * self._T
        diffusion = sigma_spy * np.sqrt(self._T) * Z_corr[:, spy_idx]
        S_T = S0 * np.exp(drift + diffusion)

        # Put payoff per contract (×100 shares per contract)
        intrinsic = np.maximum(K - S_T, 0.0)
        net_pnl = intrinsic - put_premium  # Per share P&L

        wins = net_pnl > 0
        win_rate = float(np.mean(wins))

        return {
            "strike": K,
            "premium_paid": put_premium,
            "expected_return_pct": float(np.mean(net_pnl) / put_premium * 100),
            "win_rate": win_rate,
            "avg_win": float(np.mean(net_pnl[wins])) if np.any(wins) else 0.0,
            "avg_loss": float(np.mean(net_pnl[~wins])) if np.any(~wins) else 0.0,
            "max_payoff": float(np.max(net_pnl)),
            "median_pnl": float(np.median(net_pnl)),
        }
