"""
Storm Lifeboat Matrix — Monte Carlo Simulation Engine
=======================================================
100,000-path regime-switching correlated GBM with Cholesky decomposition.

Mathematical framework:
    S_T = S_0 * exp((mu - sigma^2/2)*T + sigma*sqrt(T)*Z)
    Z = Cholesky(Corr) @ Z_independent

Key features:
- 18 correlated assets via 18x18 Cholesky-decomposed correlation matrix
- 4 volatility regimes: CALM / ELEVATED / CRISIS / PANIC
- Scenario-weighted drift blending across 15 crisis scenarios
- Per-asset VaR/CVaR, percentile distributions, downside probabilities
- Portfolio-level risk metrics and mandate generation
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from strategies.storm_lifeboat.core import (
    ASSET_ORDER,
    CALM_DRIFTS,
    CRISIS_DRIFTS,
    DEFAULT_PRICES,
    MC_DEFAULT_HORIZON,
    MC_DEFAULT_PATHS,
    REGIME_VOLATILITIES,
    SCENARIOS,
    TRADING_DAYS_PER_YEAR,
    Asset,
    AssetForecast,
    MandateLevel,
    MoonPhase,
    PortfolioForecast,
    ScenarioDefinition,
    StormConfig,
    VolRegime,
    build_correlation_matrix,
)

logger = logging.getLogger(__name__)


def _classify_regime(vix: float) -> VolRegime:
    """Classify volatility regime from VIX level."""
    if vix > 40:
        return VolRegime.PANIC
    if vix > 25:
        return VolRegime.CRISIS
    if vix > 15:
        return VolRegime.ELEVATED
    return VolRegime.CALM


def _determine_mandate(
    spy_prob_10: float,
    spy_prob_15: float,
    regime: VolRegime,
    coherence: float,
    config: StormConfig,
) -> MandateLevel:
    """Determine trading mandate from probability distribution and coherence."""
    if regime == VolRegime.PANIC and spy_prob_15 > config.max_conviction_prob:
        return MandateLevel.MAX_CONVICTION
    if spy_prob_10 > config.aggressive_prob_threshold:
        return MandateLevel.AGGRESSIVE
    if regime in (VolRegime.CRISIS, VolRegime.PANIC):
        return MandateLevel.STANDARD
    if regime == VolRegime.ELEVATED:
        return MandateLevel.DEFENSIVE
    return MandateLevel.OBSERVE


class StormMonteCarloEngine:
    """Regime-switching correlated Monte Carlo engine for 18 assets.

    Runs N simulations over T trading days using:
    - Scenario-weighted drift blending
    - Regime-dependent volatility profiles
    - Cholesky-decomposed correlation structure
    - Nearest-PSD correction for numerical stability
    """

    def __init__(self, config: Optional[StormConfig] = None) -> None:
        self.config = config or StormConfig()
        self._n = self.config.n_simulations
        self._T_days = self.config.horizon_days
        self._T = self._T_days / TRADING_DAYS_PER_YEAR
        self._rng = np.random.default_rng(seed=self.config.seed)

        # Build and decompose correlation matrix
        corr = build_correlation_matrix()
        eigvals = np.linalg.eigvalsh(corr)
        if np.any(eigvals < -1e-8):
            logger.warning("Correlation matrix not PSD — applying nearest-PSD fix")
            corr = self._nearest_psd(corr)
        self._cholesky = np.linalg.cholesky(corr)
        self._n_assets = len(ASSET_ORDER)

    @staticmethod
    def _nearest_psd(mat: np.ndarray) -> np.ndarray:
        """Project matrix to nearest positive semi-definite."""
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _compute_scenario_drifts(
        self,
        active_scenarios: List[ScenarioDefinition],
        regime: VolRegime,
    ) -> np.ndarray:
        """Blend drifts from active scenarios weighted by their probabilities.

        For each asset, the final drift is:
            mu = sum(scenario.probability * scenario_drift[asset]) / sum(probabilities)
        weighted across all active scenarios, then blended with the regime base drift.
        """
        # Start with regime base drifts
        if regime in (VolRegime.CRISIS, VolRegime.PANIC):
            base_drifts = np.array([CRISIS_DRIFTS.get(a, 0.0) for a in ASSET_ORDER])
        else:
            base_drifts = np.array([CALM_DRIFTS.get(a, 0.0) for a in ASSET_ORDER])

        if not active_scenarios:
            return base_drifts

        # Compute scenario-weighted drift adjustments
        total_weight = sum(s.probability for s in active_scenarios)
        if total_weight < 0.01:
            return base_drifts

        scenario_adj = np.zeros(self._n_assets)
        for sc in active_scenarios:
            for i, asset in enumerate(ASSET_ORDER):
                if asset in sc.beneficiary_assets:
                    # Positive boost proportional to severity
                    scenario_adj[i] += sc.probability * sc.impact_severity * 0.30
                elif asset in sc.victim_assets:
                    # Negative impact proportional to severity
                    scenario_adj[i] -= sc.probability * sc.impact_severity * 0.40

        # Normalize by total scenario weight and blend 60/40 with base
        scenario_adj /= max(total_weight, 0.01)
        blended = 0.60 * base_drifts + 0.40 * scenario_adj

        return blended

    def simulate(
        self,
        prices: Optional[Dict[Asset, float]] = None,
        vix: float = 25.0,
        active_scenarios: Optional[List[ScenarioDefinition]] = None,
        regime_override: Optional[VolRegime] = None,
        moon_phase: MoonPhase = MoonPhase.NEW,
        coherence_score: float = 0.5,
    ) -> PortfolioForecast:
        """Run full Monte Carlo simulation.

        Args:
            prices: Current asset prices (defaults to March 2026 baselines)
            vix: Current VIX level for regime classification
            active_scenarios: Currently active crisis scenarios
            regime_override: Force a specific regime
            moon_phase: Current 13-moon phase
            coherence_score: PlanckPhire coherence (0-1)

        Returns:
            PortfolioForecast with per-asset forecasts and mandate
        """
        from datetime import datetime

        t0 = time.perf_counter()

        prices = prices or dict(DEFAULT_PRICES)
        regime = regime_override or _classify_regime(vix)
        scenarios = active_scenarios or [s for s in SCENARIOS if s.probability > 0.15]

        # Get regime-specific volatilities
        vol_profile = REGIME_VOLATILITIES[regime]
        sigma = np.array([vol_profile.get(a, 0.25) for a in ASSET_ORDER])

        # Compute blended drifts from active scenarios
        mu = self._compute_scenario_drifts(scenarios, regime)

        logger.info(
            "Storm MC: %d paths, %dd horizon, regime=%s, %d active scenarios",
            self._n, self._T_days, regime.value, len(scenarios),
        )

        # Generate correlated normals: (n_sims, n_assets)
        Z_ind = self._rng.standard_normal((self._n, self._n_assets))
        Z_corr = Z_ind @ self._cholesky.T

        # GBM terminal prices
        drift = (mu - 0.5 * sigma ** 2) * self._T
        diffusion = sigma * np.sqrt(self._T)

        S0 = np.array([prices.get(a, DEFAULT_PRICES[a]) for a in ASSET_ORDER])
        log_returns = drift[np.newaxis, :] + diffusion[np.newaxis, :] * Z_corr
        S_T = S0[np.newaxis, :] * np.exp(log_returns)

        # Build per-asset forecasts
        asset_forecasts: Dict[Asset, AssetForecast] = {}
        for i, asset in enumerate(ASSET_ORDER):
            terminal = S_T[:, i]
            current = S0[i]
            returns = (terminal - current) / current

            sorted_returns = np.sort(returns)
            var_idx = int(0.05 * self._n)

            forecast = AssetForecast(
                asset=asset,
                current_price=float(current),
                mean_price=float(np.mean(terminal)),
                median_price=float(np.median(terminal)),
                pct_5=float(np.percentile(terminal, 5)),
                pct_25=float(np.percentile(terminal, 25)),
                pct_75=float(np.percentile(terminal, 75)),
                pct_95=float(np.percentile(terminal, 95)),
                expected_return_pct=float(np.mean(returns) * 100),
                prob_down_10=float(np.mean(returns <= -0.10)),
                prob_down_20=float(np.mean(returns <= -0.20)),
                prob_down_30=float(np.mean(returns <= -0.30)),
                prob_up_10=float(np.mean(returns >= 0.10)),
                prob_up_20=float(np.mean(returns >= 0.20)),
                var_95=float(-sorted_returns[var_idx]),
                cvar_95=float(-np.mean(sorted_returns[:var_idx])),
                n_paths=self._n,
            )
            asset_forecasts[asset] = forecast

        # Portfolio-level metrics
        portfolio_returns = np.mean(log_returns, axis=1)
        sorted_port = np.sort(portfolio_returns)
        p_var_idx = int(0.05 * self._n)

        # SPY drives mandate
        spy_fc = asset_forecasts.get(Asset.SPY)
        spy_p10 = spy_fc.prob_down_10 if spy_fc else 0.3
        spy_p15 = float(np.mean(
            (S_T[:, ASSET_ORDER.index(Asset.SPY)] - S0[ASSET_ORDER.index(Asset.SPY)])
            / S0[ASSET_ORDER.index(Asset.SPY)] <= -0.15
        )) if Asset.SPY in ASSET_ORDER else 0.2

        mandate = _determine_mandate(spy_p10, spy_p15, regime, coherence_score, self.config)

        active_codes = [s.code for s in scenarios]
        result = PortfolioForecast(
            timestamp=datetime.utcnow(),
            regime=regime,
            asset_forecasts=asset_forecasts,
            portfolio_var_95=float(-sorted_port[p_var_idx]),
            portfolio_cvar_95=float(-np.mean(sorted_port[:p_var_idx])),
            weighted_return_pct=float(np.mean(portfolio_returns) * 100),
            mandate=mandate,
            active_scenarios=active_codes,
            coherence_score=coherence_score,
            moon_phase=moon_phase,
            horizon_days=self._T_days,
            n_simulations=self._n,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Storm MC complete: %.0fms, SPY mean=$%.1f (%+.1f%%), "
            "P(10%%down)=%.0f%%, mandate=%s",
            elapsed_ms,
            asset_forecasts[Asset.SPY].mean_price if spy_fc else 0,
            asset_forecasts[Asset.SPY].expected_return_pct if spy_fc else 0,
            spy_p10 * 100,
            mandate.value,
        )

        return result

    def simulate_put_payoff(
        self,
        asset: Asset,
        strike_pct_otm: float = 0.10,
        premium_per_share: float = 5.0,
        n_contracts: int = 1,
        forecast: Optional[PortfolioForecast] = None,
        prices: Optional[Dict[Asset, float]] = None,
    ) -> Dict[str, float]:
        """Simulate put option payoff across all MC paths.

        Args:
            asset: Which asset to buy puts on
            strike_pct_otm: Strike distance OTM (0.10 = 10% below spot)
            premium_per_share: Cost per share (not per contract)
            n_contracts: Number of contracts (× 100 shares each)
            forecast: Pre-computed forecast (re-simulates if None)
            prices: Override prices

        Returns:
            Dict with expected_return, win_rate, max_payoff, etc.
        """
        prices = prices or dict(DEFAULT_PRICES)
        S0 = prices.get(asset, DEFAULT_PRICES[asset])
        K = S0 * (1.0 - strike_pct_otm)
        multiplier = n_contracts * 100

        # Generate terminal prices for this asset
        asset_idx = ASSET_ORDER.index(asset)
        Z = self._rng.standard_normal((self._n, self._n_assets))
        Z_corr = Z @ self._cholesky.T

        regime = self.config.regime
        vol_profile = REGIME_VOLATILITIES[regime]
        sigma_a = vol_profile.get(asset, 0.25)
        mu_a = CRISIS_DRIFTS.get(asset, 0.0)

        drift = (mu_a - 0.5 * sigma_a ** 2) * self._T
        diffusion = sigma_a * np.sqrt(self._T) * Z_corr[:, asset_idx]
        S_T = S0 * np.exp(drift + diffusion)

        # Put payoff
        intrinsic = np.maximum(K - S_T, 0.0)
        total_cost = premium_per_share * multiplier
        total_payoff = intrinsic * multiplier
        net_pnl = total_payoff - total_cost

        wins = net_pnl > 0
        win_rate = float(np.mean(wins))

        return {
            "asset": asset.value,
            "spot": S0,
            "strike": K,
            "premium_per_share": premium_per_share,
            "n_contracts": n_contracts,
            "total_cost": total_cost,
            "expected_pnl": float(np.mean(net_pnl)),
            "median_pnl": float(np.median(net_pnl)),
            "expected_return_pct": float(np.mean(net_pnl) / total_cost * 100),
            "win_rate": win_rate,
            "avg_win": float(np.mean(net_pnl[wins])) if np.any(wins) else 0.0,
            "avg_loss": float(np.mean(net_pnl[~wins])) if np.any(~wins) else 0.0,
            "max_payoff": float(np.max(net_pnl)),
            "p5_pnl": float(np.percentile(net_pnl, 5)),
            "p95_pnl": float(np.percentile(net_pnl, 95)),
        }
