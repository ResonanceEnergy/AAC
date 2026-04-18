"""Pairs Trading — Cointegration Scanner & Z-Score Signals.

Finds cointegrated asset pairs via Engle-Granger and generates
mean-reversion trading signals from spread z-scores.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CointegrationResult:
    """Result of cointegration test between two assets."""

    asset_a: str
    asset_b: str
    p_value: float
    hedge_ratio: float        # OLS beta
    half_life: float          # mean-reversion speed (days)
    is_cointegrated: bool     # p_value < threshold
    test_statistic: float = 0.0
    critical_values: dict[str, float] = field(default_factory=dict)


@dataclass
class PairsSignal:
    """Trading signal for a cointegrated pair."""

    asset_a: str
    asset_b: str
    z_score: float
    signal: str              # "long_spread", "short_spread", "flat"
    spread_value: float
    hedge_ratio: float
    half_life: float
    entry_z: float = 2.0
    exit_z: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": f"{self.asset_a}/{self.asset_b}",
            "z_score": round(self.z_score, 3),
            "signal": self.signal,
            "spread": round(self.spread_value, 4),
            "hedge_ratio": round(self.hedge_ratio, 4),
            "half_life": round(self.half_life, 1),
        }


@dataclass
class PairsScanResult:
    """Result of scanning multiple pairs for cointegration."""

    n_pairs_tested: int
    n_cointegrated: int
    pairs: list[CointegrationResult] = field(default_factory=list)
    signals: list[PairsSignal] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pairs Trading Engine
# ---------------------------------------------------------------------------

class PairsTradingEngine:
    """Cointegration-based pairs trading.

    Parameters
    ----------
    p_threshold : float
        P-value threshold for Engle-Granger cointegration test.
    entry_z : float
        Z-score threshold to enter a spread trade.
    exit_z : float
        Z-score threshold to exit.
    lookback : int
        Lookback window for z-score calculation.
    """

    def __init__(
        self,
        p_threshold: float = 0.05,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        lookback: int = 60,
    ) -> None:
        self.p_threshold = p_threshold
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback

    # ── Cointegration testing ─────────────────────────────────────────────

    def test_cointegration(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        asset_a: str = "A",
        asset_b: str = "B",
    ) -> CointegrationResult:
        """Run Engle-Granger cointegration test on two price series."""
        from statsmodels.tsa.stattools import coint

        # Align and drop NaNs
        combined = pd.concat([prices_a, prices_b], axis=1).dropna()
        if len(combined) < 30:
            return CointegrationResult(
                asset_a=asset_a,
                asset_b=asset_b,
                p_value=1.0,
                hedge_ratio=0.0,
                half_life=float("inf"),
                is_cointegrated=False,
            )

        a = combined.iloc[:, 0].values
        b = combined.iloc[:, 1].values

        stat, pvalue, crit = coint(a, b)

        # OLS hedge ratio
        hedge_ratio = self._ols_hedge_ratio(a, b)

        # Spread and half-life
        spread = a - hedge_ratio * b
        half_life = self._half_life(spread)

        crit_values = {}
        if crit is not None and len(crit) == 3:
            crit_values = {"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])}

        return CointegrationResult(
            asset_a=asset_a,
            asset_b=asset_b,
            p_value=float(pvalue),
            hedge_ratio=float(hedge_ratio),
            half_life=float(half_life),
            is_cointegrated=float(pvalue) < self.p_threshold,
            test_statistic=float(stat),
            critical_values=crit_values,
        )

    def scan_pairs(
        self,
        price_data: pd.DataFrame,
    ) -> PairsScanResult:
        """Scan all column pairs in a DataFrame for cointegration.

        Parameters
        ----------
        price_data : pd.DataFrame
            Columns = tickers, rows = dates, values = prices.
        """
        tickers = list(price_data.columns)
        n = len(tickers)
        results: list[CointegrationResult] = []

        for i in range(n):
            for j in range(i + 1, n):
                result = self.test_cointegration(
                    prices_a=price_data[tickers[i]],
                    prices_b=price_data[tickers[j]],
                    asset_a=tickers[i],
                    asset_b=tickers[j],
                )
                if result.is_cointegrated:
                    results.append(result)

        # Sort by p-value
        results.sort(key=lambda r: r.p_value)

        # Generate signals for cointegrated pairs
        signals: list[PairsSignal] = []
        for r in results:
            sig = self.generate_signal(
                prices_a=price_data[r.asset_a],
                prices_b=price_data[r.asset_b],
                coint_result=r,
            )
            signals.append(sig)

        return PairsScanResult(
            n_pairs_tested=n * (n - 1) // 2,
            n_cointegrated=len(results),
            pairs=results,
            signals=signals,
        )

    # ── Signal generation ─────────────────────────────────────────────────

    def generate_signal(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        coint_result: CointegrationResult,
    ) -> PairsSignal:
        """Generate z-score based trading signal for a pair."""
        combined = pd.concat([prices_a, prices_b], axis=1).dropna()
        if len(combined) < self.lookback:
            return PairsSignal(
                asset_a=coint_result.asset_a,
                asset_b=coint_result.asset_b,
                z_score=0.0,
                signal="flat",
                spread_value=0.0,
                hedge_ratio=coint_result.hedge_ratio,
                half_life=coint_result.half_life,
            )

        a = combined.iloc[:, 0].values
        b = combined.iloc[:, 1].values

        spread = a - coint_result.hedge_ratio * b
        recent = spread[-self.lookback:]

        mean = float(np.mean(recent))
        std = float(np.std(recent))
        if std < 1e-10:
            std = 1.0

        current_spread = float(spread[-1])
        z = (current_spread - mean) / std

        # Signal logic
        if z > self.entry_z:
            signal = "short_spread"  # spread too high, short A long B
        elif z < -self.entry_z:
            signal = "long_spread"   # spread too low, long A short B
        elif abs(z) < self.exit_z:
            signal = "flat"          # mean-reverted, close positions
        else:
            signal = "flat"          # in no-trade zone

        return PairsSignal(
            asset_a=coint_result.asset_a,
            asset_b=coint_result.asset_b,
            z_score=float(z),
            signal=signal,
            spread_value=current_spread,
            hedge_ratio=coint_result.hedge_ratio,
            half_life=coint_result.half_life,
            entry_z=self.entry_z,
            exit_z=self.exit_z,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _ols_hedge_ratio(a: np.ndarray, b: np.ndarray) -> float:
        """Simple OLS regression: a = beta * b + alpha."""
        b_with_const = np.column_stack([b, np.ones(len(b))])
        try:
            beta, _ = np.linalg.lstsq(b_with_const, a, rcond=None)[:2]
            return float(beta[0])
        except np.linalg.LinAlgError:
            return 1.0

    @staticmethod
    def _half_life(spread: np.ndarray) -> float:
        """Estimate half-life of mean reversion via AR(1)."""
        if len(spread) < 10:
            return float("inf")
        lag = spread[:-1]
        diff = np.diff(spread)

        # OLS: diff = beta * lag + alpha
        lag_with_const = np.column_stack([lag, np.ones(len(lag))])
        try:
            result = np.linalg.lstsq(lag_with_const, diff, rcond=None)
            beta = float(result[0][0])
        except np.linalg.LinAlgError:
            return float("inf")

        if beta >= 0:
            return float("inf")  # not mean-reverting

        return float(-np.log(2) / beta)
