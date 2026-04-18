"""Dynamic Correlation Tracker — EWM rolling correlation + contagion detection.

Replaces the static 11×11 Cholesky matrix in the Monte Carlo engine with
a live, time-varying correlation system that detects regime shifts and
contagion spill-overs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContagionAlert:
    """Fired when cross-asset correlations spike beyond normal."""

    timestamp: str
    asset_a: str
    asset_b: str
    current_corr: float
    baseline_corr: float
    z_score: float
    severity: str  # "watch", "warning", "critical"
    message: str


@dataclass
class CorrelationSnapshot:
    """Point-in-time state of the dynamic correlation matrix."""

    timestamp: str
    correlation_matrix: pd.DataFrame
    eigenvalues: list[float]
    effective_n_assets: float  # 1/Σλ_i² — how many "independent" bets
    absorption_ratio: float  # % of variance explained by top eigenvalues
    contagion_alerts: list[ContagionAlert] = field(default_factory=list)
    regime: str = "normal"  # "normal", "decorrelating", "contagion"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "effective_n_assets": round(self.effective_n_assets, 2),
            "absorption_ratio": round(self.absorption_ratio, 4),
            "regime": self.regime,
            "n_alerts": len(self.contagion_alerts),
        }


# ---------------------------------------------------------------------------
# Correlation Tracker
# ---------------------------------------------------------------------------

class CorrelationTracker:
    """Real-time EWM rolling correlation with contagion detection.

    Parameters
    ----------
    halflife : int
        EWM half-life in trading days (default 21 = ~1 month).
    lookback : int
        Maximum lookback window for returns.
    contagion_z_threshold : float
        Z-score above which a correlation spike generates an alert.
    absorption_warning : float
        Absorption ratio threshold for regime = "contagion".
    """

    def __init__(
        self,
        halflife: int = 21,
        lookback: int = 252,
        contagion_z_threshold: float = 2.0,
        absorption_warning: float = 0.80,
    ) -> None:
        self.halflife = halflife
        self.lookback = lookback
        self.contagion_z_threshold = contagion_z_threshold
        self.absorption_warning = absorption_warning
        self._history: Optional[pd.DataFrame] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, prices: pd.DataFrame) -> CorrelationSnapshot:
        """Compute the latest correlation snapshot from price data.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily close prices (columns = tickers, rows = dates).
            Must be sorted by date ascending with ≥ 30 rows.
        """
        if prices.shape[0] < 30:
            raise ValueError("Need ≥30 price observations")

        self._history = prices.tail(self.lookback).copy()
        returns = self._history.pct_change().dropna()

        corr_ewm = self._ewm_correlation(returns)
        corr_long = returns.corr()

        eigenvalues = self._eigenvalues(corr_ewm)
        eff_n = self._effective_n(eigenvalues)
        absorption = self._absorption_ratio(eigenvalues)

        alerts = self._detect_contagion(corr_ewm, corr_long, returns)

        if absorption > self.absorption_warning:
            regime = "contagion"
        elif absorption < 0.40:
            regime = "decorrelating"
        else:
            regime = "normal"

        ts = str(prices.index[-1]) if hasattr(prices.index[-1], 'isoformat') else str(prices.index[-1])

        snap = CorrelationSnapshot(
            timestamp=ts,
            correlation_matrix=corr_ewm,
            eigenvalues=eigenvalues,
            effective_n_assets=eff_n,
            absorption_ratio=absorption,
            contagion_alerts=alerts,
            regime=regime,
        )
        _log.info(
            "correlation_updated",
            regime=regime,
            absorption=round(absorption, 3),
            eff_n=round(eff_n, 1),
            n_alerts=len(alerts),
        )
        return snap

    def get_ewm_covariance(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return EWM covariance matrix (annualized) for use in optimization."""
        returns = prices.pct_change().dropna().tail(self.lookback)
        return self._ewm_covariance(returns) * 252

    # ── EWM calculations ──────────────────────────────────────────────────

    def _ewm_correlation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Exponentially-weighted correlation matrix."""
        ewm_cov = self._ewm_covariance(returns)
        d = np.sqrt(np.diag(ewm_cov.values))
        d = np.where(d > 1e-12, d, 1e-12)
        D_inv = np.diag(1.0 / d)
        corr = D_inv @ ewm_cov.values @ D_inv
        np.fill_diagonal(corr, 1.0)
        return pd.DataFrame(corr, index=returns.columns, columns=returns.columns)

    def _ewm_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Exponentially-weighted covariance matrix."""
        n = len(returns)
        alpha = 1 - np.exp(-np.log(2) / self.halflife)
        weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        demeaned = returns - returns.mean()
        weighted = demeaned.multiply(np.sqrt(weights), axis=0)
        cov = weighted.T @ weighted
        return cov

    # ── Eigenvalue analysis ───────────────────────────────────────────────

    @staticmethod
    def _eigenvalues(corr: pd.DataFrame) -> list[float]:
        evals = np.linalg.eigvalsh(corr.values)
        return sorted([float(e) for e in evals], reverse=True)

    @staticmethod
    def _effective_n(eigenvalues: list[float]) -> float:
        """Effective number of independent risk factors."""
        total = sum(eigenvalues)
        if total < 1e-12:
            return 1.0
        normalized = [e / total for e in eigenvalues]
        hhi = sum(p ** 2 for p in normalized)
        return 1.0 / hhi if hhi > 1e-12 else 1.0

    @staticmethod
    def _absorption_ratio(eigenvalues: list[float], n_top: int = 3) -> float:
        """Fraction of total variance captured by top eigenvalues."""
        total = sum(eigenvalues)
        if total < 1e-12:
            return 0.0
        top_sum = sum(eigenvalues[:n_top])
        return top_sum / total

    # ── Contagion detection ───────────────────────────────────────────────

    def _detect_contagion(
        self,
        corr_short: pd.DataFrame,
        corr_long: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> list[ContagionAlert]:
        """Compare short-term (EWM) vs long-term correlation to find spikes."""
        alerts: list[ContagionAlert] = []
        tickers = list(corr_short.columns)
        n = len(tickers)

        # Estimate std of correlation under rolling windows
        rolling_corrs: list[pd.DataFrame] = []
        window = min(63, len(returns) // 2)
        if window < 20:
            return alerts  # not enough data for rolling stats

        for start in range(0, len(returns) - window + 1, 5):
            chunk = returns.iloc[start : start + window]
            rolling_corrs.append(chunk.corr())

        for i in range(n):
            for j in range(i + 1, n):
                a, b = tickers[i], tickers[j]
                current = corr_short.iloc[i, j]
                baseline = corr_long.iloc[i, j]

                # Collect historical correlations for this pair
                pair_hist = [rc.iloc[i, j] for rc in rolling_corrs]
                if len(pair_hist) < 5:
                    continue
                std = float(np.std(pair_hist))
                if std < 1e-6:
                    continue

                z = (current - baseline) / std

                if abs(z) >= self.contagion_z_threshold:
                    severity = "critical" if abs(z) > 3.0 else "warning" if abs(z) > 2.5 else "watch"
                    alerts.append(ContagionAlert(
                        timestamp=str(returns.index[-1]),
                        asset_a=a,
                        asset_b=b,
                        current_corr=round(float(current), 4),
                        baseline_corr=round(float(baseline), 4),
                        z_score=round(float(z), 2),
                        severity=severity,
                        message=f"{a}/{b} correlation spiked to {current:.2f} (baseline {baseline:.2f}, z={z:.1f})",
                    ))

        alerts.sort(key=lambda a: abs(a.z_score), reverse=True)
        return alerts
