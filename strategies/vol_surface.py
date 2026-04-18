"""Volatility Surface & Term Structure — AAC v3.6.0

Constructs implied volatility surfaces from options chains, computes
skew metrics and term structure, classifies vol regimes.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VolPoint:
    """Single point on the vol surface."""

    strike: float
    expiry_days: int          # DTE
    implied_vol: float
    option_type: str          # "call" or "put"
    moneyness: float = 0.0   # strike / spot
    delta: float = 0.0


@dataclass
class SkewMetrics:
    """Skew and smile metrics for a single expiry."""

    expiry_days: int
    atm_vol: float
    rr_25d: float          # 25-delta risk reversal (call - put IV)
    bf_25d: float          # 25-delta butterfly (0.5*(call+put) - atm)
    skew_slope: float      # linear slope of IV vs moneyness
    smile_curvature: float # quadratic coefficient


@dataclass
class TermStructurePoint:
    """ATM vol at a given expiry."""

    expiry_days: int
    atm_vol: float


@dataclass
class VolSurfaceSnapshot:
    """Complete vol surface analysis."""

    ticker: str
    spot_price: float
    timestamp: str
    points: list[VolPoint] = field(default_factory=list)
    skew_by_expiry: list[SkewMetrics] = field(default_factory=list)
    term_structure: list[TermStructurePoint] = field(default_factory=list)
    vol_regime: str = "normal"       # low / normal / elevated / crisis
    surface_quality: float = 0.0     # number of valid points

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "spot_price": round(self.spot_price, 2),
            "vol_regime": self.vol_regime,
            "n_points": len(self.points),
            "term_structure": [
                {"dte": t.expiry_days, "atm_vol": round(t.atm_vol, 4)}
                for t in self.term_structure
            ],
            "skew_by_expiry": [
                {
                    "dte": s.expiry_days,
                    "atm_vol": round(s.atm_vol, 4),
                    "rr_25d": round(s.rr_25d, 4),
                    "bf_25d": round(s.bf_25d, 4),
                }
                for s in self.skew_by_expiry
            ],
        }


# ---------------------------------------------------------------------------
# Vol Surface Builder
# ---------------------------------------------------------------------------

class VolSurfaceBuilder:
    """Constructs vol surface from options chain data (e.g., yfinance).

    Parameters
    ----------
    min_dte : int
        Minimum DTE to include.
    max_dte : int
        Maximum DTE to include.
    moneyness_range : tuple
        (min, max) moneyness range — e.g. (0.80, 1.20) for ±20%.
    """

    def __init__(
        self,
        min_dte: int = 7,
        max_dte: int = 365,
        moneyness_range: tuple[float, float] = (0.80, 1.20),
    ) -> None:
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.moneyness_range = moneyness_range

    def build_from_chains(
        self,
        ticker: str,
        spot_price: float,
        chains: list[dict[str, Any]],
        timestamp: str = "",
    ) -> VolSurfaceSnapshot:
        """Build vol surface from a list of options chain dicts.

        Parameters
        ----------
        chains : list[dict]
            Each dict has keys: strike, dte, impliedVolatility, optionType, delta (optional).
        """
        points: list[VolPoint] = []

        for c in chains:
            strike = float(c.get("strike", 0))
            dte = int(c.get("dte", c.get("daysToExpiration", 0)))
            iv = float(c.get("impliedVolatility", c.get("iv", 0)))
            opt_type = str(c.get("optionType", c.get("option_type", "call"))).lower()
            delta = float(c.get("delta", 0))

            if iv <= 0 or strike <= 0 or spot_price <= 0:
                continue
            if dte < self.min_dte or dte > self.max_dte:
                continue

            moneyness = strike / spot_price
            if moneyness < self.moneyness_range[0] or moneyness > self.moneyness_range[1]:
                continue

            points.append(VolPoint(
                strike=strike,
                expiry_days=dte,
                implied_vol=iv,
                option_type=opt_type,
                moneyness=moneyness,
                delta=delta,
            ))

        # Group by expiry
        expiries: dict[int, list[VolPoint]] = {}
        for p in points:
            expiries.setdefault(p.expiry_days, []).append(p)

        # Compute per-expiry metrics
        skew_list: list[SkewMetrics] = []
        term_list: list[TermStructurePoint] = []

        for dte in sorted(expiries.keys()):
            pts = expiries[dte]
            metrics = self._compute_skew(pts, spot_price, dte)
            if metrics is not None:
                skew_list.append(metrics)
                term_list.append(TermStructurePoint(
                    expiry_days=dte,
                    atm_vol=metrics.atm_vol,
                ))

        # Classify regime from front-month ATM vol
        vol_regime = self._classify_regime(term_list)

        return VolSurfaceSnapshot(
            ticker=ticker,
            spot_price=spot_price,
            timestamp=timestamp,
            points=points,
            skew_by_expiry=skew_list,
            term_structure=term_list,
            vol_regime=vol_regime,
            surface_quality=len(points),
        )

    def _compute_skew(
        self,
        points: list[VolPoint],
        spot: float,
        dte: int,
    ) -> Optional[SkewMetrics]:
        """Compute skew metrics for a single expiry slice."""
        if len(points) < 3:
            return None

        # Find ATM vol (closest to spot)
        sorted_pts = sorted(points, key=lambda p: abs(p.moneyness - 1.0))
        atm_vol = sorted_pts[0].implied_vol

        # Separate OTM puts (moneyness < 1) and OTM calls (moneyness > 1)
        puts = [p for p in points if p.moneyness < 0.98]
        calls = [p for p in points if p.moneyness > 1.02]

        # Approximate 25-delta wings
        put_25d_vol = self._wing_vol(puts, target_moneyness=0.95) or atm_vol
        call_25d_vol = self._wing_vol(calls, target_moneyness=1.05) or atm_vol

        rr_25d = call_25d_vol - put_25d_vol
        bf_25d = 0.5 * (call_25d_vol + put_25d_vol) - atm_vol

        # Linear skew slope: fit IV vs moneyness
        moneyness_arr = np.array([p.moneyness for p in points])
        iv_arr = np.array([p.implied_vol for p in points])

        slope = 0.0
        curvature = 0.0
        if len(points) >= 3:
            try:
                coeffs = np.polyfit(moneyness_arr, iv_arr, min(2, len(points) - 1))
                if len(coeffs) == 3:
                    curvature = float(coeffs[0])
                    slope = float(coeffs[1])
                elif len(coeffs) == 2:
                    slope = float(coeffs[0])
            except (np.linalg.LinAlgError, ValueError):
                pass

        return SkewMetrics(
            expiry_days=dte,
            atm_vol=atm_vol,
            rr_25d=rr_25d,
            bf_25d=bf_25d,
            skew_slope=slope,
            smile_curvature=curvature,
        )

    @staticmethod
    def _wing_vol(
        pts: list[VolPoint],
        target_moneyness: float,
    ) -> Optional[float]:
        """Find IV closest to a target moneyness."""
        if not pts:
            return None
        closest = min(pts, key=lambda p: abs(p.moneyness - target_moneyness))
        return closest.implied_vol

    @staticmethod
    def _classify_regime(term: list[TermStructurePoint]) -> str:
        """Classify vol regime from front-month ATM vol."""
        if not term:
            return "normal"
        front_vol = term[0].atm_vol
        if front_vol < 0.12:
            return "low"
        if front_vol < 0.20:
            return "normal"
        if front_vol < 0.35:
            return "elevated"
        return "crisis"
