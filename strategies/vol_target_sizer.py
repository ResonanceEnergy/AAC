"""Volatility-Targeting Position Sizing — AAC v3.6.0

Target-volatility framework: sizes positions so portfolio volatility
stays at a target level (e.g., 15% annualized). Includes inverse-vol
weighting and vol-adjusted Kelly.
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
class VolTargetAllocation:
    """Position sizing result from vol-targeting."""

    asset: str
    raw_weight: float           # unconstrained weight
    vol_target_weight: float    # adjusted to hit target vol
    realized_vol: float         # current annualized vol
    notional_size: float = 0.0  # in dollars
    leverage: float = 1.0       # effective leverage


@dataclass
class VolTargetResult:
    """Complete vol-targeting result for portfolio."""

    target_vol: float
    portfolio_vol: float        # estimated portfolio vol
    scaling_factor: float       # multiplier to hit target
    allocations: list[VolTargetAllocation] = field(default_factory=list)
    total_leverage: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_vol": round(self.target_vol, 4),
            "portfolio_vol": round(self.portfolio_vol, 4),
            "scaling_factor": round(self.scaling_factor, 3),
            "total_leverage": round(self.total_leverage, 3),
            "allocations": [
                {
                    "asset": a.asset,
                    "weight": round(a.vol_target_weight, 4),
                    "vol": round(a.realized_vol, 4),
                }
                for a in self.allocations
            ],
        }


# ---------------------------------------------------------------------------
# Vol Target Sizer
# ---------------------------------------------------------------------------

class VolTargetSizer:
    """Position sizing based on volatility targeting.

    Parameters
    ----------
    target_vol : float
        Annualized target portfolio volatility (e.g. 0.15 for 15%).
    max_leverage : float
        Maximum total leverage allowed.
    vol_lookback : int
        Days of history for realized vol calculation.
    vol_floor : float
        Minimum vol estimate to prevent division by near-zero.
    """

    def __init__(
        self,
        target_vol: float = 0.15,
        max_leverage: float = 2.0,
        vol_lookback: int = 63,
        vol_floor: float = 0.02,
    ) -> None:
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_lookback = vol_lookback
        self.vol_floor = vol_floor

    # ── Inverse-vol weighting ─────────────────────────────────────────────

    def inverse_vol_weights(
        self,
        returns: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute inverse-volatility weights: w_i ∝ 1/σ_i.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns, columns = assets.
        """
        recent = returns.iloc[-self.vol_lookback:]
        vols = recent.std() * np.sqrt(252)
        vols = vols.clip(lower=self.vol_floor)

        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()

        return {col: float(weights[col]) for col in returns.columns}

    # ── Vol-target scaling ────────────────────────────────────────────────

    def size_portfolio(
        self,
        returns: pd.DataFrame,
        base_weights: Optional[dict[str, float]] = None,
        capital: float = 100_000.0,
    ) -> VolTargetResult:
        """Scale portfolio weights to hit target volatility.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns, columns = assets.
        base_weights : dict, optional
            Starting weights. If None, uses inverse-vol weights.
        capital : float
            Total capital for notional sizing.
        """
        if base_weights is None:
            base_weights = self.inverse_vol_weights(returns)

        assets = list(base_weights.keys())
        w = np.array([base_weights.get(a, 0.0) for a in assets])

        # Current realized vol
        recent = returns[assets].iloc[-self.vol_lookback:]
        cov = recent.cov() * 252
        port_vol = float(np.sqrt(w @ cov.values @ w))
        port_vol = max(port_vol, self.vol_floor)

        # Scale factor
        scaling = self.target_vol / port_vol
        scaling = min(scaling, self.max_leverage)

        # Build allocations
        allocations: list[VolTargetAllocation] = []
        asset_vols = recent.std() * np.sqrt(252)

        for i, asset in enumerate(assets):
            raw_w = float(w[i])
            adj_w = raw_w * scaling
            vol = max(float(asset_vols[asset]), self.vol_floor)

            allocations.append(VolTargetAllocation(
                asset=asset,
                raw_weight=raw_w,
                vol_target_weight=adj_w,
                realized_vol=vol,
                notional_size=adj_w * capital,
                leverage=adj_w / raw_w if raw_w > 1e-10 else 1.0,
            ))

        total_lev = sum(abs(a.vol_target_weight) for a in allocations)

        return VolTargetResult(
            target_vol=self.target_vol,
            portfolio_vol=port_vol,
            scaling_factor=scaling,
            allocations=allocations,
            total_leverage=total_lev,
        )

    # ── Vol-adjusted Kelly ────────────────────────────────────────────────

    def vol_adjusted_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_vol: float,
        historical_vol: float = 0.15,
    ) -> float:
        """Kelly fraction scaled by vol ratio.

        Parameters
        ----------
        win_rate : float
            Win probability (0..1).
        avg_win : float
            Average win in dollars or %.
        avg_loss : float
            Average loss (positive number).
        current_vol : float
            Current annualized vol.
        historical_vol : float
            Long-term average vol that Kelly was calibrated on.
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0

        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b

        # Scale by vol ratio: if vol is 2x normal, size 0.5x
        vol_ratio = historical_vol / max(current_vol, self.vol_floor)
        adjusted = kelly * vol_ratio

        # Clamp to [0, max_leverage]
        return max(0.0, min(adjusted, self.max_leverage))

    # ── Single-asset vol target ───────────────────────────────────────────

    def target_size_single(
        self,
        returns: pd.Series,
        capital: float = 100_000.0,
        asset_name: str = "asset",
    ) -> VolTargetAllocation:
        """Size a single asset position to target vol."""
        recent = returns.iloc[-self.vol_lookback:]
        ann_vol = float(recent.std() * np.sqrt(252))
        ann_vol = max(ann_vol, self.vol_floor)

        weight = self.target_vol / ann_vol
        weight = min(weight, self.max_leverage)

        return VolTargetAllocation(
            asset=asset_name,
            raw_weight=1.0,
            vol_target_weight=weight,
            realized_vol=ann_vol,
            notional_size=weight * capital,
            leverage=weight,
        )
