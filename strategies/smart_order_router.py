"""Smart Order Router — TWAP, VWAP, and SOR algorithms.

Breaks large orders into slices to reduce market impact,
with time-weighted and volume-weighted scheduling.
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class SORAlgorithm(Enum):
    TWAP = "twap"    # Time-Weighted Average Price
    VWAP = "vwap"    # Volume-Weighted Average Price
    ICEBERG = "iceberg"  # Show only partial quantity


@dataclass
class OrderSlice:
    """A single child order in an algorithmic execution."""

    slice_id: int
    parent_id: str
    quantity: int
    scheduled_time: str
    status: str = "pending"  # "pending", "submitted", "filled", "cancelled"
    fill_price: float = 0.0
    fill_quantity: int = 0
    submitted_at: str = ""
    filled_at: str = ""


@dataclass
class AlgoOrderResult:
    """Result of an algorithmic order execution."""

    parent_id: str
    algorithm: str
    total_quantity: int
    filled_quantity: int
    avg_fill_price: float
    vwap_benchmark: float  # market VWAP for comparison
    slippage_bps: float    # execution cost vs benchmark in bps
    n_slices: int
    n_filled: int
    n_cancelled: int
    duration_seconds: float
    slices: list[OrderSlice] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def fill_rate(self) -> float:
        return self.filled_quantity / self.total_quantity if self.total_quantity > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "algorithm": self.algorithm,
            "total_qty": self.total_quantity,
            "filled_qty": self.filled_quantity,
            "avg_fill": round(self.avg_fill_price, 4),
            "slippage_bps": round(self.slippage_bps, 2),
            "fill_rate": round(self.fill_rate, 4),
            "n_slices": self.n_slices,
        }


# ---------------------------------------------------------------------------
# Smart Order Router
# ---------------------------------------------------------------------------

class SmartOrderRouter:
    """Algorithmic order execution with TWAP, VWAP, and iceberg support.

    Parameters
    ----------
    default_slices : int
        Default number of slices for TWAP/VWAP.
    min_slice_qty : int
        Minimum quantity per slice (won't split below this).
    max_participation_rate : float
        Maximum fraction of volume to consume per slice.
    """

    def __init__(
        self,
        default_slices: int = 10,
        min_slice_qty: int = 1,
        max_participation_rate: float = 0.10,
    ) -> None:
        self.default_slices = default_slices
        self.min_slice_qty = min_slice_qty
        self.max_participation_rate = max_participation_rate
        self._order_counter = 0

    # ── TWAP ──────────────────────────────────────────────────────────────

    def create_twap(
        self,
        total_quantity: int,
        duration_minutes: int = 60,
        n_slices: Optional[int] = None,
    ) -> list[OrderSlice]:
        """Create TWAP schedule — equal quantity at equal time intervals.

        Parameters
        ----------
        total_quantity : int
            Total shares/contracts to execute.
        duration_minutes : int
            Total execution window.
        n_slices : int, optional
            Number of slices (default: ``self.default_slices``).
        """
        n = n_slices or self.default_slices
        n = min(n, total_quantity // self.min_slice_qty) if total_quantity > 0 else 1
        n = max(n, 1)

        qty_per_slice = total_quantity // n
        remainder = total_quantity % n
        interval = duration_minutes / n

        self._order_counter += 1
        parent_id = f"TWAP-{self._order_counter:06d}"

        slices: list[OrderSlice] = []
        for i in range(n):
            qty = qty_per_slice + (1 if i < remainder else 0)
            minutes_offset = interval * i
            slices.append(OrderSlice(
                slice_id=i,
                parent_id=parent_id,
                quantity=qty,
                scheduled_time=f"+{minutes_offset:.1f}min",
            ))

        _log.info(
            "twap_created",
            parent_id=parent_id,
            total=total_quantity,
            n_slices=n,
            interval_min=round(interval, 1),
        )
        return slices

    # ── VWAP ──────────────────────────────────────────────────────────────

    def create_vwap(
        self,
        total_quantity: int,
        volume_profile: list[float],
        duration_minutes: int = 390,
    ) -> list[OrderSlice]:
        """Create VWAP schedule — quantity proportional to volume profile.

        Parameters
        ----------
        total_quantity : int
            Total to execute.
        volume_profile : list[float]
            Relative volume weights per time bucket (e.g., from
            historical intraday volume). Length determines n_slices.
        duration_minutes : int
            Trading session length (default 390 = 6.5h).
        """
        if not volume_profile:
            return self.create_twap(total_quantity, duration_minutes)

        # Normalize volume profile
        total_vol = sum(volume_profile)
        if total_vol < 1e-10:
            return self.create_twap(total_quantity, duration_minutes)

        normalized = [v / total_vol for v in volume_profile]
        n = len(normalized)
        interval = duration_minutes / n

        self._order_counter += 1
        parent_id = f"VWAP-{self._order_counter:06d}"

        slices: list[OrderSlice] = []
        allocated = 0
        for i in range(n):
            if i == n - 1:
                qty = total_quantity - allocated
            else:
                qty = max(self.min_slice_qty, round(total_quantity * normalized[i]))
                qty = min(qty, total_quantity - allocated)
            allocated += qty

            if qty > 0:
                slices.append(OrderSlice(
                    slice_id=i,
                    parent_id=parent_id,
                    quantity=qty,
                    scheduled_time=f"+{interval * i:.1f}min",
                ))

        _log.info(
            "vwap_created",
            parent_id=parent_id,
            total=total_quantity,
            n_slices=len(slices),
        )
        return slices

    # ── Iceberg ───────────────────────────────────────────────────────────

    def create_iceberg(
        self,
        total_quantity: int,
        show_quantity: int = 10,
    ) -> list[OrderSlice]:
        """Create iceberg order — show only ``show_quantity`` at a time.

        Parameters
        ----------
        total_quantity : int
            Total to execute.
        show_quantity : int
            Visible quantity per slice.
        """
        show_quantity = max(show_quantity, self.min_slice_qty)
        n = math.ceil(total_quantity / show_quantity)

        self._order_counter += 1
        parent_id = f"ICE-{self._order_counter:06d}"

        slices: list[OrderSlice] = []
        remaining = total_quantity
        for i in range(n):
            qty = min(show_quantity, remaining)
            remaining -= qty
            slices.append(OrderSlice(
                slice_id=i,
                parent_id=parent_id,
                quantity=qty,
                scheduled_time="on_fill",
            ))

        _log.info(
            "iceberg_created",
            parent_id=parent_id,
            total=total_quantity,
            show_qty=show_quantity,
            n_slices=n,
        )
        return slices

    # ── Execution simulation ──────────────────────────────────────────────

    def simulate_execution(
        self,
        slices: list[OrderSlice],
        market_prices: list[float],
        market_volumes: Optional[list[float]] = None,
    ) -> AlgoOrderResult:
        """Simulate filling slices against market data (for backtesting).

        Parameters
        ----------
        slices : list[OrderSlice]
            Scheduled slices from create_twap/vwap/iceberg.
        market_prices : list[float]
            Price at each slice time. Must have len ≥ len(slices).
        market_volumes : list[float], optional
            Volume at each slice time (for participation rate checks).
        """
        if len(market_prices) < len(slices):
            raise ValueError("market_prices must be ≥ len(slices)")

        parent_id = slices[0].parent_id if slices else "SIM"
        algo = parent_id.split("-")[0].lower()
        total_qty = sum(s.quantity for s in slices)

        filled_qty = 0
        cost = 0.0
        n_filled = 0
        n_cancelled = 0

        for i, s in enumerate(slices):
            price = market_prices[i]
            vol = market_volumes[i] if market_volumes else float("inf")

            # Participation rate check
            max_fill = int(vol * self.max_participation_rate) if vol != float("inf") else s.quantity
            fill = min(s.quantity, max_fill)

            if fill > 0:
                s.status = "filled"
                s.fill_price = price
                s.fill_quantity = fill
                filled_qty += fill
                cost += fill * price
                n_filled += 1
            else:
                s.status = "cancelled"
                n_cancelled += 1

        avg_fill = cost / filled_qty if filled_qty > 0 else 0.0

        # VWAP benchmark
        if market_volumes and sum(market_volumes[:len(slices)]) > 0:
            total_vol = sum(market_volumes[:len(slices)])
            vwap = sum(
                p * v for p, v in zip(market_prices[:len(slices)], market_volumes[:len(slices)])
            ) / total_vol
        else:
            vwap = sum(market_prices[:len(slices)]) / len(slices) if slices else 0.0

        slippage = ((avg_fill - vwap) / vwap * 10_000) if vwap > 0 else 0.0

        return AlgoOrderResult(
            parent_id=parent_id,
            algorithm=algo,
            total_quantity=total_qty,
            filled_quantity=filled_qty,
            avg_fill_price=avg_fill,
            vwap_benchmark=vwap,
            slippage_bps=slippage,
            n_slices=len(slices),
            n_filled=n_filled,
            n_cancelled=n_cancelled,
            duration_seconds=0.0,
            slices=slices,
        )

    # ── Default volume profile ────────────────────────────────────────────

    @staticmethod
    def us_equity_volume_profile() -> list[float]:
        """Typical U-shaped US equity intraday volume (13 × 30min buckets).

        Higher at open and close, lower midday.
        """
        return [
            12.0,  # 09:30-10:00
            9.0,   # 10:00-10:30
            7.5,   # 10:30-11:00
            6.5,   # 11:00-11:30
            6.0,   # 11:30-12:00
            5.5,   # 12:00-12:30
            5.5,   # 12:30-13:00
            6.0,   # 13:00-13:30
            6.5,   # 13:30-14:00
            7.0,   # 14:00-14:30
            7.5,   # 14:30-15:00
            9.0,   # 15:00-15:30
            12.0,  # 15:30-16:00
        ]
