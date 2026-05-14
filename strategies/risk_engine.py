"""strategies/risk_engine.py — Sprint 3.1 / 3.3 / 3.4.

Position exposure calculator + pre-trade size enforcement + stop/TP alerts.

Responsibilities
----------------
* ExposureReport  — snapshot of portfolio risk at a point in time
* ExposureCalculator — builds ExposureReport from PositionSnapshot list
* Pre-trade gate  — check_new_position() blocks orders that would breach limits
* Stop/TP alerts  — scan for positions that have hit stop-loss or take-profit

Rules (all tunable via RiskConfig)
-----------------------------------
* MAX_SINGLE_POSITION_PCT   = 15%  of account — no single name >15%
* MAX_TOTAL_EXPOSURE_PCT    = 80%  of account — don't deploy >80% at once
* MAX_CONTRACTS_PER_POSITION = 20  (from ROLL_DISCIPLINE)
* STOP_LOSS_PCT             = -50% unrealized P&L on cost (options decay fast)
* TAKE_PROFIT_PCT           = +100% unrealized P&L on cost
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import structlog

_log = structlog.get_logger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Tunable risk limits. Defaults match War Room ROLL_DISCIPLINE + doctrine."""
    max_single_position_pct: float = 0.15   # 15 % of account per name
    max_total_exposure_pct: float = 0.80    # 80 % of account deployed at most
    max_contracts_per_position: int = 20    # from ROLL_DISCIPLINE
    stop_loss_pct: float = -0.50            # -50 % unrealised → flag for close
    take_profit_pct: float = 1.00           # +100 % unrealised → flag for close
    max_daily_loss_pct: float = 0.05        # 5 % of account as daily loss ceiling


# ── Per-position output ───────────────────────────────────────────────────────

@dataclass
class PositionRisk:
    """Risk metrics for one position."""
    symbol: str
    sec_type: str
    market_value: float
    unrealized_pnl: float
    cost_basis: float          # abs(quantity) * avg_cost * multiplier
    pnl_pct_of_cost: float     # unrealised / cost_basis
    account_pct: float         # abs(market_value) / account_value
    quantity: float
    is_long: bool
    # Alerts
    stop_triggered: bool = False
    take_profit_triggered: bool = False
    oversized: bool = False     # exceeds max_single_position_pct
    oversized_contracts: bool = False  # exceeds max_contracts_per_position

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "sec_type": self.sec_type,
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "cost_basis": round(self.cost_basis, 2),
            "pnl_pct_of_cost": round(self.pnl_pct_of_cost * 100, 1),
            "account_pct": round(self.account_pct * 100, 1),
            "quantity": self.quantity,
            "is_long": self.is_long,
            "stop_triggered": self.stop_triggered,
            "take_profit_triggered": self.take_profit_triggered,
            "oversized": self.oversized,
            "oversized_contracts": self.oversized_contracts,
        }


# ── Portfolio-level output ────────────────────────────────────────────────────

@dataclass
class ExposureReport:
    """Snapshot of full portfolio risk."""
    account_value_usd: float
    total_exposure_usd: float
    total_exposure_pct: float       # total_exposure / account_value
    total_unrealized_pnl: float
    position_count: int
    positions: List[PositionRisk]
    alerts: List[str]               # human-readable warnings
    exposure_ok: bool               # True when within all limits
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "account_value_usd": round(self.account_value_usd, 2),
            "total_exposure_usd": round(self.total_exposure_usd, 2),
            "total_exposure_pct": round(self.total_exposure_pct * 100, 1),
            "total_unrealized_pnl": round(self.total_unrealized_pnl, 2),
            "position_count": self.position_count,
            "exposure_ok": self.exposure_ok,
            "alerts": self.alerts,
            "positions": [p.to_dict() for p in self.positions],
            "generated_at": self.generated_at,
        }


# ── Calculator ────────────────────────────────────────────────────────────────

class ExposureCalculator:
    """Builds an ExposureReport from a list of PositionSnapshot objects.

    Usage::

        calc = ExposureCalculator()
        report = calc.calculate(positions, account_value_usd=50_000)
        if not report.exposure_ok:
            for alert in report.alerts:
                print(alert)
    """

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig()

    def calculate(self, positions: list, account_value_usd: float) -> ExposureReport:
        """Build a full ExposureReport.  Never raises.

        Args:
            positions: List[PositionSnapshot] — from PositionTracker.all()
            account_value_usd: Current account net liquidation value.
        """
        if account_value_usd <= 0:
            account_value_usd = 1.0   # avoid division-by-zero; alerts will fire

        position_risks: List[PositionRisk] = []
        total_exposure = 0.0
        total_upnl = 0.0
        alerts: List[str] = []

        for pos in positions:
            pr = self._evaluate_position(pos, account_value_usd)
            position_risks.append(pr)
            total_exposure += abs(pos.market_value)
            total_upnl += pos.unrealized_pnl

            if pr.stop_triggered:
                alerts.append(
                    f"STOP: {pos.symbol} unrealised P&L {pr.pnl_pct_of_cost:.0%} — "
                    f"consider closing (stop at {self.config.stop_loss_pct:.0%})"
                )
            if pr.take_profit_triggered:
                alerts.append(
                    f"TP: {pos.symbol} unrealised P&L {pr.pnl_pct_of_cost:.0%} — "
                    f"consider harvesting (target {self.config.take_profit_pct:.0%})"
                )
            if pr.oversized:
                alerts.append(
                    f"SIZE: {pos.symbol} is {pr.account_pct:.0%} of account "
                    f"(limit {self.config.max_single_position_pct:.0%})"
                )
            if pr.oversized_contracts:
                alerts.append(
                    f"CONTRACTS: {pos.symbol} has {abs(pos.quantity):.0f} contracts "
                    f"(limit {self.config.max_contracts_per_position})"
                )

        total_exp_pct = total_exposure / account_value_usd

        if total_exp_pct > self.config.max_total_exposure_pct:
            alerts.append(
                f"OVEREXPOSED: {total_exp_pct:.0%} of account deployed "
                f"(limit {self.config.max_total_exposure_pct:.0%})"
            )

        # Daily loss guard (unrealised proxy — realised not tracked here)
        daily_loss_limit = account_value_usd * self.config.max_daily_loss_pct
        if total_upnl < -daily_loss_limit:
            alerts.append(
                f"DAILY LOSS: unrealised P&L ${total_upnl:,.0f} exceeds daily limit "
                f"${-daily_loss_limit:,.0f}"
            )

        exposure_ok = len(alerts) == 0

        return ExposureReport(
            account_value_usd=account_value_usd,
            total_exposure_usd=total_exposure,
            total_exposure_pct=total_exp_pct,
            total_unrealized_pnl=total_upnl,
            position_count=len(positions),
            positions=position_risks,
            alerts=alerts,
            exposure_ok=exposure_ok,
        )

    def check_new_position(
        self,
        ticker: str,
        size_fraction: float,          # fraction of account to deploy (0–1)
        account_value_usd: float,
        current_positions: list,
        contracts: int = 1,
    ) -> tuple[bool, str]:
        """Pre-trade gate: would opening this position breach any limit?

        Returns (approved: bool, reason: str).
        """
        if account_value_usd <= 0:
            return False, "account value unknown"

        order_value = account_value_usd * size_fraction

        # Single-name limit
        if size_fraction > self.config.max_single_position_pct:
            return False, (
                f"{ticker}: size {size_fraction:.1%} > max {self.config.max_single_position_pct:.1%}"
            )

        # Contracts limit
        if contracts > self.config.max_contracts_per_position:
            return False, (
                f"{ticker}: {contracts} contracts > max {self.config.max_contracts_per_position}"
            )

        # Total exposure after this order
        current_exposure = sum(abs(p.market_value) for p in current_positions)
        new_exposure_pct = (current_exposure + order_value) / account_value_usd
        if new_exposure_pct > self.config.max_total_exposure_pct:
            return False, (
                f"Adding {ticker} would bring total exposure to {new_exposure_pct:.1%} "
                f"(limit {self.config.max_total_exposure_pct:.1%})"
            )

        return True, ""

    # ── internal ──────────────────────────────────────────────────────────────

    def _evaluate_position(self, pos, account_value_usd: float) -> PositionRisk:
        """Compute PositionRisk for a single PositionSnapshot."""
        multiplier = getattr(pos, "multiplier", 100) or 100
        qty = abs(pos.quantity)
        cost_basis = qty * pos.avg_cost * multiplier if pos.avg_cost else abs(pos.market_value)

        pnl_pct = (pos.unrealized_pnl / cost_basis) if cost_basis > 0 else 0.0
        acct_pct = abs(pos.market_value) / account_value_usd

        return PositionRisk(
            symbol=pos.symbol,
            sec_type=pos.sec_type,
            market_value=pos.market_value,
            unrealized_pnl=pos.unrealized_pnl,
            cost_basis=cost_basis,
            pnl_pct_of_cost=pnl_pct,
            account_pct=acct_pct,
            quantity=pos.quantity,
            is_long=pos.quantity > 0,
            stop_triggered=pnl_pct < self.config.stop_loss_pct,
            take_profit_triggered=pnl_pct > self.config.take_profit_pct,
            oversized=acct_pct > self.config.max_single_position_pct,
            oversized_contracts=(
                pos.sec_type == "OPT" and qty > self.config.max_contracts_per_position
            ),
        )
