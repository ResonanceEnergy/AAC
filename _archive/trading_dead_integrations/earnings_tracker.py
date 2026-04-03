"""
Earnings Tracker — BARREN WUFFET Portfolio P&L
===============================================

Tracks portfolio earnings, P&L snapshots, and generates
periodic reports. Integrates with paper trading and live
execution pipelines.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PnLSnapshot:
    """Point-in-time P&L record."""
    timestamp: str
    portfolio_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    trade_count: int
    win_rate: float  # 0-100
    best_trade: float
    worst_trade: float
    notes: str = ""


@dataclass
class EarningsRecord:
    """Daily earnings summary."""
    date: str
    gross_pnl: float
    fees: float
    net_pnl: float
    trades_executed: int
    strategies_active: int
    portfolio_value_eod: float


class EarningsTracker:
    """
    Track and persist earnings across all trading strategies.

    Usage:
        tracker = EarningsTracker()
        tracker.record_trade(pnl=150.0, strategy="bw-arb-dex")
        tracker.take_snapshot()
        report = tracker.daily_report()
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = data_dir or Path("data/earnings")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._trades_today: List[Dict[str, Any]] = []
        self._snapshots: List[PnLSnapshot] = []
        self._daily_records: List[EarningsRecord] = []

        # Running totals
        self.total_realized_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.portfolio_value: float = 10_000.0  # Default paper balance
        self.win_count: int = 0
        self.loss_count: int = 0
        self.best_trade: float = 0.0
        self.worst_trade: float = 0.0

        self._load_state()

    # ── Trade Recording ────────────────────────────────────────────────

    def record_trade(
        self,
        pnl: float,
        strategy: str = "unknown",
        fees: float = 0.0,
        notes: str = "",
    ) -> None:
        """Record a completed trade."""
        self._trades_today.append({
            "timestamp": datetime.now().isoformat(),
            "pnl": pnl,
            "fees": fees,
            "strategy": strategy,
            "notes": notes,
        })

        net = pnl - fees
        self.total_realized_pnl += net
        self.total_fees += fees
        self.portfolio_value += net

        if pnl > 0:
            self.win_count += 1
        elif pnl < 0:
            self.loss_count += 1

        self.best_trade = max(self.best_trade, pnl)
        self.worst_trade = min(self.worst_trade, pnl)

        logger.info(f"Trade recorded: {strategy} PnL=${pnl:.2f} fees=${fees:.2f}")

    # ── Snapshots ──────────────────────────────────────────────────────

    def take_snapshot(self, unrealized_pnl: float = 0.0) -> PnLSnapshot:
        """Take a point-in-time P&L snapshot."""
        total = self.win_count + self.loss_count
        snap = PnLSnapshot(
            timestamp=datetime.now().isoformat(),
            portfolio_value=round(self.portfolio_value, 2),
            unrealized_pnl=round(unrealized_pnl, 2),
            realized_pnl=round(self.total_realized_pnl, 2),
            total_pnl=round(self.total_realized_pnl + unrealized_pnl, 2),
            trade_count=total,
            win_rate=round((self.win_count / total * 100) if total > 0 else 0, 1),
            best_trade=round(self.best_trade, 2),
            worst_trade=round(self.worst_trade, 2),
        )
        self._snapshots.append(snap)
        return snap

    # ── Reports ────────────────────────────────────────────────────────

    def daily_report(self) -> EarningsRecord:
        """Generate today's earnings report."""
        today_pnl = sum(t["pnl"] for t in self._trades_today)
        today_fees = sum(t["fees"] for t in self._trades_today)
        strategies = set(t["strategy"] for t in self._trades_today)

        record = EarningsRecord(
            date=datetime.now().strftime("%Y-%m-%d"),
            gross_pnl=round(today_pnl, 2),
            fees=round(today_fees, 2),
            net_pnl=round(today_pnl - today_fees, 2),
            trades_executed=len(self._trades_today),
            strategies_active=len(strategies),
            portfolio_value_eod=round(self.portfolio_value, 2),
        )
        self._daily_records.append(record)
        return record

    def get_summary(self) -> Dict[str, Any]:
        """Get overall earnings summary."""
        total = self.win_count + self.loss_count
        return {
            "portfolio_value": round(self.portfolio_value, 2),
            "total_realized_pnl": round(self.total_realized_pnl, 2),
            "total_fees": round(self.total_fees, 2),
            "total_trades": total,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": round((self.win_count / total * 100) if total > 0 else 0, 1),
            "best_trade": round(self.best_trade, 2),
            "worst_trade": round(self.worst_trade, 2),
            "snapshots": len(self._snapshots),
            "daily_records": len(self._daily_records),
        }

    # ── Persistence ────────────────────────────────────────────────────

    def save_state(self) -> None:
        """Persist current state to JSON."""
        state = {
            "portfolio_value": self.portfolio_value,
            "total_realized_pnl": self.total_realized_pnl,
            "total_fees": self.total_fees,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "snapshots": [asdict(s) for s in self._snapshots[-100:]],
            "daily_records": [asdict(r) for r in self._daily_records[-365:]],
        }
        state_file = self.data_dir / "earnings_state.json"
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.info(f"Earnings state saved to {state_file}")

    def _load_state(self) -> None:
        """Load persisted state if available."""
        state_file = self.data_dir / "earnings_state.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                self.portfolio_value = state.get("portfolio_value", 10_000.0)
                self.total_realized_pnl = state.get("total_realized_pnl", 0.0)
                self.total_fees = state.get("total_fees", 0.0)
                self.win_count = state.get("win_count", 0)
                self.loss_count = state.get("loss_count", 0)
                self.best_trade = state.get("best_trade", 0.0)
                self.worst_trade = state.get("worst_trade", 0.0)
                logger.info(f"Loaded earnings state: ${self.portfolio_value:.2f}")
            except Exception as e:
                logger.warning(f"Failed to load earnings state: {e}")

    def end_of_day(self) -> EarningsRecord:
        """End-of-day routine: generate report, save state, reset daily trades."""
        report = self.daily_report()
        self.save_state()
        self._trades_today.clear()
        return report
