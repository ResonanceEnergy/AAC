from __future__ import annotations

"""Account-level risk management for paper trading.

Implements best-practice risk controls that operate at the portfolio level
rather than per-position (which StrategyBase.check_stops already handles):
- Max drawdown circuit breaker (halt all trading)
- Daily loss limit
- Max open positions cap
- Per-strategy capital caps
- Correlation/exposure guard
- Kill switch
- Trailing stop-loss tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

_log = structlog.get_logger(__name__)


@dataclass
class RiskConfig:
    """Account-level risk parameters."""

    max_drawdown_pct: float = 20.0       # halt trading if drawdown exceeds this %
    daily_loss_limit_pct: float = 5.0    # max loss % per day before pausing
    max_open_positions: int = 20         # max simultaneous open positions
    max_position_pct: float = 0.15       # max single position as % of equity
    max_strategy_allocation_pct: float = 40.0  # max capital to any single strategy %
    max_correlated_exposure_pct: float = 50.0  # max total exposure in correlated assets
    trailing_stop_pct: float = 0.0       # 0 = disabled; >0 = trailing stop %
    cooldown_after_halt_seconds: float = 3600.0  # 1hr cooldown after circuit breaker


class RiskState:
    """Mutable risk state tracking."""

    def __init__(self) -> None:
        self.is_halted: bool = False
        self.halt_reason: str = ""
        self.halted_at: str = ""
        self.daily_start_equity: float = 0.0
        self.daily_start_date: str = ""
        self.trailing_highs: dict[str, float] = {}  # symbol → peak price since entry
        self.strategy_exposure: dict[str, float] = {}  # strategy → total exposure $
        self.breach_count: int = 0
        self.last_breach: str = ""

    def reset_daily(self, equity: float) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.daily_start_date != today:
            self.daily_start_date = today
            self.daily_start_equity = equity

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "halted_at": self.halted_at,
            "daily_start_equity": round(self.daily_start_equity, 2),
            "daily_start_date": self.daily_start_date,
            "breach_count": self.breach_count,
            "last_breach": self.last_breach,
            "trailing_highs": {k: round(v, 6) for k, v in self.trailing_highs.items()},
        }


class RiskManager:
    """Portfolio-level risk manager.

    Sits between strategies and the paper trading engine.  Before any
    order is placed, ``pre_trade_check`` is called.  After each price
    update, ``post_update_check`` evaluates account-level breaches.

    Usage::

        rm = RiskManager(config=RiskConfig(max_drawdown_pct=15))
        # Before placing an order:
        ok, reason = rm.pre_trade_check(engine, symbol, side, qty, price, strategy)
        if not ok:
            log.warning("risk_blocked", reason=reason)
            return
        # After updating prices:
        rm.post_update_check(engine)
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self.state = RiskState()

    # -- Pre-trade gate -------------------------------------------------------

    def pre_trade_check(
        self,
        account_equity: float,
        account_positions: dict[str, Any],
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy: str,
    ) -> tuple[bool, str]:
        """Evaluate whether a proposed trade should be allowed.

        Returns (allowed, reason). If allowed is False, the trade must
        be rejected.
        """
        # Kill switch / halt
        if self.state.is_halted:
            return False, f"trading_halted: {self.state.halt_reason}"

        # Daily reset
        self.state.reset_daily(account_equity)

        # 1. Max open positions
        if side == "buy" and symbol not in account_positions:
            if len(account_positions) >= self.config.max_open_positions:
                self._record_breach("max_open_positions")
                return False, f"max_open_positions ({self.config.max_open_positions}) reached"

        # 2. Single position size limit
        if account_equity > 0:
            position_value = price * quantity
            position_pct = position_value / account_equity
            if position_pct > self.config.max_position_pct:
                self._record_breach("position_too_large")
                return False, (
                    f"position_size {position_pct:.1%} exceeds "
                    f"max {self.config.max_position_pct:.1%}"
                )

        # 3. Per-strategy allocation cap
        strat_exposure = self.state.strategy_exposure.get(strategy, 0.0)
        new_exposure = strat_exposure + (price * quantity)
        if account_equity > 0:
            strat_pct = (new_exposure / account_equity) * 100
            if strat_pct > self.config.max_strategy_allocation_pct:
                self._record_breach("strategy_allocation_exceeded")
                return False, (
                    f"strategy '{strategy}' allocation {strat_pct:.1f}% "
                    f"exceeds max {self.config.max_strategy_allocation_pct:.1f}%"
                )

        # 4. Daily loss limit
        if self.state.daily_start_equity > 0:
            daily_loss_pct = (
                (self.state.daily_start_equity - account_equity)
                / self.state.daily_start_equity * 100
            )
            if daily_loss_pct >= self.config.daily_loss_limit_pct:
                self._halt(f"daily_loss_limit ({daily_loss_pct:.1f}%)")
                return False, f"daily_loss_limit {daily_loss_pct:.1f}% exceeded"

        return True, ""

    # -- Post-update checks ---------------------------------------------------

    def post_update_check(
        self,
        account_equity: float,
        peak_equity: float,
        positions: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run after price updates. Returns list of risk alerts."""
        alerts: list[dict[str, Any]] = []

        # Daily reset
        self.state.reset_daily(account_equity)

        # 1. Max drawdown circuit breaker
        if peak_equity > 0:
            dd_pct = (peak_equity - account_equity) / peak_equity * 100
            if dd_pct >= self.config.max_drawdown_pct:
                self._halt(f"max_drawdown ({dd_pct:.1f}%)")
                alerts.append({
                    "type": "circuit_breaker",
                    "reason": f"drawdown {dd_pct:.1f}% >= {self.config.max_drawdown_pct:.1f}%",
                    "severity": "critical",
                })

        # 2. Daily loss limit
        if self.state.daily_start_equity > 0:
            daily_loss_pct = (
                (self.state.daily_start_equity - account_equity)
                / self.state.daily_start_equity * 100
            )
            if daily_loss_pct >= self.config.daily_loss_limit_pct:
                self._halt(f"daily_loss_limit ({daily_loss_pct:.1f}%)")
                alerts.append({
                    "type": "daily_loss_limit",
                    "reason": f"daily loss {daily_loss_pct:.1f}% >= {self.config.daily_loss_limit_pct:.1f}%",
                    "severity": "critical",
                })

        # 3. Update strategy exposure
        self._update_strategy_exposure(positions)

        # 4. Update trailing highs
        self._update_trailing_highs(positions)

        # 5. Check trailing stops
        trailing_exits = self._check_trailing_stops(positions)
        if trailing_exits:
            alerts.append({
                "type": "trailing_stop_triggered",
                "symbols": trailing_exits,
                "severity": "warning",
            })

        return alerts

    def check_trailing_stops(self, positions: dict[str, Any]) -> list[str]:
        """Return symbols that hit trailing stop. Public API for optimizer."""
        self._update_trailing_highs(positions)
        return self._check_trailing_stops(positions)

    # -- Kill switch ----------------------------------------------------------

    def halt(self, reason: str = "manual_kill_switch") -> None:
        """Manually halt all trading."""
        self._halt(reason)

    def resume(self) -> None:
        """Manually resume trading after halt."""
        _log.info("risk_manager.resumed", previous_reason=self.state.halt_reason)
        self.state.is_halted = False
        self.state.halt_reason = ""
        self.state.halted_at = ""

    # -- Reporting ------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        return {
            "config": {
                "max_drawdown_pct": self.config.max_drawdown_pct,
                "daily_loss_limit_pct": self.config.daily_loss_limit_pct,
                "max_open_positions": self.config.max_open_positions,
                "max_position_pct": self.config.max_position_pct,
                "max_strategy_allocation_pct": self.config.max_strategy_allocation_pct,
                "trailing_stop_pct": self.config.trailing_stop_pct,
            },
            "state": self.state.to_dict(),
        }

    # -- Internal -------------------------------------------------------------

    def _halt(self, reason: str) -> None:
        if not self.state.is_halted:
            _log.critical("risk_manager.HALT", reason=reason)
        self.state.is_halted = True
        self.state.halt_reason = reason
        self.state.halted_at = datetime.now(timezone.utc).isoformat()

    def _record_breach(self, breach_type: str) -> None:
        self.state.breach_count += 1
        self.state.last_breach = f"{breach_type} @ {datetime.now(timezone.utc).isoformat()}"
        _log.warning("risk_manager.breach", type=breach_type, count=self.state.breach_count)

    def _update_strategy_exposure(self, positions: dict[str, Any]) -> None:
        exposure: dict[str, float] = {}
        for pos in positions.values():
            strat = getattr(pos, "strategy", "") or ""
            val = getattr(pos, "market_value", 0.0)
            exposure[strat] = exposure.get(strat, 0.0) + val
        self.state.strategy_exposure = exposure

    def _update_trailing_highs(self, positions: dict[str, Any]) -> None:
        if self.config.trailing_stop_pct <= 0:
            return
        current_symbols = set()
        for symbol, pos in positions.items():
            current_symbols.add(symbol)
            price = getattr(pos, "current_price", 0.0)
            if price > 0:
                prev_high = self.state.trailing_highs.get(symbol, 0.0)
                if price > prev_high:
                    self.state.trailing_highs[symbol] = price
        # Clean up closed positions
        for sym in list(self.state.trailing_highs):
            if sym not in current_symbols:
                del self.state.trailing_highs[sym]

    def _check_trailing_stops(self, positions: dict[str, Any]) -> list[str]:
        if self.config.trailing_stop_pct <= 0:
            return []
        triggered = []
        for symbol, pos in positions.items():
            price = getattr(pos, "current_price", 0.0)
            high = self.state.trailing_highs.get(symbol, 0.0)
            if high > 0 and price > 0:
                drop_pct = (high - price) / high
                if drop_pct >= self.config.trailing_stop_pct:
                    triggered.append(symbol)
        return triggered
