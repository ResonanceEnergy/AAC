"""Campaign-level risk controls and kill-switch logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class RiskState:
    """Tracks equity, drawdown, and kill-switch state for a campaign."""

    equity: float
    peak_equity: float
    day_start_equity: float
    consecutive_losses: int = 0
    open_positions: int = 0
    cooldown_until: datetime | None = None

    def record_trade(self, pnl: float) -> None:
        self.equity += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def new_day(self) -> None:
        self.day_start_equity = self.equity


def daily_drawdown_ok(state: RiskState, max_dd_pct: float) -> bool:
    return state.equity >= state.day_start_equity * (1.0 - max_dd_pct)


def campaign_drawdown_ok(state: RiskState, max_dd_pct: float) -> bool:
    return state.equity >= state.peak_equity * (1.0 - max_dd_pct)


def kill_switch_active(
    state: RiskState, max_consec: int, cooldown_hours: int, now: datetime | None = None
) -> bool:
    """Return True if the kill-switch should prevent trading."""
    if state.cooldown_until and now:
        if now < state.cooldown_until:
            return True

    if state.consecutive_losses >= max_consec:
        if now:
            state.cooldown_until = now + timedelta(hours=cooldown_hours)
        return True

    return False


def can_open_position(
    state: RiskState,
    max_open: int,
    max_dd_daily: float,
    max_dd_campaign: float,
    max_consec_loss: int,
    cooldown_hours: int,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Check all risk gates.  Returns (allowed, reason)."""
    if kill_switch_active(state, max_consec_loss, cooldown_hours, now):
        return False, "kill_switch"
    if not daily_drawdown_ok(state, max_dd_daily):
        return False, "daily_drawdown"
    if not campaign_drawdown_ok(state, max_dd_campaign):
        return False, "campaign_drawdown"
    if state.open_positions >= max_open:
        return False, "max_positions"
    return True, "ok"


def make_initial_state(starting_equity: float) -> RiskState:
    return RiskState(
        equity=starting_equity,
        peak_equity=starting_equity,
        day_start_equity=starting_equity,
    )
