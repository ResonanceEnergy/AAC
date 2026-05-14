"""strategies/roll_manager.py — Sprint 3.2.

21-DTE roll trigger engine.  Evaluates each option position and returns a
RollDecision describing whether to HOLD, ROLL, CLOSE, or flag a DEAD_PUT.

Rules (from ROLL_DISCIPLINE in war_room_engine.py)
---------------------------------------------------
* roll_trigger_dte       = 21   — evaluate roll at 21 DTE, NOT 7
* dead_put_gate          = True — if market_price ~ $0, do NOT roll, re-evaluate
* max_otm_pct_short_dated = 5%  — max 5% OTM for puts ≤ 90 DTE
* max_contracts          = 20   — cap enforced by risk_engine, logged here

Roll logic
----------
DTE > 21            → HOLD
DTE 8–21            → ROLL (get out before gamma kills remaining value)
DTE 0–7             → CLOSE (last-resort close — too late to roll efficiently)
market_price ≈ $0   → DEAD_PUT (don't waste commission rolling a worthless option)
DTE < 0 (expired)   → EXPIRED
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import List, Optional

import structlog

_log = structlog.get_logger(__name__)

# ── Policy constants (mirrors ROLL_DISCIPLINE) ────────────────────────────────

_ROLL_TRIGGER_DTE: int = 21
_CLOSE_TRIGGER_DTE: int = 7
_DEAD_PUT_PRICE_THRESHOLD: float = 0.05    # treat market_price ≤ $0.05 as dead
_MAX_OTM_PCT_SHORT_DATED: float = 0.05     # 5% OTM for puts ≤ 90 DTE


# ── Enums / dataclasses ───────────────────────────────────────────────────────

class RollAction(str, Enum):
    HOLD     = "hold"       # DTE > roll_trigger — no action needed yet
    ROLL     = "roll"       # DTE in [close_trigger+1 … roll_trigger] — roll now
    CLOSE    = "close"      # DTE ≤ close_trigger — too late to roll, close
    DEAD_PUT = "dead_put"   # market_price ~ $0 — don't roll, re-evaluate thesis
    EXPIRED  = "expired"    # DTE < 0 — position should have been cleared
    NOT_OPTION = "not_option"  # non-option position — no roll logic applies


@dataclass
class RollDecision:
    """Roll evaluation result for a single position."""
    symbol: str
    sec_type: str
    dte: Optional[int]             # None for non-options
    action: RollAction
    reason: str
    expiry: Optional[str] = None   # YYYYMMDD
    strike: Optional[float] = None
    right: Optional[str] = None    # 'C' or 'P'
    market_price: float = 0.0
    quantity: float = 0.0
    urgent: bool = False           # True when immediate action is needed

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "sec_type": self.sec_type,
            "dte": self.dte,
            "action": self.action.value,
            "reason": self.reason,
            "expiry": self.expiry,
            "strike": self.strike,
            "right": self.right,
            "market_price": round(self.market_price, 4),
            "quantity": self.quantity,
            "urgent": self.urgent,
        }


# ── Utility ───────────────────────────────────────────────────────────────────

def days_to_expiry(expiry_str: str, as_of: Optional[date] = None) -> int:
    """Return calendar days until expiry.

    Args:
        expiry_str: YYYYMMDD or YYYY-MM-DD format.
        as_of: Reference date (defaults to today UTC).

    Returns:
        Integer days.  Negative means already expired.
    """
    ref = as_of or date.today()
    fmt = "%Y%m%d" if len(expiry_str) == 8 else "%Y-%m-%d"
    exp_date = datetime.strptime(expiry_str, fmt).date()
    return (exp_date - ref).days


# ── Manager ───────────────────────────────────────────────────────────────────

class RollManager:
    """Evaluates option positions for roll / close / hold decisions.

    Usage::

        mgr = RollManager()
        decisions = mgr.evaluate(position_tracker.all())
        urgent = [d for d in decisions if d.urgent]
        for d in urgent:
            print(d.symbol, d.action, d.reason)
    """

    def __init__(
        self,
        roll_trigger_dte: int = _ROLL_TRIGGER_DTE,
        close_trigger_dte: int = _CLOSE_TRIGGER_DTE,
        dead_put_threshold: float = _DEAD_PUT_PRICE_THRESHOLD,
    ) -> None:
        self.roll_trigger_dte = roll_trigger_dte
        self.close_trigger_dte = close_trigger_dte
        self.dead_put_threshold = dead_put_threshold

    def evaluate(self, positions: list) -> List[RollDecision]:
        """Evaluate all positions and return one RollDecision per position.

        Non-option positions (STK, ETF, CASH, CRYPTO) get RollAction.NOT_OPTION.
        """
        decisions: List[RollDecision] = []
        for pos in positions:
            decisions.append(self._evaluate_one(pos))

        urgent = [d for d in decisions if d.urgent]
        if urgent:
            _log.warning(
                "Roll manager: %d position(s) require urgent action",
                len(urgent),
                symbols=[d.symbol for d in urgent],
            )
        return decisions

    def urgent_only(self, positions: list) -> List[RollDecision]:
        """Return only decisions that require immediate action."""
        return [d for d in self.evaluate(positions) if d.urgent]

    def summary(self, positions: list) -> dict:
        """JSON-safe summary of all roll decisions."""
        decisions = self.evaluate(positions)
        by_action: dict = {}
        for d in decisions:
            by_action.setdefault(d.action.value, []).append(d.symbol)
        return {
            "total_evaluated": len(decisions),
            "urgent_count": sum(1 for d in decisions if d.urgent),
            "by_action": by_action,
            "decisions": [d.to_dict() for d in decisions],
        }

    # ── internal ──────────────────────────────────────────────────────────────

    def _evaluate_one(self, pos) -> RollDecision:
        """Evaluate a single PositionSnapshot."""
        sec_type = getattr(pos, "sec_type", "STK")
        symbol = getattr(pos, "symbol", "?")
        market_price = float(getattr(pos, "market_price", 0))
        quantity = float(getattr(pos, "quantity", 0))
        expiry = getattr(pos, "expiry", None)
        strike = getattr(pos, "strike", None)
        right = getattr(pos, "right", None)

        # Non-option
        if sec_type != "OPT":
            return RollDecision(
                symbol=symbol,
                sec_type=sec_type,
                dte=None,
                action=RollAction.NOT_OPTION,
                reason="Not an options position",
                market_price=market_price,
                quantity=quantity,
                urgent=False,
            )

        # Option without expiry (data gap)
        if not expiry:
            return RollDecision(
                symbol=symbol,
                sec_type=sec_type,
                dte=None,
                action=RollAction.HOLD,
                reason="Expiry date unavailable — cannot evaluate DTE",
                expiry=expiry,
                strike=strike,
                right=right,
                market_price=market_price,
                quantity=quantity,
                urgent=False,
            )

        try:
            dte = days_to_expiry(expiry)
        except ValueError:
            return RollDecision(
                symbol=symbol,
                sec_type=sec_type,
                dte=None,
                action=RollAction.HOLD,
                reason=f"Could not parse expiry '{expiry}'",
                expiry=expiry,
                strike=strike,
                right=right,
                market_price=market_price,
                quantity=quantity,
                urgent=False,
            )

        # Dead put / dead call gate (check before DTE)
        if market_price <= self.dead_put_threshold and market_price >= 0:
            return RollDecision(
                symbol=symbol,
                sec_type=sec_type,
                dte=dte,
                action=RollAction.DEAD_PUT,
                reason=(
                    f"Market price ${market_price:.4f} ≤ dead threshold "
                    f"${self.dead_put_threshold:.2f} — do NOT roll, re-evaluate thesis"
                ),
                expiry=expiry,
                strike=strike,
                right=right,
                market_price=market_price,
                quantity=quantity,
                urgent=dte <= self.roll_trigger_dte,
            )

        # Expired
        if dte < 0:
            return RollDecision(
                symbol=symbol,
                sec_type=sec_type,
                dte=dte,
                action=RollAction.EXPIRED,
                reason=f"Expired {abs(dte)} day(s) ago — confirm settlement",
                expiry=expiry,
                strike=strike,
                right=right,
                market_price=market_price,
                quantity=quantity,
                urgent=True,
            )

        # Last chance close (≤ close_trigger_dte)
        if dte <= self.close_trigger_dte:
            return RollDecision(
                symbol=symbol,
                sec_type=sec_type,
                dte=dte,
                action=RollAction.CLOSE,
                reason=(
                    f"{dte} DTE — too close to expiry to roll efficiently. "
                    f"Close position to preserve remaining value."
                ),
                expiry=expiry,
                strike=strike,
                right=right,
                market_price=market_price,
                quantity=quantity,
                urgent=True,
            )

        # Roll window (close_trigger < dte ≤ roll_trigger)
        if dte <= self.roll_trigger_dte:
            return RollDecision(
                symbol=symbol,
                sec_type=sec_type,
                dte=dte,
                action=RollAction.ROLL,
                reason=(
                    f"{dte} DTE — roll trigger at {self.roll_trigger_dte} DTE. "
                    f"Roll to next expiry before theta erosion accelerates."
                ),
                expiry=expiry,
                strike=strike,
                right=right,
                market_price=market_price,
                quantity=quantity,
                urgent=True,
            )

        # Safe — hold
        return RollDecision(
            symbol=symbol,
            sec_type=sec_type,
            dte=dte,
            action=RollAction.HOLD,
            reason=f"{dte} DTE — above roll trigger ({self.roll_trigger_dte} DTE). No action needed.",
            expiry=expiry,
            strike=strike,
            right=right,
            market_price=market_price,
            quantity=quantity,
            urgent=False,
        )
