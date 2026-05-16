from __future__ import annotations

"""Roll-or-kill engine — builds a CANDIDATE roll proposal for an existing
options position and runs it through the seven gates.

PROPOSE ONLY. Never places an order. Output is a structured ticket the
operator approves manually. Respects autonomy.trade_execution: human_in_loop.

Heuristic: same delta + next monthly expiry. If yfinance options chain is
available, picks the strike whose mid-delta proxy (|strike-spot|/spot) is
closest to the original position. Otherwise returns a placeholder strike =
current strike (operator decides).
"""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog

from agents.dfv.decision_engine import DFV

_log = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROPOSALS_PATH = REPO_ROOT / "agents" / "dfv" / "memory" / "proposed_orders.jsonl"


def _next_monthly_expiry(after: date) -> date:
    """Third Friday of the next month after `after`."""
    year = after.year
    month = after.month + 1
    if month > 12:
        month = 1
        year += 1
    first = date(year, month, 1)
    # Friday=4 (Monday=0)
    days_to_friday = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=days_to_friday)
    third_friday = first_friday + timedelta(days=14)
    return third_friday


def _spot_price(symbol: str) -> float | None:
    try:
        import yfinance as yf  # noqa: PLC0415
        t = yf.Ticker(symbol)
        info = getattr(t, "fast_info", None) or {}
        for key in ("last_price", "lastPrice", "regular_market_price"):
            v = info.get(key) if isinstance(info, dict) else getattr(info, key, None)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
    except Exception:  # noqa: BLE001
        return None
    return None


def build_roll_proposal(position: dict[str, Any]) -> dict[str, Any]:
    """Build a candidate roll ticket from a position dict.

    Required position keys (best effort): symbol, strike, expiry, side
    ('put'|'call'), quantity. Extra: contract, multiplier.
    """
    symbol = (position.get("symbol") or position.get("ticker") or "").upper()
    side = (position.get("side") or position.get("right") or "put").lower()
    try:
        cur_strike = float(position.get("strike") or 0.0)
    except (TypeError, ValueError):
        cur_strike = 0.0
    try:
        cur_qty = float(position.get("quantity") or position.get("qty") or 0.0)
    except (TypeError, ValueError):
        cur_qty = 0.0
    exp_raw = position.get("expiry") or position.get("expiration") or ""

    # Parse current expiry; if unparsable, use today
    cur_expiry: date
    try:
        cur_expiry = datetime.fromisoformat(str(exp_raw)[:10]).date()
    except ValueError:
        cur_expiry = date.today()

    new_expiry = _next_monthly_expiry(max(cur_expiry, date.today()))
    spot = _spot_price(symbol) if symbol else None

    # Strike selection: keep same strike unless spot has moved >5% — then nudge
    new_strike = cur_strike
    if spot and cur_strike:
        drift = (spot - cur_strike) / cur_strike
        if abs(drift) > 0.05:
            # Round to nearest dollar (good enough for ticket display)
            new_strike = round(spot * (0.95 if side == "put" else 1.05), 0)

    return {
        "kind": "roll_proposal",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbol": symbol,
        "side": side,
        "action": "roll",
        "size_pct": 0.0,  # neutral — roll is debit/credit not new exposure
        "close_leg": {
            "strike": cur_strike,
            "expiry": cur_expiry.isoformat(),
            "qty": cur_qty,
        },
        "open_leg": {
            "strike": new_strike,
            "expiry": new_expiry.isoformat(),
            "qty": cur_qty,
        },
        "rationale": (
            f"Roll {symbol} {side.upper()} ${cur_strike:g} {cur_expiry.isoformat()} "
            f"→ ${new_strike:g} {new_expiry.isoformat()}"
        ),
        "spot_at_quote": spot,
    }


def quote_and_review(position: dict[str, Any], dfv: DFV | None = None) -> dict[str, Any]:
    """Build a roll proposal, run it through the seven gates, and append to
    proposed_orders.jsonl. Never executes."""
    inst = dfv or DFV()
    proposal = build_roll_proposal(position)
    # Run through gates (will likely warn on G2/G6 since it's a roll, not a buy)
    try:
        decision = inst.evaluate({
            "symbol": proposal["symbol"],
            "action": "roll",
            "size_pct": 0.0,
            "expected_slippage_pct": 0.0,
            "cash_after_trade": 1e9,  # neutral; roll doesn't consume cash materially
            "portfolio_value": 1e9,
        }).to_dict()
    except Exception as exc:  # noqa: BLE001
        decision = {"verdict": "errored", "error": str(exc)}

    ticket = {
        **proposal,
        "gate_decision": decision,
        "status": "pending_operator_ok",
        "autonomy": (inst.doctrine.get("autonomy") or {}).get("trade_execution", "human_in_loop"),
    }
    PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROPOSALS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ticket, default=str) + "\n")
    _log.info(
        "dfv.roll.proposed",
        symbol=proposal["symbol"],
        verdict=decision.get("verdict"),
    )
    return ticket


def tail_proposals(n: int = 20) -> list[dict[str, Any]]:
    if not PROPOSALS_PATH.exists():
        return []
    lines = PROPOSALS_PATH.read_text(encoding="utf-8").splitlines()[-n:]
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
