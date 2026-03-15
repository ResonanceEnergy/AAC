"""Manual execution adapter — generates trade plan files for human MT5 execution.

This adapter does NOT connect to any API.  It writes trade plans as structured
files that you execute manually in MT5/PU Prime, then import the account
history CSV back for journaling.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .broker_base import BrokerAdapter


class ManualMT5Adapter(BrokerAdapter):
    """Generates trade plan JSON files for manual execution in MT5."""

    def __init__(self, plans_dir: str | Path = "modules/aac_puprime_vce/data/trade_plans"):
        self.plans_dir = Path(plans_dir)
        self.plans_dir.mkdir(parents=True, exist_ok=True)

    def get_account_info(self) -> dict[str, Any]:
        return {"mode": "manual", "note": "Check MT5 terminal for live account info"}

    def place_order(
        self, symbol: str, side: str, size: float,
        stop: float, take_profit: float,
    ) -> dict[str, Any]:
        plan = {
            "action": "OPEN",
            "symbol": symbol,
            "side": side.upper(),
            "size": round(size, 4),
            "stop_loss": round(stop, 6),
            "take_profit": round(take_profit, 6),
            "generated_at": datetime.now().isoformat(),
            "status": "PENDING_MANUAL_EXECUTION",
            "instructions": (
                f"Open MT5 → {symbol} → New Order → "
                f"{'Buy' if side == 'long' else 'Sell'} {round(size, 4)} lots → "
                f"SL: {round(stop, 6)} → TP: {round(take_profit, 6)}"
            ),
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plan_path = self.plans_dir / f"plan_{symbol}_{ts}.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        return plan

    def close_position(self, symbol: str) -> dict[str, Any]:
        plan = {
            "action": "CLOSE",
            "symbol": symbol,
            "generated_at": datetime.now().isoformat(),
            "status": "PENDING_MANUAL_EXECUTION",
            "instructions": f"Open MT5 → {symbol} → Close position",
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plan_path = self.plans_dir / f"close_{symbol}_{ts}.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        return plan

    def get_open_positions(self) -> list[dict[str, Any]]:
        return [{"mode": "manual", "note": "Check MT5 terminal for open positions"}]
