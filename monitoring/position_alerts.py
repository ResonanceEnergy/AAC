"""
Position Alerts Module
======================
Pure functions that check account positions for actionable conditions:
  - Approaching expiry  (DTE thresholds: 14, 7, 3, 1, 0)
  - Stale balance data   (>24 h since last refresh)
  - Low cash             (<$100 USD in any account with positions)
  - Severe unrealised P&L(<-50%)

Returns lists of alert dicts compatible with
``ContinuousMonitoringService._create_alert(alert_id, severity, message)``.

No daemon, no CLI, no Telegram — notification delivery is the caller's job.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# ── Paths & Thresholds ──────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BALANCES_FILE = PROJECT_ROOT / "data" / "account_balances.json"

DTE_THRESHOLDS = [14, 7, 3, 1, 0]
DATA_STALE_HOURS = 24
CASH_WARN_USD = 100.0
PNL_WARN_PCT = -50.0


# ── Helpers ──────────────────────────────────────────────────────────────

def load_balances() -> dict[str, Any]:
    """Load data/account_balances.json.  Returns ``{}`` on error."""
    if not BALANCES_FILE.exists():
        return {}
    try:
        with open(BALANCES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("Failed to read balances file: %s", exc)
        return {}


def parse_expiry(pos: dict[str, Any]) -> date | None:
    """Parse expiry from a position dict (YYYYMMDD or YYYY-MM-DD)."""
    raw = pos.get("expiry") or pos.get("expiration") or ""
    if not raw:
        return None
    raw = str(raw).replace("-", "")
    if len(raw) == 8 and raw.isdigit():
        return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
    return None


# ── Alert: position DTE ─────────────────────────────────────────────────

def check_position_dte(
    data: dict[str, Any],
    *,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Return alerts for positions nearing or past expiry.

    Each alert dict has keys:
        alert_id  — unique key for dedup  (``dte:<acct>:<label>:<threshold>``)
        severity  — ``"critical"`` | ``"warning"``
        message   — human-readable message
        meta      — extra data (dte, expiry, account, symbol)
    """
    alerts: list[dict[str, Any]] = []
    if today is None:
        today = date.today()
    accounts = data.get("accounts", {})

    for acct_name, acct in accounts.items():
        for pos in acct.get("positions", []):
            expiry = parse_expiry(pos)
            if expiry is None:
                continue

            dte = (expiry - today).days
            symbol = pos.get("symbol", "???")
            strike = pos.get("strike", "")
            right = pos.get("right", "")
            qty = pos.get("qty", 0)
            label = f"{symbol} ${strike}{right}" if strike else symbol

            for threshold in DTE_THRESHOLDS:
                if dte <= threshold:
                    if dte < 0:
                        tag = "EXPIRED"
                        severity = "critical"
                    elif dte == 0:
                        tag = "EXPIRING TODAY"
                        severity = "critical"
                    elif dte <= 3:
                        tag = "CRITICAL"
                        severity = "critical"
                    elif dte <= 7:
                        tag = "WARNING"
                        severity = "warning"
                    else:
                        tag = "NOTICE"
                        severity = "warning"

                    alerts.append({
                        "alert_id": f"dte:{acct_name}:{label}:{threshold}d",
                        "severity": severity,
                        "message": (
                            f"{tag}: {label} in {acct_name} — "
                            f"{dte}d to expiry ({expiry}), qty={qty}. "
                            f"{'Close or roll' if dte <= 7 else 'Monitor'}"
                        ),
                        "meta": {
                            "type": "position_dte",
                            "account": acct_name,
                            "symbol": label,
                            "dte": dte,
                            "expiry": str(expiry),
                        },
                    })
                    break  # only the most urgent threshold

    return alerts


# ── Alert: data freshness ───────────────────────────────────────────────

def check_data_freshness(
    data: dict[str, Any],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return an alert if account_balances.json is stale."""
    alerts: list[dict[str, Any]] = []
    if now is None:
        now = datetime.now()
    meta = data.get("_meta", {})
    updated_str = meta.get("updated", "")
    if not updated_str:
        return alerts

    try:
        updated = datetime.fromisoformat(updated_str)
    except ValueError:
        return alerts

    age_hours = (now - updated).total_seconds() / 3600

    if age_hours > DATA_STALE_HOURS:
        alerts.append({
            "alert_id": f"stale_data:{int(age_hours // 24)}d",
            "severity": "warning",
            "message": (
                f"Account balances are {age_hours:.0f}h old "
                f"(last update: {updated_str}). "
                f"Run: python -m config.account_balances auto-update"
            ),
            "meta": {
                "type": "data_stale",
                "age_hours": round(age_hours, 1),
                "last_update": updated_str,
            },
        })

    return alerts


# ── Alert: low cash ─────────────────────────────────────────────────────

def check_low_cash(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return alerts for accounts with critically low cash and open positions."""
    alerts: list[dict[str, Any]] = []
    accounts = data.get("accounts", {})

    for acct_name, acct in accounts.items():
        balance = float(acct.get("balance", 0))
        currency = acct.get("currency", "USD")
        threshold = CASH_WARN_USD
        if currency == "CAD":
            fx = float(data.get("fx", {}).get("cad_usd", 0.70))
            threshold = CASH_WARN_USD / fx if fx > 0 else CASH_WARN_USD

        if balance < threshold and acct.get("positions"):
            alerts.append({
                "alert_id": f"low_cash:{acct_name}",
                "severity": "warning",
                "message": (
                    f"Low cash: {acct_name} has {currency} ${balance:,.2f} "
                    f"(below ${threshold:,.0f}). Cannot execute rolls or new trades"
                ),
                "meta": {
                    "type": "low_cash",
                    "account": acct_name,
                    "balance": balance,
                    "currency": currency,
                },
            })

    return alerts


# ── Alert: P&L ──────────────────────────────────────────────────────────

def check_pnl(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return alerts for positions with severe unrealized losses."""
    alerts: list[dict[str, Any]] = []
    accounts = data.get("accounts", {})

    for acct_name, acct in accounts.items():
        for pos in acct.get("positions", []):
            pnl = pos.get("unrealizedPNL")
            cost = pos.get("avgCost", 0)
            qty = pos.get("qty", 1)
            if pnl is None or cost == 0:
                continue
            total_cost = abs(float(cost)) * abs(float(qty))
            if total_cost == 0:
                continue
            pnl_pct = (float(pnl) / total_cost) * 100

            if pnl_pct < PNL_WARN_PCT:
                symbol = pos.get("symbol", "???")
                alerts.append({
                    "alert_id": f"pnl:{acct_name}:{symbol}",
                    "severity": "warning",
                    "message": (
                        f"P&L alert: {symbol} in {acct_name} at {pnl_pct:.0f}% "
                        f"(unrealized ${float(pnl):,.2f}). Consider closing or adjusting"
                    ),
                    "meta": {
                        "type": "pnl_warning",
                        "account": acct_name,
                        "symbol": symbol,
                        "pnl_pct": round(pnl_pct, 1),
                        "unrealized_pnl": float(pnl),
                    },
                })

    return alerts


# ── Aggregate check ─────────────────────────────────────────────────────

def run_all_checks(
    data: dict[str, Any] | None = None,
    *,
    today: date | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Run every check and return combined alert list.

    If *data* is ``None``, loads from the default balances file.
    """
    if data is None:
        data = load_balances()
    if not data:
        return []

    alerts: list[dict[str, Any]] = []
    alerts.extend(check_position_dte(data, today=today))
    alerts.extend(check_data_freshness(data, now=now))
    alerts.extend(check_low_cash(data))
    alerts.extend(check_pnl(data))
    return alerts
