"""
IBKR Market-Data Client — replaces yfinance for live IV/HV + NYSE breadth.

Why this exists
---------------
Yahoo Finance has been progressively gutted:

* ``^TRIN`` / ``^TICK`` are delisted on Yahoo (HTTP 404).
* ``yfinance.Ticker.info`` for ETF shares-outstanding is rate-limited and
  often blank.
* IV/HV calc derived from Yahoo option chains is the slowest part of the
  dashboard's call-options pillar (17 tickers × full chain fetch).

IBKR's TWS API exposes all of this **for free** with the same US Securities
Snapshot subscription you already have:

* ``Index('TICK-NYSE', 'NYSE')`` / ``Index('TRIN-NYSE', 'NYSE')`` for breadth.
* ``reqMktData(stock, genericTickList='104,106')`` returns:
    - ``ticker.histVolatility``     -> 30-day annualised historical vol (tick 104)
    - ``ticker.impliedVolatility``  -> 30-day annualised model implied vol (tick 106)

This client is intentionally **synchronous** (uses ``ib_insync.util.run``)
and **short-lived** — it connects, fetches a batch, disconnects. No long-
running event loop, so it's safe to call from Streamlit's cached collectors.

Falls back gracefully: every public method returns ``None`` (or empty dict)
if ib_insync is missing or TWS is unreachable, so callers can degrade to
yfinance without try/except boilerplate.

Configuration
-------------
Reads ``.env``:

    IBKR_HOST=127.0.0.1
    IBKR_PORT=7496        # falls back to 7497 (paper) automatically
    IBKR_CLIENT_ID=27     # uses 27 to avoid clashing with the trading connector

Env override:

    AAC_DATA_PRIMARY=ibkr|yfinance   # default: ibkr
"""

from __future__ import annotations

import os
import time
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

try:
    from ib_insync import IB, Index, Stock  # type: ignore[import-untyped]
    _IB_AVAILABLE = True
except ImportError:
    IB = None  # type: ignore[assignment]
    Index = None  # type: ignore[assignment]
    Stock = None  # type: ignore[assignment]
    _IB_AVAILABLE = False


# Module-level cache to keep one IB connection alive within a single process
# across multiple collector calls (Streamlit reruns share the same process).
_IB_INSTANCE: Any = None
_LAST_CONNECT_ATTEMPT: float = 0.0
_RECONNECT_COOLDOWN_S: float = 60.0  # don't hammer TWS if it's down
_LAST_CONNECT_FAILED: bool = False


def _is_enabled() -> bool:
    return _IB_AVAILABLE and os.environ.get("AAC_DATA_PRIMARY", "ibkr").lower() == "ibkr"


def _get_ib() -> Any:
    """Return a connected IB instance, or None if unavailable.

    Caches the connection at module level. Honours a 60s cooldown after a
    failed connect so that a downed TWS doesn't add 5s × N retries to every
    refresh cycle.
    """
    global _IB_INSTANCE, _LAST_CONNECT_ATTEMPT, _LAST_CONNECT_FAILED

    if not _is_enabled():
        return None

    if _IB_INSTANCE is not None and _IB_INSTANCE.isConnected():
        return _IB_INSTANCE

    now = time.time()
    if _LAST_CONNECT_FAILED and (now - _LAST_CONNECT_ATTEMPT) < _RECONNECT_COOLDOWN_S:
        return None

    _LAST_CONNECT_ATTEMPT = now
    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    primary_port = int(os.environ.get("IBKR_PORT", "7496"))
    client_id = int(os.environ.get("IBKR_CLIENT_ID_DATA", "27"))
    candidates = [primary_port]
    # Try paper as a soft fallback if live refused.
    if primary_port == 7496:
        candidates.append(7497)
    elif primary_port == 4001:
        candidates.append(4002)

    for port in candidates:
        try:
            # ib_insync's sync connect() internally schedules connectAsync on the
            # thread's event loop. Worker threads (e.g. xAI council, scheduler)
            # don't have one — without this guard the coroutine is created but
            # never awaited, raising "There is no current event loop in thread X"
            # and spamming RuntimeWarning.
            import asyncio as _asyncio  # noqa: PLC0415
            try:
                _asyncio.get_event_loop()
            except RuntimeError:
                _asyncio.set_event_loop(_asyncio.new_event_loop())
            ib = IB()
            ib.connect(host, port, clientId=client_id, timeout=4)
            _IB_INSTANCE = ib
            _LAST_CONNECT_FAILED = False
            _log.info("ibkr_data_connected", host=host, port=port, client_id=client_id)
            return _IB_INSTANCE
        except (ConnectionRefusedError, OSError, TimeoutError) as exc:
            _log.debug("ibkr_data_connect_failed", host=host, port=port, error=str(exc))
            continue
        except Exception as exc:  # ib_insync wraps lots of error types
            # Narrowed by name to keep AAC convention; fall through to next port.
            _log.debug("ibkr_data_connect_unexpected", port=port, error_type=type(exc).__name__, error=str(exc))
            continue

    _LAST_CONNECT_FAILED = True
    return None


def disconnect() -> None:
    global _IB_INSTANCE, _LAST_CONNECT_FAILED
    if _IB_INSTANCE is not None:
        try:
            _IB_INSTANCE.disconnect()
        except (RuntimeError, OSError):
            pass
        _IB_INSTANCE = None
        _LAST_CONNECT_FAILED = False


def is_available() -> bool:
    """Cheap health check — True if TWS is reachable for market data."""
    return _get_ib() is not None


# ── Breadth ────────────────────────────────────────────────────────────────


_BREADTH_CONTRACTS = {
    # IBKR uses dash-separated index symbols on the NYSE feed.
    "tick": ("TICK-NYSE", "NYSE"),
    "trin": ("TRIN-NYSE", "NYSE"),
    "advances": ("AD-NYSE", "NYSE"),
    "declines": ("DC-NYSE", "NYSE"),
}


def get_breadth_snapshot(wait_s: float = 1.5) -> dict[str, Any] | None:
    """Return TICK / TRIN / advances / declines from IBKR, or None if unavailable."""
    ib = _get_ib()
    if ib is None or Index is None:
        return None

    out: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "ibkr"}
    tickers = {}
    try:
        for name, (sym, exch) in _BREADTH_CONTRACTS.items():
            try:
                contract = Index(sym, exch)
                ib.qualifyContracts(contract)
                tickers[name] = ib.reqMktData(contract, snapshot=False, regulatorySnapshot=False)
            except (RuntimeError, OSError, ValueError) as exc:
                _log.debug("breadth_contract_failed", name=name, error=str(exc))
        ib.sleep(wait_s)
        for name, t in tickers.items():
            val = None
            for attr in ("last", "close", "marketPrice"):
                v = getattr(t, attr, None)
                if callable(v):
                    try:
                        v = v()
                    except (RuntimeError, ValueError):
                        v = None
                if v is not None and v == v and v > 0:  # NaN check
                    val = float(v)
                    break
            out[name] = val
    finally:
        for name, (sym, exch) in _BREADTH_CONTRACTS.items():
            try:
                ib.cancelMktData(Index(sym, exch))
            except (RuntimeError, OSError, ValueError):
                pass

    if out.get("tick") is None and out.get("trin") is None:
        return None

    adv = out.get("advances") or 0
    dec = out.get("declines") or 0
    out["adv_minus_decl"] = adv - dec if (adv and dec) else None
    out["advance_decline_ratio"] = (adv / dec) if (adv and dec) else None
    out["regime"] = _classify_breadth(out.get("trin"), out.get("tick"))
    out["notes"] = "IBKR NYSE indices (TICK-NYSE, TRIN-NYSE)"
    return out


def _classify_breadth(trin: float | None, tick: float | None) -> str:
    if trin is None and tick is None:
        return "unknown"
    if trin is not None and trin > 1.5:
        return "oversold"
    if trin is not None and trin < 0.7:
        return "overbought"
    if tick is not None and tick > 800:
        return "strong_buy_tape"
    if tick is not None and tick < -800:
        return "strong_sell_tape"
    return "neutral"


# ── Per-stock IV / HV ─────────────────────────────────────────────────────


def get_iv_hv_snapshot(tickers: list[str], wait_s: float = 2.0) -> dict[str, dict[str, Any]] | None:
    """Batch-fetch IV (tick 106) + HV (tick 104) for a list of US equity tickers.

    Returns ``{ticker: {"spot": float, "iv": float, "hv": float, "iv_hv_ratio": float}}``
    keyed by ticker symbol. Returns None if IBKR is unavailable.

    Uses the IBKR generic tick list ``104,106`` which is **free** with the
    standard US Securities Snapshot subscription — no extra cost.
    """
    ib = _get_ib()
    if ib is None or Stock is None:
        return None

    out: dict[str, dict[str, Any]] = {}
    contracts = []
    tickers_by_sym: dict[str, Any] = {}
    try:
        for sym in tickers:
            try:
                c = Stock(sym, "SMART", "USD")
                ib.qualifyContracts(c)
                contracts.append((sym, c))
                t = ib.reqMktData(c, genericTickList="104,106", snapshot=False)
                tickers_by_sym[sym] = t
            except (RuntimeError, OSError, ValueError) as exc:
                _log.debug("iv_hv_contract_failed", ticker=sym, error=str(exc))
                out[sym] = {"error": str(exc)}

        ib.sleep(wait_s)

        for sym, t in tickers_by_sym.items():
            spot = None
            for attr in ("last", "marketPrice", "close"):
                v = getattr(t, attr, None)
                if callable(v):
                    try:
                        v = v()
                    except (RuntimeError, ValueError):
                        v = None
                if v is not None and v == v and v > 0:
                    spot = float(v)
                    break

            iv = getattr(t, "impliedVolatility", None)
            hv = getattr(t, "histVolatility", None)
            iv = float(iv) if (iv is not None and iv == iv and iv > 0) else None
            hv = float(hv) if (hv is not None and hv == hv and hv > 0) else None
            ratio = (iv / hv) if (iv is not None and hv is not None and hv > 0) else None

            out[sym] = {
                "ticker": sym,
                "spot": spot,
                "implied_vol": iv,
                "realized_hv": hv,
                "iv_hv_ratio": round(ratio, 3) if ratio is not None else None,
                "option_available": iv is not None,
                "source": "ibkr",
            }
    finally:
        for _sym, c in contracts:
            try:
                ib.cancelMktData(c)
            except (RuntimeError, OSError, ValueError):
                pass

    return out
