"""Live portfolio refresher — pulls broker data and writes account_balances.json.

Sources (each independent — one broker down does NOT fail the others):
    - IBKR (via TradingExecution.exchange_connectors.ibkr_connector.IBKRConnector)
        Account summary (NetLiq, BuyPow, Cash, uPnL, rPnL, margin) for BASE/USD/CAD
        Positions (STK, OPT, FUT, CRYPTO) with qty/avgCost/marketValue/unrealizedPNL
    - Moomoo (via moomoo SDK direct — needs OpenD on 127.0.0.1:11111)
        accinfo_query for USD AND CAD
        position_list_query for stocks + options
    - WealthSimple via SnapTrade (read-only)
        Requires SNAPTRADE_CLIENT_ID/CONSUMER_KEY/USER_ID/USER_SECRET in .env
        See scripts/_setup_snaptrade.py for one-time registration + OAuth link.

Output schema preserved: data/account_balances.json keeps _meta + fx + injection
blocks intact; only accounts.ibkr / accounts.moomoo / accounts.wealthsimple
are replaced with live snapshots (other accounts left untouched).

Public API:
    refresh_portfolio_live(write: bool = True) -> dict   # synchronous wrapper

Each source returns a status dict. On any error the source is marked status="error"
with an error message and the previous JSON value for that account is preserved.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import structlog

_log = structlog.get_logger()

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_BAL_PATH = _ROOT / "data" / "account_balances.json"

# Ensure .env is loaded so SNAPTRADE_*, NDAX_*, IBKR_*, MOOMOO_* are present
# when this module is invoked outside the dashboard/launcher (e.g. cron, CLI).
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    pass

# Per-process refresh lock — avoid concurrent broker hammering when multiple
# Streamlit/FastAPI workers race on the same cache miss.
_REFRESH_LOCK = threading.Lock()


# ── IBKR ────────────────────────────────────────────────────────────────────


async def _refresh_ibkr_async() -> dict[str, Any]:
    """Pull live IBKR account summary + positions. Returns account dict."""
    from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector

    connector = IBKRConnector()
    try:
        ok = await connector.connect()
    except (ConnectionError, OSError, RuntimeError) as exc:
        return {"status": "error", "error": f"connect: {exc}"}

    if not ok:
        return {"status": "error", "error": "connect returned False"}

    try:
        summary = await connector.get_account_summary()
        positions_raw = await connector.get_positions()
    except (RuntimeError, OSError, ValueError) as exc:
        return {"status": "error", "error": f"fetch: {exc}"}
    finally:
        try:
            await connector.disconnect()
        except (RuntimeError, OSError):
            pass

    # Translate to account_balances.json schema (IBKR-style field names)
    positions: list[dict[str, Any]] = []
    in_positions = 0.0
    for p in positions_raw:
        qty = float(p.get("quantity", 0) or 0)
        if qty == 0:
            continue
        mv = float(p.get("market_value", 0) or 0)
        in_positions += abs(mv)
        entry = {
            "symbol": p.get("symbol", "?"),
            "secType": p.get("sec_type", "STK"),
            "exchange": p.get("exchange", ""),
            "currency": p.get("currency", "USD"),
            "qty": qty,
            "avgCost": round(float(p.get("avg_cost", 0) or 0), 6),
            "marketPrice": round(float(p.get("market_price", 0) or 0), 6),
            "marketValue": round(mv, 2),
            "unrealizedPNL": round(float(p.get("unrealized_pnl", 0) or 0), 2),
            "realizedPNL": round(float(p.get("realized_pnl", 0) or 0), 2),
        }
        if p.get("sec_type") == "OPT":
            entry["expiry"] = p.get("expiry")
            entry["strike"] = p.get("strike")
            entry["right"] = p.get("right")
            entry["multiplier"] = p.get("multiplier", "100")
        if p.get("sec_type") == "FUT":
            entry["expiry"] = p.get("expiry")
            entry["multiplier"] = p.get("multiplier", "")
        positions.append(entry)

    # Pick best available NetLiq / Cash from BASE first, then USD, then CAD
    def _pick(tag: str) -> float | None:
        for ccy in ("BASE", "USD", "CAD"):
            v = summary.get(f"{tag}_{ccy}")
            if v is not None:
                return float(v)
        return None

    net_liq = _pick("NetLiquidation") or 0.0
    total_cash = _pick("TotalCashValue") or 0.0
    buying_power = _pick("BuyingPower") or 0.0
    avail_funds = _pick("AvailableFunds") or 0.0
    upnl = _pick("UnrealizedPnL") or 0.0
    rpnl = _pick("RealizedPnL") or 0.0
    maint_margin = _pick("MaintMarginReq") or 0.0
    init_margin = _pick("InitMarginReq") or 0.0

    # Determine reporting currency — prefer the one that actually had values
    base_ccy = "USD"
    for ccy in ("BASE", "USD", "CAD"):
        if summary.get(f"NetLiquidation_{ccy}") is not None:
            base_ccy = ccy if ccy != "BASE" else base_ccy
            break

    return {
        "status": "ok",
        "balance": round(avail_funds, 2),
        "currency": base_ccy,
        "total_assets": round(net_liq, 2),
        "in_positions": round(in_positions, 2),
        "buying_power": round(buying_power, 2),
        "total_cash": round(total_cash, 2),
        "unrealized_pnl": round(upnl, 2),
        "realized_pnl": round(rpnl, 2),
        "maint_margin": round(maint_margin, 2),
        "init_margin": round(init_margin, 2),
        "platform": "TWS",
        "account_id": connector.account or os.getenv("IBKR_ACCOUNT", ""),
        "note": f"Live IBKR refresh — {len(positions)} positions, NetLiq {base_ccy} {net_liq:,.2f}",
        "verified": dt.date.today().isoformat(),
        "positions": positions,
    }


def _refresh_ibkr() -> dict[str, Any]:
    """Sync wrapper — runs the IBKR async fetch on a fresh event loop."""
    try:
        return asyncio.run(_refresh_ibkr_async())
    except RuntimeError as exc:
        # Already in an event loop (e.g. called from async context)
        if "asyncio.run() cannot be called" in str(exc):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_refresh_ibkr_async())
            finally:
                loop.close()
        return {"status": "error", "error": f"event_loop: {exc}"}


# ── Moomoo ──────────────────────────────────────────────────────────────────


def _refresh_moomoo() -> dict[str, Any]:
    """Pull live Moomoo account info (USD+CAD) + positions via OpenD."""
    try:
        from moomoo import (
            RET_OK,
            Currency,
            OpenSecTradeContext,
            SecurityFirm,
            TrdEnv,
            TrdMarket,
        )
    except ImportError as exc:
        return {"status": "error", "error": f"moomoo SDK missing: {exc}"}

    host = os.getenv("MOOMOO_HOST", "127.0.0.1")
    try:
        port = int(os.getenv("MOOMOO_PORT", "11111"))
    except ValueError:
        port = 11111

    env_str = os.getenv("MOOMOO_TRADE_ENV", os.getenv("MOOMOO_ENV", "REAL")).upper()
    trd_env = TrdEnv.REAL if env_str == "REAL" else TrdEnv.SIMULATE
    firm_str = os.getenv("MOOMOO_SECURITY_FIRM", "FUTUCA").upper()
    security_firm = getattr(SecurityFirm, firm_str, SecurityFirm.FUTUCA)

    try:
        ctx = OpenSecTradeContext(
            host=host,
            port=port,
            filter_trdmarket=TrdMarket.US,
            security_firm=security_firm,
        )
    except (ConnectionError, OSError, RuntimeError) as exc:
        return {"status": "error", "error": f"OpenD connect: {exc}"}

    # Optional unlock for REAL mode
    trade_pwd = os.getenv("MOOMOO_TRADE_PASSWORD", "")
    if trd_env == TrdEnv.REAL and trade_pwd:
        try:
            ctx.unlock_trade(trade_pwd)
        except (RuntimeError, OSError):
            pass  # if unlock fails read-only queries usually still work

    accinfo: dict[str, Any] = {}
    currency_used = "USD"

    def _flt(v: Any, default: float = 0.0) -> float:
        """Moomoo returns 'N/A' strings for empty fields — coerce safely."""
        if v is None:
            return default
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return f
    try:
        for currency, ccy_label in ((Currency.USD, "USD"), (Currency.CAD, "CAD")):
            ret, data = ctx.accinfo_query(trd_env=trd_env, currency=currency)
            if ret == RET_OK and not data.empty:
                row = data.iloc[0]
                accinfo[ccy_label] = {
                    "cash": _flt(row.get("cash")),
                    "market_val": _flt(row.get("market_val")),
                    "total_assets": _flt(row.get("total_assets")),
                    "frozen_cash": _flt(row.get("frozen_cash")),
                    "available_funds": _flt(row.get("available_funds")),
                    "buying_power": _flt(row.get("power")),
                    "unrealized_pl": _flt(row.get("unrealized_pl")),
                    "realized_pl": _flt(row.get("realized_pl")),
                }
                if accinfo[ccy_label]["total_assets"] > 0 and currency_used == "USD":
                    currency_used = ccy_label
    except (RuntimeError, OSError, ValueError) as exc:
        try:
            ctx.close()
        except (RuntimeError, OSError):
            pass
        return {"status": "error", "error": f"accinfo_query: {exc}"}

    # Positions
    positions: list[dict[str, Any]] = []
    in_positions = 0.0
    try:
        ret2, pos_data = ctx.position_list_query(trd_env=trd_env)
        if ret2 == RET_OK and not pos_data.empty:
            for _, row in pos_data.iterrows():
                qty = _flt(row.get("qty"))
                if qty == 0:
                    continue
                mv = _flt(row.get("market_val"))
                in_positions += abs(mv)
                positions.append({
                    "symbol": str(row.get("code", "")),
                    "qty": qty,
                    "cost_price": _flt(row.get("cost_price")),
                    "market_val": round(mv, 2),
                    "nominal_price": _flt(row.get("nominal_price")),
                    "pl_val": round(_flt(row.get("pl_val")), 2),
                    "pl_ratio": round(_flt(row.get("pl_ratio")), 4),
                    "position_side": str(row.get("position_side", "LONG")),
                })
    except (RuntimeError, OSError, ValueError) as exc:
        try:
            ctx.close()
        except (RuntimeError, OSError):
            pass
        return {"status": "error", "error": f"position_list_query: {exc}"}

    try:
        ctx.close()
    except (RuntimeError, OSError):
        pass

    if not accinfo:
        return {"status": "error", "error": "no accinfo returned (check OpenD login + trade unlock)"}

    primary = accinfo.get(currency_used) or next(iter(accinfo.values()))

    return {
        "status": "ok",
        "balance": round(primary.get("cash", 0), 2),
        "currency": currency_used,
        "total_assets": round(primary.get("total_assets", 0), 2),
        "in_positions": round(in_positions, 2),
        "buying_power": round(primary.get("buying_power", 0), 2),
        "available_funds": round(primary.get("available_funds", 0), 2),
        "frozen_cash": round(primary.get("frozen_cash", 0), 2),
        "unrealized_pnl": round(primary.get("unrealized_pl", 0), 2),
        "realized_pnl": round(primary.get("realized_pl", 0), 2),
        "platform": "Moomoo",
        "account_id": firm_str,
        "by_currency": {ccy: {k: round(v, 2) for k, v in vals.items()} for ccy, vals in accinfo.items()},
        "note": f"Live Moomoo refresh — {firm_str}/{env_str}, {len(positions)} positions, total {currency_used} {primary.get('total_assets', 0):,.2f}",
        "verified": dt.date.today().isoformat(),
        "positions": positions,
    }


# ── NDAX (ccxt) ─────────────────────────────────────────────────────────────


def _refresh_ndax() -> dict[str, Any]:
    """Pull NDAX balances + crypto positions via ccxt (read-only fetch_balance).

    Returns status="not_configured" if NDAX_API_KEY/SECRET/USER_ID missing in .env.
    ccxt 4.x quirk: ndax needs uid+login+password — we use NDAX_USER_ID for login
    and NDAX_API_SECRET as password.
    """
    api_key = os.getenv("NDAX_API_KEY", "")
    api_secret = os.getenv("NDAX_API_SECRET", "")
    uid = os.getenv("NDAX_USER_ID", "")

    if not api_key or not api_secret or not uid:
        return {
            "status": "not_configured",
            "error": "NDAX_API_KEY / NDAX_API_SECRET / NDAX_USER_ID missing in .env",
            "setup_hint": "Set NDAX_API_KEY, NDAX_API_SECRET, NDAX_USER_ID in .env (regenerate keys at ndax.com → API)",
        }

    try:
        import ccxt
    except ImportError as exc:
        return {"status": "error", "error": f"ccxt not installed: {exc}"}

    try:
        exchange = ccxt.ndax({
            "apiKey": api_key,
            "secret": api_secret,
            "uid": uid,
            "login": uid,
            "password": api_secret,
            "enableRateLimit": True,
        })
        bal = exchange.fetch_balance()
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        return {"status": "error", "error": f"fetch_balance: {exc}"}
    except Exception as exc:  # noqa: BLE001 — ccxt raises bespoke errors
        return {"status": "error", "error": f"ndax: {type(exc).__name__}: {exc}"}

    total = bal.get("total", {}) or {}
    free = bal.get("free", {}) or {}

    cad_cash = float(free.get("CAD", total.get("CAD", 0)) or 0)
    positions: list[dict[str, Any]] = []
    mtm_cad = cad_cash

    for asset, amount in total.items():
        if asset == "CAD":
            continue
        qty = float(amount or 0)
        if qty <= 0:
            continue
        price_cad = 0.0
        try:
            ticker = exchange.fetch_ticker(f"{asset}/CAD")
            price_cad = float(ticker.get("last") or ticker.get("close") or 0.0)
        except Exception:  # noqa: BLE001 — pair may not exist
            price_cad = 0.0
        value_cad = qty * price_cad if price_cad > 0 else 0.0
        mtm_cad += value_cad
        positions.append({
            "symbol": asset,
            "qty": round(qty, 8),
            "price_cad": round(price_cad, 4),
            "market_val_cad": round(value_cad, 2),
        })

    return {
        "status": "ok",
        "balance": round(cad_cash, 2),
        "currency": "CAD",
        "total_assets": round(mtm_cad, 2),
        "in_positions": round(mtm_cad - cad_cash, 2),
        "platform": "NDAX",
        "account_id": uid,
        "note": (
            f"Live NDAX refresh — {len(positions)} crypto position(s), "
            f"CAD cash {cad_cash:,.2f}, total CAD {mtm_cad:,.2f}"
        ),
        "verified": dt.date.today().isoformat(),
        "positions": positions,
    }


# ── WealthSimple via SnapTrade ──────────────────────────────────────────────


def _refresh_wealthsimple() -> dict[str, Any]:
    """Pull WealthSimple balances + positions through SnapTrade (read-only).

    Returns status="not_configured" with setup hint if SNAPTRADE_* env not set.
    """
    client_id = os.getenv("SNAPTRADE_CLIENT_ID", "")
    consumer_key = os.getenv("SNAPTRADE_CONSUMER_KEY", "")
    user_id = os.getenv("SNAPTRADE_USER_ID", "")
    user_secret = os.getenv("SNAPTRADE_USER_SECRET", "")

    if not client_id or not consumer_key:
        return {
            "status": "not_configured",
            "error": "SNAPTRADE_CLIENT_ID / SNAPTRADE_CONSUMER_KEY missing in .env",
            "setup_hint": "Sign up at https://dashboard.snaptrade.com then run: python scripts/_setup_snaptrade.py --register",
        }
    if not user_id or not user_secret:
        return {
            "status": "not_configured",
            "error": "SNAPTRADE_USER_ID / SNAPTRADE_USER_SECRET missing in .env",
            "setup_hint": "Run: python scripts/_setup_snaptrade.py --register, then --connect to link Wealthsimple",
        }

    try:
        from snaptrade_client import Configuration
        from snaptrade_client.client import SnapTrade
    except ImportError as exc:
        return {"status": "error", "error": f"snaptrade-python-sdk not installed: {exc}"}

    try:
        cfg = Configuration(consumer_key=consumer_key, client_id=client_id)
        snap = SnapTrade(configuration=cfg)
    except (RuntimeError, ValueError) as exc:
        return {"status": "error", "error": f"snaptrade init: {exc}"}

    balances_by_acct: dict[str, dict[str, Any]] = {}
    positions: list[dict[str, Any]] = []
    in_positions = 0.0
    total_value = 0.0

    try:
        accounts_resp = snap.account_information.list_user_accounts(
            user_id=user_id, user_secret=user_secret,
        )
        accounts = getattr(accounts_resp, "body", accounts_resp) or []
    except (RuntimeError, OSError, ValueError) as exc:
        return {"status": "error", "error": f"list_user_accounts: {exc}"}

    if not accounts:
        return {
            "status": "not_connected",
            "error": "SnapTrade user has no linked brokerage. Run: python scripts/_setup_snaptrade.py --connect",
        }

    for acct in accounts:
        acct_id = str(getattr(acct, "id", None) or (acct.get("id") if isinstance(acct, dict) else "unknown"))
        acct_name = getattr(acct, "name", None) or (acct.get("name") if isinstance(acct, dict) else acct_id)

        # Balances
        try:
            bal_resp = snap.account_information.get_user_account_balance(
                user_id=user_id, user_secret=user_secret, account_id=acct_id,
            )
            bal_list = getattr(bal_resp, "body", bal_resp) or []
            for b in bal_list:
                bd = b if isinstance(b, dict) else {}
                ccy_obj = bd.get("currency") if bd else getattr(b, "currency", None)
                if isinstance(ccy_obj, dict):
                    ccy = ccy_obj.get("code") or "CAD"
                elif hasattr(ccy_obj, "code"):
                    ccy = ccy_obj.code
                else:
                    ccy = str(ccy_obj or "CAD")
                cash = float(bd.get("cash") if bd else getattr(b, "cash", 0) or 0)
                buying_power = float(bd.get("buying_power") if bd else getattr(b, "buying_power", 0) or 0)
                # SnapTrade balance object has no `total` — use cash as the
                # account-level liquid balance; positions contribute via market_value below.
                total = cash
                key = f"{acct_name}_{ccy}"
                balances_by_acct[key] = {
                    "total": round(total, 2),
                    "cash": round(cash, 2),
                    "buying_power": round(buying_power, 2),
                    "currency": ccy,
                }
                total_value += total
        except (RuntimeError, OSError, ValueError) as exc:
            balances_by_acct[f"{acct_name}_ERR"] = {"error": str(exc)}

        # Positions
        try:
            pos_resp = snap.account_information.get_user_account_positions(
                user_id=user_id, user_secret=user_secret, account_id=acct_id,
            )
            pos_list = getattr(pos_resp, "body", pos_resp) or []
            for pos in pos_list:
                pd = pos if isinstance(pos, dict) else {}
                sym_obj = pd.get("symbol") if pd else getattr(pos, "symbol", None)
                # SnapTrade nests UniversalSymbol either directly or one level deep
                # (BrokerageSymbol → symbol → UniversalSymbol). Walk both shapes.
                if isinstance(sym_obj, dict):
                    inner = sym_obj.get("symbol")
                    if isinstance(inner, dict):
                        sym = inner.get("symbol") or inner.get("raw_symbol") or "?"
                    elif isinstance(inner, str):
                        sym = inner
                    else:
                        sym = sym_obj.get("raw_symbol") or "?"
                else:
                    sym = getattr(sym_obj, "symbol", None) or str(sym_obj or "?")
                qty = float(pd.get("units", 0) if pd else getattr(pos, "units", 0) or 0)
                price = float(pd.get("price", 0) if pd else getattr(pos, "price", 0) or 0)
                avg = float(pd.get("average_purchase_price", 0) if pd else getattr(pos, "average_purchase_price", 0) or 0)
                mv = float(pd.get("market_value", 0) if pd else getattr(pos, "market_value", 0) or 0)
                # SnapTrade often returns market_value=0; recompute from price * qty when so
                if mv == 0 and price and qty:
                    mv = price * qty
                pnl = (price - avg) * qty if avg and price else 0.0
                in_positions += abs(mv)
                positions.append({
                    "account": acct_name,
                    "symbol": sym,
                    "qty": qty,
                    "avgCost": round(avg, 4),
                    "marketPrice": round(price, 4),
                    "marketValue": round(mv, 2),
                    "unrealizedPNL": round(pnl, 2),
                })
        except (RuntimeError, OSError, ValueError) as exc:
            positions.append({"account": acct_name, "error": str(exc)})

    return {
        "status": "ok",
        "balance": round(total_value, 2),
        "currency": "CAD",
        "total_assets": round(total_value + in_positions, 2),
        "in_positions": round(in_positions, 2),
        "platform": "WealthSimple",
        "account_id": "snaptrade",
        "by_account": balances_by_acct,
        "note": f"Live WealthSimple via SnapTrade — {len(accounts)} account(s), {len(positions)} position(s)",
        "verified": dt.date.today().isoformat(),
        "positions": positions,
    }


# ── Merge & write ───────────────────────────────────────────────────────────


def _load_existing() -> dict[str, Any]:
    if not _BAL_PATH.exists():
        return {"_meta": {}, "accounts": {}, "fx": {"cad_usd": 0.72}}
    try:
        return json.loads(_BAL_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("balances_load_failed", err=str(exc))
        return {"_meta": {}, "accounts": {}, "fx": {"cad_usd": 0.72}}


def _atomic_write(payload: dict[str, Any]) -> None:
    _BAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _BAL_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(_BAL_PATH)


def refresh_portfolio_live(write: bool = True) -> dict[str, Any]:
    """Pull live snapshots from IBKR + Moomoo + WealthSimple, optionally persist.

    Returns a status dict:
        {
          "ts": ISO timestamp,
          "sources": {ibkr/moomoo/wealthsimple: {status, error?, total_assets?, positions_count?}},
          "wrote": bool,
          "path": str,
          "duration_seconds": float,
        }
    """
    # Single in-process gate so concurrent dashboard requests don't race
    if not _REFRESH_LOCK.acquire(blocking=False):
        return {"status": "busy", "note": "another refresh in progress"}

    started = time.time()
    sources: dict[str, dict[str, Any]] = {}

    try:
        # IBKR (async under the hood — runs on its own loop)
        try:
            sources["ibkr"] = _refresh_ibkr()
        except Exception as exc:  # noqa: BLE001  — top-level safety
            sources["ibkr"] = {"status": "error", "error": f"unhandled: {exc}"}

        # Moomoo (sync)
        try:
            sources["moomoo"] = _refresh_moomoo()
        except Exception as exc:  # noqa: BLE001
            sources["moomoo"] = {"status": "error", "error": f"unhandled: {exc}"}

        # WealthSimple via SnapTrade (sync)
        try:
            sources["wealthsimple"] = _refresh_wealthsimple()
        except Exception as exc:  # noqa: BLE001
            sources["wealthsimple"] = {"status": "error", "error": f"unhandled: {exc}"}

        # NDAX via ccxt (sync)
        try:
            sources["ndax"] = _refresh_ndax()
        except Exception as exc:  # noqa: BLE001
            sources["ndax"] = {"status": "error", "error": f"unhandled: {exc}"}

        wrote = False
        if write:
            existing = _load_existing()
            existing.setdefault("accounts", {})
            existing.setdefault("fx", {"cad_usd": 0.72})

            for key in ("ibkr", "moomoo", "wealthsimple", "ndax"):
                src = sources.get(key, {})
                if src.get("status") == "ok":
                    # Only fields we don't want to clobber on each refresh
                    prior = existing["accounts"].get(key, {})
                    new_acct = {k: v for k, v in src.items() if k not in ("status", "error")}
                    # Preserve any prior bookkeeping fields not produced live
                    for keep in ("notes_history", "tags"):
                        if keep in prior:
                            new_acct[keep] = prior[keep]
                    existing["accounts"][key] = new_acct

            existing.setdefault("_meta", {})
            existing["_meta"]["updated"] = dt.datetime.now().isoformat()
            existing["_meta"]["source"] = "live_portfolio_refresh"
            existing["_meta"]["live_refresh"] = {
                k: {"status": v.get("status"), "error": v.get("error"),
                    "positions": len(v.get("positions", []) or []),
                    "total_assets": v.get("total_assets")}
                for k, v in sources.items()
            }

            try:
                _atomic_write(existing)
                wrote = True
            except OSError as exc:
                _log.warning("balances_write_failed", err=str(exc))

        # Trim the per-source dicts in the return value (don't echo full positions back)
        summary = {
            k: {
                "status": v.get("status"),
                "error": v.get("error"),
                "total_assets": v.get("total_assets"),
                "currency": v.get("currency"),
                "positions_count": len(v.get("positions", []) or []),
                "verified": v.get("verified"),
                "setup_hint": v.get("setup_hint"),
            }
            for k, v in sources.items()
        }

        return {
            "ts": dt.datetime.now().isoformat(),
            "sources": summary,
            "wrote": wrote,
            "path": str(_BAL_PATH),
            "duration_seconds": round(time.time() - started, 2),
        }
    finally:
        _REFRESH_LOCK.release()


# ── CLI ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Refresh AAC live portfolio data")
    parser.add_argument("--no-write", action="store_true", help="Do not persist to JSON")
    parser.add_argument("--quiet", action="store_true", help="Print only the JSON status")
    args = parser.parse_args()

    result = refresh_portfolio_live(write=not args.no_write)
    if not args.quiet:
        print(f"Refresh complete in {result.get('duration_seconds', 0)}s "
              f"(wrote={result.get('wrote')}) -> {result.get('path')}")
    print(json.dumps(result, indent=2, default=str))
