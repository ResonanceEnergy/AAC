"""
Dashboard Pillars — focused collectors for the revamped Streamlit dashboard.

Three pillars, one collector each:

    1. collect_call_options() — IV/HV readings, vol-premium signals, own call
       positions with Greeks, covered-call screen on owned underlyings.
    2. collect_index_flow()   — index ETF flows (SPY/QQQ/IWM/DIA), UW market
       tide + dark pool, GEX walls, NYSE breadth, COT positioning.
    3. collect_quant_research() — vol premium aggregate, signal journal hit
       rates, walk-forward backtest results (gated; runs only on demand).

Each collector is fail-soft: any sub-call failure becomes a string in the
returned dict's ``errors`` list rather than an exception.
"""

from __future__ import annotations

import os
import time
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

# Env switches — set to "1" to re-enable known-flaky sources.
_ENABLE_CFTC = os.environ.get("AAC_DASHBOARD_ENABLE_CFTC", "0") == "1"
_ENABLE_BREADTH = os.environ.get("AAC_DASHBOARD_ENABLE_BREADTH", "0") == "1"


# ── Helpers ────────────────────────────────────────────────────────────────


def _safe(label: str, fn, *args, errors: list[str], **kwargs) -> Any:
    """Run ``fn(*args, **kwargs)``; on failure, append to errors and return None."""
    try:
        return fn(*args, **kwargs)
    except (RuntimeError, OSError, ValueError, TypeError, KeyError, AttributeError, ImportError) as exc:
        msg = f"{label}: {type(exc).__name__}: {exc}"
        _log.warning("pillar_collector_failed", label=label, error=str(exc))
        errors.append(msg)
        return None


# ── Pillar A — Call Options ────────────────────────────────────────────────


# Sensible call-options watchlist: own-portfolio names plus liquid index/sector
# tickers where call strategies are routinely deployed.
_CALL_OPTIONS_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",          # Indices
    "GLD", "SLV", "USO",                  # Commodities
    "XLE", "XLF", "XLK", "XLV", "XLY",   # Sector SPDRs
    "TSLA", "NVDA", "AAPL", "MSFT",      # Mega-cap names
    "OWL",                                # WS TFSA holding
]


def collect_call_options(own_call_positions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Pillar A — call options research, flow, and own-position Greeks.

    Args:
        own_call_positions: List of position dicts (account/symbol/strike/qty/etc.)
            already filtered to type == "call". Optional — used to compute
            covered-call yield + Greeks roll-ups for what we actually own.
    """
    errors: list[str] = []
    started = time.time()

    # IV/HV vol-premium readings (yfinance) — heavy, cache upstream.
    readings = _safe(
        "vol_premium_readings",
        _vol_premium_readings,
        _CALL_OPTIONS_UNIVERSE,
        errors=errors,
    ) or []

    # Sort by IV/HV ratio descending (richest premium first).
    # Readings may be VolPremiumReading dataclasses OR dicts — _ratio handles both.
    readings_sorted = sorted(readings, key=_ratio, reverse=True)

    # Top call positions enrichment (Greeks lookup if available).
    own_calls = own_call_positions or []
    own_call_summary = _summarise_own_calls(own_calls)

    # Covered-call screener on liquid names where we hold the underlying.
    cc_candidates = _safe(
        "covered_call_screener",
        _build_covered_call_screen,
        own_calls,
        errors=errors,
    ) or []

    return {
        "as_of": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": round(time.time() - started, 2),
        "universe_size": len(_CALL_OPTIONS_UNIVERSE),
        "vol_premium_readings": [_reading_to_dict(r) for r in readings_sorted],
        "rich_premium_count": sum(1 for r in readings if _ratio(r) >= 1.20),
        "cheap_premium_count": sum(1 for r in readings if 0 < _ratio(r) <= 0.85),
        "own_call_summary": own_call_summary,
        "covered_call_candidates": cc_candidates,
        "errors": errors,
    }


def _ratio(r: Any) -> float:
    return getattr(r, "iv_hv_ratio", 0.0) if hasattr(r, "iv_hv_ratio") else float(r.get("iv_hv_ratio", 0.0))


def _vol_premium_readings(universe: list[str]) -> list[Any]:
    # Try IBKR-native IV/HV first (free, fast — one batched reqMktData call).
    ibkr_rows = _try_ibkr_iv_hv(universe)
    if ibkr_rows:
        return ibkr_rows
    # Fallback: yfinance per-ticker chain scrape.
    from strategies.vol_premium_signals import get_vol_premium_readings  # noqa: PLC0415
    return get_vol_premium_readings(tickers=universe, fetch_iv=True)


def _try_ibkr_iv_hv(universe: list[str]) -> list[dict[str, Any]]:
    """Return Pillar-A-shaped readings from IBKR, or [] if unavailable."""
    try:
        from integrations.ibkr_market_data_client import get_iv_hv_snapshot, is_available  # noqa: PLC0415
    except ImportError:
        return []
    try:
        if not is_available():
            return []
        snap = get_iv_hv_snapshot(universe) or {}
    except (RuntimeError, OSError, ValueError) as exc:
        _log.debug("ibkr_iv_hv_fallback", error=str(exc))
        return []

    out: list[dict[str, Any]] = []
    for sym in universe:
        row = snap.get(sym) or {}
        out.append({
            "ticker": sym,
            "spot": row.get("spot") or 0.0,
            "realized_hv": row.get("realized_hv") or 0.0,
            "implied_vol": row.get("implied_vol") or 0.0,
            "iv_hv_ratio": row.get("iv_hv_ratio") or 0.0,
            "option_available": bool(row.get("option_available")),
        })
    return out


def _reading_to_dict(r: Any) -> dict[str, Any]:
    if isinstance(r, dict):
        return r
    return {
        "ticker": getattr(r, "ticker", "?"),
        "spot": getattr(r, "spot", 0.0),
        "realized_hv": getattr(r, "realized_hv", 0.0),
        "implied_vol": getattr(r, "implied_vol", 0.0),
        "iv_hv_ratio": getattr(r, "iv_hv_ratio", 0.0),
        "option_available": getattr(r, "option_available", False),
    }


def _summarise_own_calls(own_calls: list[dict[str, Any]]) -> dict[str, Any]:
    if not own_calls:
        return {"position_count": 0, "total_market_value": 0.0, "total_unrealized_pnl": 0.0}
    total_mv = sum(_num(p.get("market_value", 0)) for p in own_calls)
    total_pnl = sum(_num(p.get("unrealized_pnl", 0)) for p in own_calls)
    by_underlying: dict[str, dict[str, Any]] = {}
    for p in own_calls:
        sym = str(p.get("symbol", "?"))
        bucket = by_underlying.setdefault(sym, {"contracts": 0, "market_value": 0.0, "pnl": 0.0})
        bucket["contracts"] += int(_num(p.get("qty", 0)))
        bucket["market_value"] += _num(p.get("market_value", 0))
        bucket["pnl"] += _num(p.get("unrealized_pnl", 0))
    return {
        "position_count": len(own_calls),
        "total_market_value": round(total_mv, 2),
        "total_unrealized_pnl": round(total_pnl, 2),
        "by_underlying": [
            {"symbol": k, **{k2: round(v2, 2) if isinstance(v2, float) else v2 for k2, v2 in v.items()}}
            for k, v in sorted(by_underlying.items(), key=lambda kv: kv[1]["market_value"], reverse=True)
        ],
    }


def _build_covered_call_screen(own_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Screen owned long-call underlyings for income-CC candidates.

    Uses yfinance front-month chains; falls back gracefully when chain
    data is unavailable.
    """
    if not own_calls:
        return []

    try:
        import yfinance as yf  # noqa: PLC0415

        from strategies.options_income_systems import CoveredCallScreener  # noqa: PLC0415
    except ImportError:
        return []

    underlyings = sorted({str(p.get("symbol", "")).upper() for p in own_calls if p.get("symbol")})
    candidates: list[dict[str, Any]] = []
    for sym in underlyings[:10]:  # cap to keep API calls bounded
        try:
            t = yf.Ticker(sym)
            spot_hist = t.history(period="2d")
            if spot_hist.empty:
                continue
            spot = float(spot_hist["Close"].iloc[-1])
            exps = t.options
            if not exps:
                continue
            chain = t.option_chain(exps[0]).calls
            if chain.empty:
                continue
            chain = chain.copy()
            chain["_dist"] = (chain["strike"] - spot * 1.05).abs()  # ~5% OTM target
            row = chain.loc[chain["_dist"].idxmin()]
            candidates.append({
                "symbol": sym,
                "price": round(spot, 2),
                "call_premium": float(row.get("lastPrice", 0.0) or 0.0),
                "call_delta": 0.30,  # placeholder — yfinance doesn't expose Greeks
                "iv": float(row.get("impliedVolatility", 0.0) or 0.0),
                "dte": 30,
                "div_yield": 0.0,
                "earnings_in_cycle": False,
            })
        except (RuntimeError, OSError, ValueError, KeyError, AttributeError):
            continue

    return CoveredCallScreener.screen(candidates, mode="income") if candidates else []


# ── Pillar B — Index Flow ──────────────────────────────────────────────────


_INDEX_FLOW_TICKERS = ["SPY", "QQQ", "IWM", "DIA"]


def collect_index_flow(uw_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pillar B — index ETF flows, options flow, GEX walls, breadth, COT."""
    errors: list[str] = []
    started = time.time()

    # ETF flows — index-only subset.
    etf_flows = _safe("etf_flow", _etf_flow_for_universe, _INDEX_FLOW_TICKERS, errors=errors) or []

    # NYSE breadth (Yahoo ^TRIN delisted; gated by AAC_DASHBOARD_ENABLE_BREADTH).
    breadth: dict[str, Any] = {}
    if _ENABLE_BREADTH:
        breadth = _safe("breadth", _breadth_snapshot, errors=errors) or {}

    # COT positioning (CFTC zip 404 for 2025/2026; gated by AAC_DASHBOARD_ENABLE_CFTC).
    cot: list[dict[str, Any]] = []
    if _ENABLE_CFTC:
        cot = _safe("cot_positioning", _cot_positioning, errors=errors) or []

    # Reuse the live UW payload that the dashboard already collects each cycle.
    uw = uw_payload or {}
    market_tide = uw.get("market_tide_latest") or {}
    gex_walls = uw.get("gex_walls") or {}
    dark_pool_total = uw.get("dark_pool_notional", 0.0)
    pcr = uw.get("put_call_ratio", 0.0)
    market_tone = uw.get("market_tone", "unknown")

    return {
        "as_of": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": round(time.time() - started, 2),
        "etf_flows": etf_flows,
        "breadth": breadth,
        "cot_positioning": cot,
        "options_flow": {
            "market_tone": market_tone,
            "put_call_ratio": pcr,
            "market_tide": market_tide,
            "net_call_premium": uw.get("market_tide_net_call_premium", 0.0),
            "net_put_premium": uw.get("market_tide_net_put_premium", 0.0),
            "dark_pool_notional": dark_pool_total,
            "dark_pool_trade_count": uw.get("dark_pool_trade_count", 0),
            "top_flow_tickers": uw.get("top_flow_tickers", []),
            "options_flow_signal_count": uw.get("options_flow_signal_count", 0),
        },
        "gex_walls": gex_walls,
        "errors": errors,
    }


def _etf_flow_for_universe(tickers: list[str]) -> list[dict[str, Any]]:
    from integrations.etf_flow_client import ETFFlowClient  # noqa: PLC0415

    client = ETFFlowClient()
    out: list[dict[str, Any]] = []
    for sym in tickers:
        try:
            snap = client.get_snapshot(sym, persist=True)
            d = snap.to_dict()
            out.append({
                "symbol": d.get("symbol", sym),
                "date": d.get("date", ""),
                "price": d.get("nav_or_price"),
                "shares_outstanding": d.get("shares_outstanding"),
                "total_assets": d.get("total_assets"),
                "daily_flow_usd": d.get("daily_flow_usd"),
                "error": d.get("error"),
            })
        except (RuntimeError, OSError, ValueError, KeyError, AttributeError) as exc:
            out.append({"symbol": sym, "error": str(exc)})
    return out


def _breadth_snapshot() -> dict[str, Any]:
    from integrations.breadth_client import BreadthClient  # noqa: PLC0415

    client = BreadthClient()
    snap = client.get_snapshot()
    return snap.to_dict() if hasattr(snap, "to_dict") else dict(snap.__dict__)


def _cot_positioning() -> list[dict[str, Any]]:
    from integrations.cftc_cot_client import CFTCCotClient  # noqa: PLC0415

    client = CFTCCotClient()
    out: list[dict[str, Any]] = []
    for market in ("ES", "NQ", "RTY", "YM", "VX"):
        try:
            latest = client.get_latest(market)
            if latest is None:
                continue
            sig = client.get_extreme_signal(market)
            row = latest.to_dict()
            row["leveraged_net"] = latest.leveraged_net
            row["asset_mgr_net"] = latest.asset_mgr_net
            row["dealer_net"] = latest.dealer_net
            row["extreme_signal"] = sig.to_dict() if sig else None
            out.append(row)
        except (RuntimeError, OSError, ValueError, KeyError, AttributeError):
            continue
    return out


# ── Pillar C — Quant Research ──────────────────────────────────────────────


def collect_quant_research(
    backtest_ticker: str | None = None,
    run_walk_forward: bool = False,
) -> dict[str, Any]:
    """Pillar C — backtests, signals, hit rates.

    Walk-forward backtest is gated behind ``run_walk_forward`` because it
    costs ~5–15s per call and needs ~500 days of yfinance history.
    """
    errors: list[str] = []
    started = time.time()

    # Vol-premium signals (already a quant signal source).
    vol_signals = _safe("vol_premium_signals", _vol_premium_signal_dicts, errors=errors) or []

    # 90-day proxy backtest (cheap).
    backtest_report = _safe("simple_backtest", _simple_backtest_report, errors=errors)

    # Walk-forward (gated).
    walk_forward = None
    if run_walk_forward:
        walk_forward = _safe(
            "walk_forward",
            _walk_forward_report,
            backtest_ticker or "SPY",
            errors=errors,
        )

    # Signal journal hit rates.
    hit_rates = _safe("signal_journal_hit_rates", _signal_hit_rates, errors=errors) or {}

    return {
        "as_of": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": round(time.time() - started, 2),
        "vol_premium_signals": vol_signals,
        "vol_premium_signal_count": len(vol_signals),
        "simple_backtest": backtest_report,
        "walk_forward": walk_forward,
        "hit_rates": hit_rates,
        "errors": errors,
    }


def _vol_premium_signal_dicts() -> list[dict[str, Any]]:
    from strategies.vol_premium_signals import generate_vol_premium_signals  # noqa: PLC0415

    sigs = generate_vol_premium_signals()
    out: list[dict[str, Any]] = []
    for s in sigs:
        out.append({
            "ticker": getattr(s, "ticker", getattr(s, "symbol", "?")),
            "direction": getattr(s, "direction", getattr(s, "side", "?")),
            "confidence": round(float(getattr(s, "confidence", 0.0)), 3),
            "size_pct": round(float(getattr(s, "size_pct", 0.0)) * 100, 2),
            "strategy": getattr(s, "strategy", "vol_premium"),
            "rationale": str(getattr(s, "rationale", ""))[:160],
        })
    return out


def _simple_backtest_report() -> dict[str, Any]:
    from strategies.simple_backtest import run_backtest  # noqa: PLC0415

    report = run_backtest()
    return report.to_dict() if hasattr(report, "to_dict") else {"error": "no to_dict"}


def _walk_forward_report(ticker: str) -> dict[str, Any]:
    from strategies.simple_backtest import run_walk_forward  # noqa: PLC0415

    return run_walk_forward(ticker=ticker)


def _signal_hit_rates() -> dict[str, Any]:
    from strategies.signal_journal import SignalJournal  # noqa: PLC0415

    journal = SignalJournal()
    try:
        rates = journal.get_hit_rates() or {}
        return {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in rates.items()}
    finally:
        try:
            journal.close()
        except (RuntimeError, OSError, AttributeError):
            pass


# ── Util ───────────────────────────────────────────────────────────────────


def _num(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
