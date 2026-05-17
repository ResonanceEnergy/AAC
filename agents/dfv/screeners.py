from __future__ import annotations

"""DFV watchlist screeners.

The watchlist is *not* a manual list — it is the latest output of the
screeners below. ``run_all()`` is called from ``eod()`` and replaces the
file at ``agents/dfv/memory/watchlist.json`` with top-N candidates per
screen.

Each screener returns a list of dicts:
    {"symbol": str, "score": float, "rationale": str}
"""

from datetime import datetime, timezone
from typing import Any, Callable

import structlog

_log = structlog.get_logger(__name__)


# Hard-coded universes. These mirror the arms in ``allocation_arms`` and
# keep the screener self-contained (no external sector-mapping service
# required). Extend in-code when a new arm goes live.
_BDC_UNIVERSE: list[str] = [
    "ARCC", "MAIN", "OBDC", "BXSL", "GBDC", "PSEC", "TSLX",
    "FSK", "GAIN", "HTGC", "OCSL", "GLAD", "TCPC", "TPVG",
]

_OIL_UNIVERSE: list[str] = [
    "XLE", "XOP", "OIH", "USO", "BNO", "XOM", "CVX", "OXY",
    "COP", "EOG", "MPC", "VLO", "PSX", "PXD", "SLB",
]

_VOL_UNIVERSE: list[str] = [
    "SPY", "QQQ", "IWM", "DIA", "^VIX", "UVXY", "VXX",
    "TSLA", "NVDA", "META", "AAPL", "MSFT", "AMZN", "GOOGL", "GME",
]


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_yf_close(symbol: str) -> tuple[float, float] | None:
    """Return (last_close, prev_close) or None if unavailable. Pure best-effort."""
    try:
        import yfinance as yf  # type: ignore  # noqa: PLC0415
        hist = yf.Ticker(symbol).history(period="5d", auto_adjust=False)
        if hist is None or len(hist) < 2:
            return None
        closes = hist["Close"].dropna().tolist()
        if len(closes) < 2:
            return None
        return float(closes[-1]), float(closes[-2])
    except Exception as exc:  # noqa: BLE001
        _log.debug("screeners.yf_close_failed", symbol=symbol, error=str(exc))
        return None


def _bdc_stress() -> list[dict[str, Any]]:
    """Rank BDCs by recent drawdown (proxy for NAV-discount widening).

    A real implementation reads NAV vs price; here we use a 5-day return
    proxy because that is the data actually available in this repo
    without paid feeds. Bigger drawdown = higher stress score = top of
    the watchlist.
    """
    results: list[dict[str, Any]] = []
    for sym in _BDC_UNIVERSE:
        pair = _safe_yf_close(sym)
        if not pair:
            continue
        last, prev = pair
        ret = (last - prev) / prev if prev else 0.0
        # Stress = negative return magnitude; positive returns get small scores.
        score = -ret * 100.0
        results.append({
            "symbol": sym,
            "score": round(score, 3),
            "rationale": f"5d return {ret:+.2%}; price {last:.2f} — proxy for NAV-discount stress",
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _oil_geopolitical() -> list[dict[str, Any]]:
    """Rank oil names by 5-day momentum (regime/geopolitical proxy).

    Use 5d return as proxy for geopolitical-driven repricing. Positive
    momentum = top of the watchlist (consistent with the iran_oil arm
    being a *long* exposure, not a hedge).
    """
    results: list[dict[str, Any]] = []
    for sym in _OIL_UNIVERSE:
        pair = _safe_yf_close(sym)
        if not pair:
            continue
        last, prev = pair
        ret = (last - prev) / prev if prev else 0.0
        results.append({
            "symbol": sym,
            "score": round(ret * 100.0, 3),
            "rationale": f"5d momentum {ret:+.2%}; price {last:.2f} — oil regime tape",
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _vol_squeeze() -> list[dict[str, Any]]:
    """Rank by 5-day absolute move (vol-regime / squeeze setup proxy).

    Without paid IV-rank feed, abs-return is the cheap stand-in for
    realized-vol breakouts. Highest |return| floats to the top.
    """
    results: list[dict[str, Any]] = []
    for sym in _VOL_UNIVERSE:
        pair = _safe_yf_close(sym)
        if not pair:
            continue
        last, prev = pair
        ret = (last - prev) / prev if prev else 0.0
        results.append({
            "symbol": sym,
            "score": round(abs(ret) * 100.0, 3),
            "rationale": f"|5d move| {abs(ret):+.2%}; price {last:.2f} — vol breakout proxy",
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


_SCREENERS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "bdc_stress": _bdc_stress,
    "oil_geopolitical": _oil_geopolitical,
    "vol_squeeze": _vol_squeeze,
}


def run_all(
    *,
    active: list[str] | None = None,
    top_n_per_screen: int = 5,
    total_cap: int = 15,
) -> dict[str, dict[str, Any]]:
    """Run each active screener, take top-N from each, merge to a watchlist dict.

    Returns the dict in the shape Watchlist.replace_all expects:
        {SYMBOL: {"added": iso, "reason": str, "source": str,
                  "score": float, "screener": name}}

    Highest-scoring screener wins if the same symbol appears in two.
    Total entries capped at ``total_cap``.
    """
    active = active or list(_SCREENERS.keys())
    merged: dict[str, dict[str, Any]] = {}
    now = _ts()
    for name in active:
        fn = _SCREENERS.get(name)
        if fn is None:
            _log.warning("screeners.unknown", name=name)
            continue
        try:
            rows = fn()[:top_n_per_screen]
        except Exception as exc:  # noqa: BLE001 — never let one screener kill EOD
            _log.warning("screeners.failed", name=name, error=str(exc))
            continue
        for row in rows:
            sym = str(row.get("symbol", "")).upper()
            if not sym:
                continue
            score = float(row.get("score") or 0.0)
            entry = {
                "added": now,
                "reason": row.get("rationale", ""),
                "source": f"screener:{name}",
                "screener": name,
                "score": score,
            }
            prior = merged.get(sym)
            if prior is None or score > float(prior.get("score") or 0.0):
                merged[sym] = entry

    # Cap by score, descending.
    ranked = sorted(merged.items(), key=lambda kv: float(kv[1].get("score") or 0.0), reverse=True)
    return dict(ranked[:total_cap])
