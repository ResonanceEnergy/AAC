#!/usr/bin/env python3
"""
NYSE Market Breadth Client
==========================
Free market breadth indicators with graceful degradation.

Primary source: Stooq daily CSV (no key, no ticker delisting).
    https://stooq.com/q/d/?s=<symbol>&i=d

Stooq symbols used:
    ^trin   -> Arms Index (TRIN)
    ^tick   -> NYSE TICK (intraday last)
    ^add    -> NYSE Advances - Declines

Fallback source: yfinance (^TRIN / ^TICK / ^ADV / ^DECL / ^ADD).
    NOTE: As of 2026-Q2 Yahoo dropped most of these tickers and they return
    "possibly delisted; no price data found". The client now caches a
    "source unavailable" verdict for an hour so the dashboard doesn't spam
    the log every minute.

Derived:
    advance_decline_ratio = ADV / DECL  (only if both available)
    mcclellan_oscillator  = EMA(ADV-DECL, 19) - EMA(ADV-DECL, 39)

Usage:
    from integrations.breadth_client import BreadthClient
    bc = BreadthClient()
    snap = bc.get_snapshot()
"""
from __future__ import annotations

import csv
import io
import logging
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _try_ibkr_breadth() -> dict | None:
    """Pull TICK/TRIN/AD/DC from IBKR if AAC_DATA_PRIMARY=ibkr and TWS up."""
    try:
        from integrations.ibkr_market_data_client import get_breadth_snapshot, is_available
    except ImportError:
        return None
    try:
        if not is_available():
            return None
        return get_breadth_snapshot()
    except (RuntimeError, OSError, ValueError) as exc:
        logger.debug("ibkr breadth fallback: %s", exc)
        return None

# Yahoo Finance ticker symbols for NYSE breadth (legacy fallback only).
BREADTH_TICKERS = {
    "trin": "^TRIN",
    "tick": "^TICK",
    "adv_minus_decl": "^ADD",   # NYSE Advances - Declines
    "declines": "^DECL",        # NYSE Declines (raw)
    "advances": "^ADV",         # NYSE Advances (raw)
}

# Stooq symbol map (lower-case, no caret).
STOOQ_SYMBOLS = {
    "trin": "^trin",
    "tick": "^tick",
    "adv_minus_decl": "^add",
}

# Cache "source unavailable" verdict for this many seconds so we don't spam
# yfinance / network on every dashboard cycle.
_SOURCE_FAILURE_TTL_SECONDS = 3600

# Module-level state so multiple BreadthClient instances share the cooldown.
_yahoo_unavailable_until: float = 0.0
_yahoo_failure_logged: bool = False
_stooq_unavailable_until: float = 0.0
_stooq_failure_logged: bool = False


def _yahoo_in_cooldown() -> bool:
    return time.time() < _yahoo_unavailable_until


def _stooq_in_cooldown() -> bool:
    return time.time() < _stooq_unavailable_until


def _trip_yahoo_cooldown(reason: str) -> None:
    global _yahoo_unavailable_until, _yahoo_failure_logged
    _yahoo_unavailable_until = time.time() + _SOURCE_FAILURE_TTL_SECONDS
    if not _yahoo_failure_logged:
        logger.warning(
            "breadth: Yahoo source disabled for %ds (%s)",
            _SOURCE_FAILURE_TTL_SECONDS,
            reason,
        )
        _yahoo_failure_logged = True


def _trip_stooq_cooldown(reason: str) -> None:
    global _stooq_unavailable_until, _stooq_failure_logged
    _stooq_unavailable_until = time.time() + _SOURCE_FAILURE_TTL_SECONDS
    if not _stooq_failure_logged:
        logger.warning(
            "breadth: Stooq source disabled for %ds (%s)",
            _SOURCE_FAILURE_TTL_SECONDS,
            reason,
        )
        _stooq_failure_logged = True


@dataclass
class BreadthSnapshot:
    """Point-in-time NYSE breadth reading."""

    timestamp: str
    trin: Optional[float] = None
    tick: Optional[float] = None
    advances: Optional[float] = None
    declines: Optional[float] = None
    adv_minus_decl: Optional[float] = None
    advance_decline_ratio: Optional[float] = None
    mcclellan_oscillator: Optional[float] = None
    regime: str = "unknown"       # bullish / bearish / neutral / unknown
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BreadthClient:
    """yfinance-backed NYSE breadth indicator client.

    yfinance is synchronous; methods are intentionally sync to avoid
    wrapping blocking IO in a fake async shell.
    """

    def __init__(self) -> None:
        self._yf = None

    def _yfinance(self):
        if self._yf is None:
            try:
                import yfinance as yf  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    "yfinance not installed; add to requirements.txt"
                ) from exc
            self._yf = yf
        return self._yf

    def _last_close(self, ticker: str) -> Optional[float]:
        """Return the most recent non-NaN close for a Yahoo ticker."""
        if _yahoo_in_cooldown():
            return None
        yf = self._yfinance()
        try:
            hist = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
        except (ValueError, KeyError, ConnectionError, TimeoutError) as exc:
            logger.warning("breadth: %s history failed: %s", ticker, exc)
            return None
        if hist is None or hist.empty or "Close" not in hist.columns:
            # Yahoo silently returns empty when a symbol is delisted; treat as
            # a dead source and short-circuit subsequent calls for an hour.
            _trip_yahoo_cooldown(f"{ticker} returned no data")
            return None
        closes = hist["Close"].dropna()
        if closes.empty:
            _trip_yahoo_cooldown(f"{ticker} returned no closes")
            return None
        return float(closes.iloc[-1])

    def _close_series(self, ticker: str, days: int) -> List[float]:
        if _yahoo_in_cooldown():
            return []
        yf = self._yfinance()
        try:
            hist = yf.Ticker(ticker).history(period=f"{max(days, 60)}d", auto_adjust=False)
        except (ValueError, KeyError, ConnectionError, TimeoutError) as exc:
            logger.warning("breadth: %s series failed: %s", ticker, exc)
            return []
        if hist is None or hist.empty or "Close" not in hist.columns:
            _trip_yahoo_cooldown(f"{ticker} series returned no data")
            return []
        return [float(v) for v in hist["Close"].dropna().tolist()]

    # ------------------------------------------------------------------
    # Stooq fallback (no API key, no auth, daily CSV)
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_stooq_csv(symbol: str) -> List[Dict[str, str]]:
        """Fetch Stooq daily CSV and return rows. Empty on any failure."""
        if _stooq_in_cooldown():
            return []
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "AAC/3.6.0 BreadthClient"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    _trip_stooq_cooldown(f"HTTP {resp.status} for {symbol}")
                    return []
                body = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            _trip_stooq_cooldown(f"{symbol}: {exc}")
            return []
        if not body or body.lower().startswith("no data"):
            return []
        reader = csv.DictReader(io.StringIO(body))
        return list(reader)

    def _stooq_last_close(self, key: str) -> Optional[float]:
        symbol = STOOQ_SYMBOLS.get(key)
        if not symbol:
            return None
        rows = self._fetch_stooq_csv(symbol)
        if not rows:
            return None
        # Stooq returns ascending dates; last row is most recent close.
        last = rows[-1]
        try:
            return float(last.get("Close") or last.get("close") or "")
        except (TypeError, ValueError):
            return None

    def _stooq_close_series(self, key: str, days: int) -> List[float]:
        symbol = STOOQ_SYMBOLS.get(key)
        if not symbol:
            return []
        rows = self._fetch_stooq_csv(symbol)
        if not rows:
            return []
        closes: List[float] = []
        for row in rows[-max(days, 60):]:
            raw = row.get("Close") or row.get("close")
            if raw in (None, ""):
                continue
            try:
                closes.append(float(raw))
            except (TypeError, ValueError):
                continue
        return closes

    def _resolve_last_close(self, key: str) -> Optional[float]:
        """Try Stooq first, then Yahoo, returning None if both unavailable."""
        value = self._stooq_last_close(key)
        if value is not None:
            return value
        ticker = BREADTH_TICKERS.get(key)
        if not ticker:
            return None
        return self._last_close(ticker)

    def _resolve_close_series(self, key: str, days: int) -> List[float]:
        series = self._stooq_close_series(key, days=days)
        if series:
            return series
        ticker = BREADTH_TICKERS.get(key)
        if not ticker:
            return []
        return self._close_series(ticker, days=days)

    @staticmethod
    def _ema(values: List[float], period: int) -> Optional[float]:
        """Exponential moving average; returns last value or None if insufficient data."""
        if len(values) < period:
            return None
        k = 2.0 / (period + 1)
        ema = values[0]
        for v in values[1:]:
            ema = v * k + ema * (1 - k)
        return ema

    def mcclellan_oscillator(self, adv_minus_decl_series: List[float]) -> Optional[float]:
        """McClellan Oscillator = EMA(19) - EMA(39) of (Advances - Declines).

        Standard interpretation:
          > +100  : overbought
          0..+100 : bullish
          -100..0 : bearish
          < -100  : oversold
        """
        ema19 = self._ema(adv_minus_decl_series, 19)
        ema39 = self._ema(adv_minus_decl_series, 39)
        if ema19 is None or ema39 is None:
            return None
        return ema19 - ema39

    @staticmethod
    def _classify(snap: "BreadthSnapshot") -> str:
        """Classify regime from TRIN + adv/decl ratio + McClellan."""
        votes = {"bullish": 0, "bearish": 0}

        # TRIN: <0.8 strong buying, >1.2 strong selling
        if snap.trin is not None:
            if snap.trin < 0.8:
                votes["bullish"] += 1
            elif snap.trin > 1.2:
                votes["bearish"] += 1

        # Adv/Decl ratio: >1.5 bullish, <0.67 bearish
        if snap.advance_decline_ratio is not None:
            if snap.advance_decline_ratio > 1.5:
                votes["bullish"] += 1
            elif snap.advance_decline_ratio < 0.67:
                votes["bearish"] += 1

        # McClellan: >0 bullish, <0 bearish
        if snap.mcclellan_oscillator is not None:
            if snap.mcclellan_oscillator > 0:
                votes["bullish"] += 1
            elif snap.mcclellan_oscillator < 0:
                votes["bearish"] += 1

        if votes["bullish"] >= 2 and votes["bullish"] > votes["bearish"]:
            return "bullish"
        if votes["bearish"] >= 2 and votes["bearish"] > votes["bullish"]:
            return "bearish"
        if votes["bullish"] == 0 and votes["bearish"] == 0:
            return "unknown"
        return "neutral"

    def get_snapshot(self) -> BreadthSnapshot:
        """Fetch a current breadth snapshot with derived metrics + regime.

        Tries IBKR (TICK-NYSE / TRIN-NYSE indices) first if AAC_DATA_PRIMARY=ibkr
        and TWS is reachable; falls back to the legacy Yahoo/Stooq path.
        """
        # ── IBKR primary path ────────────────────────────────────────────
        ib_snap = _try_ibkr_breadth()
        if ib_snap is not None:
            snap = BreadthSnapshot(timestamp=ib_snap.get("timestamp", datetime.utcnow().isoformat()))
            snap.trin = ib_snap.get("trin")
            snap.tick = ib_snap.get("tick")
            snap.advances = ib_snap.get("advances")
            snap.declines = ib_snap.get("declines")
            snap.adv_minus_decl = ib_snap.get("adv_minus_decl")
            snap.advance_decline_ratio = ib_snap.get("advance_decline_ratio")
            snap.regime = ib_snap.get("regime", "unknown")
            snap.notes.append(ib_snap.get("notes", "ibkr"))
            return snap

        # ── Legacy yfinance/Stooq path ──────────────────────────────────
        snap = BreadthSnapshot(timestamp=datetime.utcnow().isoformat())
        try:
            snap.trin = self._resolve_last_close("trin")
            snap.tick = self._resolve_last_close("tick")
            snap.advances = self._resolve_last_close("advances")
            snap.declines = self._resolve_last_close("declines")
            snap.adv_minus_decl = self._resolve_last_close("adv_minus_decl")

            if (
                snap.advances is not None
                and snap.declines is not None
                and snap.declines > 0
            ):
                snap.advance_decline_ratio = snap.advances / snap.declines

            # McClellan needs at least 39 daily samples of (ADV-DECL)
            series = self._resolve_close_series("adv_minus_decl", days=60)
            if len(series) >= 39:
                snap.mcclellan_oscillator = self.mcclellan_oscillator(series)
            else:
                snap.notes.append(
                    f"McClellan needs >=39 samples; got {len(series)}"
                )

            # If every reading is None, surface that the source is dead so
            # callers can skip downstream calculations cleanly.
            if all(
                v is None
                for v in (snap.trin, snap.tick, snap.advances, snap.declines, snap.adv_minus_decl)
            ):
                snap.error = "breadth source unavailable (Yahoo delisted, Stooq unreachable)"
                snap.regime = "unknown"
                return snap

            snap.regime = self._classify(snap)
            return snap
        except (RuntimeError, ValueError) as exc:
            snap.error = str(exc)
            return snap

    def get_history(self, days: int = 60) -> Dict[str, List[float]]:
        """Return parallel close-price arrays for all breadth keys."""
        return {
            label: self._resolve_close_series(label, days=days)
            for label in BREADTH_TICKERS.keys()
        }


__all__ = ["BreadthClient", "BreadthSnapshot", "BREADTH_TICKERS"]
