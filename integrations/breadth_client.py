#!/usr/bin/env python3
"""
NYSE Market Breadth Client
==========================
Free market breadth indicators via yfinance (no API key required).

Indices pulled:
    ^TRIN   — Arms Index / Short-Term Trading Index
    ^TICK   — NYSE TICK (last-trade up vs down)
    ^ADD    — NYSE Advances minus Declines
    ^DECL   — NYSE Declines (raw count)

Derived:
    advance_decline_ratio = ADV / DECL
    mcclellan_oscillator  = EMA(ADV-DECL, 19) - EMA(ADV-DECL, 39)

Source: Yahoo Finance (delayed ~15 min; close-of-day for breadth aggregates).

Usage:
    from integrations.breadth_client import BreadthClient
    bc = BreadthClient()
    snap = bc.get_snapshot()         # current values + classification
    series = bc.get_history(days=60) # historical breadth for McClellan
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Yahoo Finance ticker symbols for NYSE breadth
BREADTH_TICKERS = {
    "trin": "^TRIN",
    "tick": "^TICK",
    "adv_minus_decl": "^ADD",   # NYSE Advances - Declines
    "declines": "^DECL",        # NYSE Declines (raw)
    "advances": "^ADV",         # NYSE Advances (raw)
}


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
        yf = self._yfinance()
        try:
            hist = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
        except (ValueError, KeyError, ConnectionError, TimeoutError) as exc:
            logger.warning("breadth: %s history failed: %s", ticker, exc)
            return None
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        closes = hist["Close"].dropna()
        if closes.empty:
            return None
        return float(closes.iloc[-1])

    def _close_series(self, ticker: str, days: int) -> List[float]:
        yf = self._yfinance()
        try:
            hist = yf.Ticker(ticker).history(period=f"{max(days, 60)}d", auto_adjust=False)
        except (ValueError, KeyError, ConnectionError, TimeoutError) as exc:
            logger.warning("breadth: %s series failed: %s", ticker, exc)
            return []
        if hist is None or hist.empty or "Close" not in hist.columns:
            return []
        return [float(v) for v in hist["Close"].dropna().tolist()]

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
        """Fetch a current breadth snapshot with derived metrics + regime."""
        snap = BreadthSnapshot(timestamp=datetime.utcnow().isoformat())
        try:
            snap.trin = self._last_close(BREADTH_TICKERS["trin"])
            snap.tick = self._last_close(BREADTH_TICKERS["tick"])
            snap.advances = self._last_close(BREADTH_TICKERS["advances"])
            snap.declines = self._last_close(BREADTH_TICKERS["declines"])
            snap.adv_minus_decl = self._last_close(BREADTH_TICKERS["adv_minus_decl"])

            if (
                snap.advances is not None
                and snap.declines is not None
                and snap.declines > 0
            ):
                snap.advance_decline_ratio = snap.advances / snap.declines

            # McClellan needs at least 39 daily samples of (ADV-DECL)
            series = self._close_series(BREADTH_TICKERS["adv_minus_decl"], days=60)
            if len(series) >= 39:
                snap.mcclellan_oscillator = self.mcclellan_oscillator(series)
            else:
                snap.notes.append(
                    f"McClellan needs >=39 samples; got {len(series)}"
                )

            snap.regime = self._classify(snap)
            return snap
        except (RuntimeError, ValueError) as exc:
            snap.error = str(exc)
            return snap

    def get_history(self, days: int = 60) -> Dict[str, List[float]]:
        """Return parallel close-price arrays for all breadth tickers."""
        return {
            label: self._close_series(ticker, days=days)
            for label, ticker in BREADTH_TICKERS.items()
        }


__all__ = ["BreadthClient", "BreadthSnapshot", "BREADTH_TICKERS"]
