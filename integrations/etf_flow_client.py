#!/usr/bin/env python3
"""
ETF Flow Client
================
Free ETF inflow/outflow estimation via yfinance shares-outstanding deltas.

ETFGI / Bloomberg are paid. The standard free proxy used by buy-side desks:

    daily_flow_usd ≈ Δ shares_outstanding × NAV (or close price)

When an ETF takes in cash, the AP creates new shares; when investors sell,
shares are redeemed. Day-over-day shares-outstanding deltas track creations
and redemptions, which is exactly net flow.

Data source: yfinance ``Ticker.info`` (free, no key). Shares outstanding is
end-of-day; we persist a rolling local snapshot in ``data/etf_flow_history.json``
so a daily delta can be computed even though yfinance only returns the latest
point-in-time value.

Universe (default): broad market + sectors + cross-asset ETFs.

Usage:
    from integrations.etf_flow_client import ETFFlowClient
    c = ETFFlowClient()
    snap = c.get_snapshot("SPY")           # one ETF, persists shares
    snaps = c.get_universe_snapshots()     # all default ETFs

The first call for a symbol always reports flow=None (need >=2 samples).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "data" / "etf_flow_history.json"

# Default ETF universe — broad market, sectors, fixed income, commodities.
DEFAULT_UNIVERSE: List[str] = [
    # Broad market
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    # Sector SPDRs
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC",
    # Commodities
    "GLD", "SLV", "USO",
    # Fixed income
    "TLT", "HYG", "LQD", "JNK", "EMB",
    # Volatility
    "VXX", "UVXY",
    # Crypto-adjacent
    "IBIT", "ETHE",
]


@dataclass
class ETFFlowSnapshot:
    """Single-day ETF flow + AUM snapshot."""

    symbol: str
    date: str                             # YYYY-MM-DD (UTC)
    nav_or_price: Optional[float] = None
    shares_outstanding: Optional[float] = None
    total_assets: Optional[float] = None
    prev_shares_outstanding: Optional[float] = None
    prev_date: Optional[str] = None
    daily_flow_usd: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ETFFlowClient:
    """yfinance-backed ETF AUM + estimated daily flow client."""

    def __init__(self, history_path: Optional[Path] = None) -> None:
        self.history_path = Path(history_path) if history_path else DEFAULT_HISTORY_PATH
        self._yf = None
        self._history: Dict[str, List[Dict[str, Any]]] = self._load_history()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load_history(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.history_path.exists():
            return {}
        try:
            with self.history_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("etf_flow: history load failed: %s", exc)
            return {}
        return data if isinstance(data, dict) else {}

    def _save_history(self) -> None:
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("w", encoding="utf-8") as fh:
                json.dump(self._history, fh, indent=2, sort_keys=True)
        except OSError as exc:
            logger.warning("etf_flow: history save failed: %s", exc)

    # ------------------------------------------------------------------
    # yfinance
    # ------------------------------------------------------------------
    def _yfinance(self):
        if self._yf is None:
            try:
                import yfinance as yf  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError("yfinance not installed") from exc
            self._yf = yf
        return self._yf

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _fetch_raw(self, symbol: str) -> Dict[str, Any]:
        yf = self._yfinance()
        ticker = yf.Ticker(symbol)
        info: Dict[str, Any] = {}
        try:
            # ``fast_info`` is preferred (single HTTP call); fall back to ``info``.
            fi = getattr(ticker, "fast_info", None)
            if fi is not None:
                # fast_info supports dict-like .get on newer yfinance, attribute on older
                for key in ("shares", "last_price", "previous_close", "market_cap"):
                    try:
                        info[key] = fi[key] if hasattr(fi, "__getitem__") else getattr(fi, key, None)
                    except (KeyError, AttributeError):
                        info[key] = None
        except (ValueError, KeyError, AttributeError, ConnectionError, TimeoutError) as exc:
            logger.debug("etf_flow: %s fast_info miss: %s", symbol, exc)

        # Backfill from .info if needed (slower, more fields)
        if not info.get("shares") or not info.get("last_price"):
            try:
                full = ticker.info or {}
            except (ValueError, KeyError, ConnectionError, TimeoutError) as exc:
                logger.warning("etf_flow: %s info fetch failed: %s", symbol, exc)
                full = {}
            info.setdefault("shares", full.get("sharesOutstanding"))
            info.setdefault("last_price", full.get("regularMarketPrice") or full.get("navPrice"))
            info.setdefault("previous_close", full.get("regularMarketPreviousClose"))
            info.setdefault("market_cap", full.get("marketCap") or full.get("totalAssets"))
        return info

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_snapshot(self, symbol: str, persist: bool = True) -> ETFFlowSnapshot:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        snap = ETFFlowSnapshot(symbol=symbol.upper(), date=today)

        try:
            raw = self._fetch_raw(symbol)
        except RuntimeError as exc:
            snap.error = str(exc)
            return snap

        snap.shares_outstanding = self._coerce_float(raw.get("shares"))
        snap.nav_or_price = self._coerce_float(raw.get("last_price"))
        snap.total_assets = self._coerce_float(raw.get("market_cap"))

        if snap.shares_outstanding is None:
            snap.notes.append("shares_outstanding unavailable from yfinance")

        # Compute delta vs most recent prior snapshot for this symbol.
        history = self._history.get(snap.symbol, [])
        prior = next(
            (h for h in reversed(history) if h.get("date") != today),
            None,
        )
        if prior:
            snap.prev_shares_outstanding = self._coerce_float(prior.get("shares_outstanding"))
            snap.prev_date = prior.get("date")
            if (
                snap.shares_outstanding is not None
                and snap.prev_shares_outstanding is not None
                and snap.nav_or_price is not None
            ):
                delta_shares = snap.shares_outstanding - snap.prev_shares_outstanding
                snap.daily_flow_usd = delta_shares * snap.nav_or_price
        else:
            snap.notes.append("no prior snapshot; flow requires >=2 samples")

        if persist and snap.shares_outstanding is not None:
            history.append(
                {
                    "date": today,
                    "shares_outstanding": snap.shares_outstanding,
                    "nav_or_price": snap.nav_or_price,
                    "total_assets": snap.total_assets,
                }
            )
            # Cap history at 400 entries per symbol (~18 months)
            self._history[snap.symbol] = history[-400:]
            self._save_history()

        return snap

    def get_universe_snapshots(
        self, symbols: Optional[List[str]] = None, persist: bool = True
    ) -> List[ETFFlowSnapshot]:
        universe = symbols or DEFAULT_UNIVERSE
        return [self.get_snapshot(s, persist=persist) for s in universe]

    def aggregate_flows(
        self, snapshots: List[ETFFlowSnapshot]
    ) -> Dict[str, float]:
        """Sum daily flows by direction; useful for risk-on/risk-off read."""
        inflow = sum(
            s.daily_flow_usd for s in snapshots
            if s.daily_flow_usd is not None and s.daily_flow_usd > 0
        )
        outflow = sum(
            s.daily_flow_usd for s in snapshots
            if s.daily_flow_usd is not None and s.daily_flow_usd < 0
        )
        return {
            "gross_inflow_usd": inflow,
            "gross_outflow_usd": outflow,
            "net_flow_usd": inflow + outflow,
            "samples": sum(1 for s in snapshots if s.daily_flow_usd is not None),
        }


__all__ = [
    "ETFFlowClient",
    "ETFFlowSnapshot",
    "DEFAULT_UNIVERSE",
    "DEFAULT_HISTORY_PATH",
]
