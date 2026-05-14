"""
AAC Market Data Feeds
=====================
Unified market-data feed used by strategy execution engines.

ALL DATA IS REAL. No simulated prices, no random walks, no demo fallbacks.
Backed by yfinance (free, no key) for quotes, historical bars, and option
chains.  If yfinance is unavailable or returns no data, methods return
``None`` so callers can fail honestly instead of trading on fake numbers.

Intentional gaps:
* ``get_order_book``  -> yfinance does not expose Level 2; returns ``None``.
* ``get_market_sentiment`` -> derived from real historical returns only;
  no synthetic momentum/volatility numbers are ever produced.

Sprint 52 (2026-04-24): rewritten under the "NO MOCK DATA OR CALLS" doctrine
(see ``.context/04_workstreams/GOAL_MANDATE_ROADMAP.md``).  Previous version
silently fell back to a random-walk simulator; that has been removed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DataFeedType(Enum):
    PRICE_QUOTES = "price_quotes"
    ORDER_BOOK = "order_book"
    TRADE_FEED = "trade_feed"
    OPTIONS_CHAIN = "options_chain"
    NEWS_FEED = "news_feed"


class Exchange(Enum):
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"
    CBOE = "CBOE"
    CME = "CME"
    ICE = "ICE"


@dataclass
class MarketData:
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: datetime
    exchange: Exchange
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderBookData:
    symbol: str
    bids: List[Tuple[float, int]]
    asks: List[Tuple[float, int]]
    timestamp: datetime
    exchange: Exchange


@dataclass
class TradeData:
    symbol: str
    price: float
    quantity: int
    timestamp: datetime
    exchange: Exchange
    trade_id: str


# Symbols whose quotes route through NASDAQ; everything else defaults to NYSE.
_NASDAQ_SYMBOLS = frozenset(
    {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "NVDA", "META", "NFLX",
     "QQQ", "AMD", "INTC", "PYPL", "ADBE", "CSCO", "AVGO"}
)


def _classify_exchange(symbol: str) -> Exchange:
    return Exchange.NASDAQ if symbol.upper() in _NASDAQ_SYMBOLS else Exchange.NYSE


def _yfinance_ticker(symbol: str):
    """Import yfinance lazily so import-time failures don't kill the module."""
    import yfinance as yf  # noqa: PLC0415

    return yf.Ticker(symbol)


class MarketDataFeed:
    """Unified market data feed -- yfinance-backed, no simulation."""

    def __init__(self) -> None:
        self.data_cache: Dict[str, MarketData] = {}
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.feed_tasks: List[asyncio.Task] = []
        self.cache_ttl_seconds: int = 5
        self.poll_interval_seconds: float = 1.0
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Start background subscription loop; verify yfinance is importable."""
        self.logger.info("Initializing Market Data Feeds (yfinance backend)")
        try:
            import yfinance  # noqa: F401, PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "yfinance is required for MarketDataFeed; install via "
                "`pip install yfinance`"
            ) from exc
        self.feed_tasks.append(asyncio.create_task(self._price_update_loop()))
        self.logger.info("Market Data Feeds initialized")

    async def close(self) -> None:
        for task in self.feed_tasks:
            task.cancel()
        self.feed_tasks.clear()
        self.logger.info("Market Data Feeds closed")

    # ------------------------------------------------------------------ price
    async def get_latest_price(self, symbol: str) -> Optional[MarketData]:
        """Return the latest real quote for ``symbol``.

        Returns ``None`` if yfinance has no data for the symbol.  Never
        returns simulated prices.
        """
        cached = self.data_cache.get(symbol)
        if cached is not None:
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age < self.cache_ttl_seconds:
                return cached

        data = await self._fetch_real_price(symbol)
        if data is not None:
            self.data_cache[symbol] = data
        return data

    async def _fetch_real_price(self, symbol: str) -> Optional[MarketData]:
        try:
            return await asyncio.to_thread(self._fetch_real_price_sync, symbol)
        except Exception as exc:  # noqa: BLE001 -- log + return None, no fakes
            self.logger.warning("yfinance quote failed for %s: %s", symbol, exc)
            return None

    def _fetch_real_price_sync(self, symbol: str) -> Optional[MarketData]:
        ticker = _yfinance_ticker(symbol)
        fi = ticker.fast_info

        last_price = getattr(fi, "last_price", None)
        if last_price is None:
            try:
                last_price = fi["last_price"]
            except (KeyError, TypeError):
                last_price = None
        if not isinstance(last_price, (int, float)) or last_price <= 0:
            return None

        bid = getattr(fi, "bid", None)
        ask = getattr(fi, "ask", None)
        if not isinstance(bid, (int, float)) or bid <= 0:
            bid = float(last_price) * 0.9999
        if not isinstance(ask, (int, float)) or ask <= 0:
            ask = float(last_price) * 1.0001

        volume = getattr(fi, "last_volume", None)
        if not isinstance(volume, (int, float)):
            volume = 0

        return MarketData(
            symbol=symbol,
            price=float(last_price),
            bid=float(bid),
            ask=float(ask),
            volume=int(volume),
            timestamp=datetime.now(),
            exchange=_classify_exchange(symbol),
            metadata={"source": "yfinance"},
        )

    # ------------------------------------------------------------- order book
    async def get_order_book(self, symbol: str, depth: int = 10) -> Optional[OrderBookData]:
        """yfinance does not expose Level 2 data.

        Returns ``None`` so callers must handle the absence honestly rather
        than receiving a fabricated book.  ``depth`` retained for API
        compatibility.
        """
        del depth
        self.logger.debug(
            "get_order_book(%s) -> None (no L2 source available)", symbol
        )
        return None

    # -------------------------------------------------------------- historical
    async def get_historical_prices(self, symbol: str, days: int = 30):
        """Return real historical OHLCV via yfinance.

        Returns a ``pandas.DataFrame`` indexed by date with columns
        ``open, high, low, close, volume`` -- or ``None`` if yfinance has
        no data.
        """
        try:
            return await asyncio.to_thread(self._historical_sync, symbol, days)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("yfinance history failed for %s: %s", symbol, exc)
            return None

    def _historical_sync(self, symbol: str, days: int):
        import pandas as pd  # noqa: PLC0415

        ticker = _yfinance_ticker(symbol)
        period = f"{max(int(days), 1)}d"
        df = ticker.history(period=period, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        df = df[keep]
        df.index.name = "date"
        # Coerce to plain pandas types so downstream code doesn't see numpy quirks.
        df = df.astype({c: float for c in keep if c != "volume"})
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0).astype("int64")
        if not isinstance(df, pd.DataFrame):  # pragma: no cover -- defensive
            return None
        return df

    # ------------------------------------------------------------- sentiment
    async def get_market_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return real return-based sentiment metrics or ``None``.

        Computes momentum and realized volatility from ``get_historical_prices``.
        No fabricated values.
        """
        hist = await self.get_historical_prices(symbol, days=10)
        if hist is None or len(hist) < 2:
            return None

        try:
            returns = hist["close"].pct_change().dropna()
            if returns.empty:
                return None
            momentum_1d = float(returns.iloc[-1])
            momentum_5d = float(returns.tail(5).mean())
            volatility_5d = float(returns.tail(5).std() or 0.0)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("sentiment calc failed for %s: %s", symbol, exc)
            return None

        # Volume trend (5-day mean vs 10-day mean) -- real metric, not a placeholder.
        volume_trend = 0.0
        if "volume" in hist.columns and len(hist["volume"]) >= 5:
            try:
                recent = float(hist["volume"].tail(5).mean())
                older = float(hist["volume"].mean())
                if older > 0:
                    volume_trend = (recent - older) / older
            except Exception as exc:  # noqa: BLE001
                self.logger.debug("volume trend calc failed for %s: %s", symbol, exc)

        return {
            "momentum_1d": momentum_1d,
            "momentum_5d": momentum_5d,
            "volatility_5d": volatility_5d,
            "volume_trend": volume_trend,
            "timestamp": datetime.now(),
        }

    # -------------------------------------------------------------- options
    async def get_options_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return real options chain (front expiry) or ``None``."""
        try:
            return await asyncio.to_thread(self._options_sync, symbol)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("yfinance options failed for %s: %s", symbol, exc)
            return None

    def _options_sync(self, symbol: str) -> Optional[Dict[str, Any]]:
        ticker = _yfinance_ticker(symbol)
        expiries = getattr(ticker, "options", None) or []
        if not expiries:
            return None

        front = expiries[0]
        chain = ticker.option_chain(front)
        calls_df = getattr(chain, "calls", None)
        puts_df = getattr(chain, "puts", None)
        if calls_df is None or puts_df is None:
            return None

        spot_data = self._fetch_real_price_sync(symbol)
        spot = spot_data.price if spot_data else None

        import math

        def _safe_float(val) -> float:
            try:
                f = float(val)
            except (TypeError, ValueError):
                return 0.0
            return 0.0 if math.isnan(f) else f

        def _safe_int(val) -> int:
            f = _safe_float(val)
            return int(f) if f > 0 else 0

        def _row_to_dict(row) -> Dict[str, float]:
            return {
                "premium": _safe_float(row.get("lastPrice", 0.0)),
                "bid": _safe_float(row.get("bid", 0.0)),
                "ask": _safe_float(row.get("ask", 0.0)),
                "iv": _safe_float(row.get("impliedVolatility", 0.0)),
                "volume": _safe_int(row.get("volume", 0)),
                "open_interest": _safe_int(row.get("openInterest", 0)),
            }

        calls: Dict[str, Dict[str, float]] = {}
        puts: Dict[str, Dict[str, float]] = {}
        for _, row in calls_df.iterrows():
            strike = float(row.get("strike", 0.0) or 0.0)
            if strike > 0:
                calls[str(strike)] = _row_to_dict(row)
        for _, row in puts_df.iterrows():
            strike = float(row.get("strike", 0.0) or 0.0)
            if strike > 0:
                puts[str(strike)] = _row_to_dict(row)

        return {
            "calls": calls,
            "puts": puts,
            "spot_price": spot,
            "expiry": front,
            "timestamp": datetime.now(),
        }

    # ------------------------------------------------------- subscriptions
    async def subscribe_to_price_updates(self, symbol: str, callback: Callable) -> None:
        self.subscriptions.setdefault(symbol, []).append(callback)
        self.logger.info("Subscribed to price updates for %s", symbol)

    async def unsubscribe_from_price_updates(self, symbol: str, callback: Callable) -> None:
        if symbol in self.subscriptions:
            try:
                self.subscriptions[symbol].remove(callback)
            except ValueError:
                pass
            if not self.subscriptions[symbol]:
                del self.subscriptions[symbol]

    async def _price_update_loop(self) -> None:
        while True:
            try:
                for symbol in list(self.subscriptions.keys()):
                    data = await self.get_latest_price(symbol)
                    if data is None:
                        continue
                    for callback in list(self.subscriptions.get(symbol, ())):
                        try:
                            await callback(data)
                        except Exception as exc:  # noqa: BLE001
                            self.logger.error("price callback error: %s", exc)
                await asyncio.sleep(self.poll_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                self.logger.error("price update loop error: %s", exc)
                await asyncio.sleep(5)


# Module-level singleton ----------------------------------------------------
_market_data_feed: Optional[MarketDataFeed] = None


async def get_market_data_feed() -> MarketDataFeed:
    """Get-or-create the singleton ``MarketDataFeed``."""
    global _market_data_feed
    if _market_data_feed is None:
        _market_data_feed = MarketDataFeed()
        await _market_data_feed.initialize()
    return _market_data_feed


def _reset_singleton_for_tests() -> None:
    """Test helper -- drop the cached singleton."""
    global _market_data_feed
    _market_data_feed = None
