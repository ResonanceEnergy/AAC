"""
Polymarket Prediction Market Agent — AAC v2.7.0
=================================================

Full integration with the Polymarket prediction market platform.

Three API layers:
  - Gamma API  (gamma-api.polymarket.com) — events, markets, tags, search
  - Data API   (data-api.polymarket.com)  — trades, timeseries, leaderboard
  - CLOB API   (clob.polymarket.com)      — orderbook, midpoint, spread, prices

Trading (order placement / cancellation) requires the official
`py-clob-client` SDK + a funded Polygon wallet.  This agent handles:
  1. Market discovery & monitoring
  2. Arbitrage opportunity scanning (cross-platform prediction spreads)
  3. Sentiment / probability extraction for AAC strategies
  4. Paper & live order placement (when py-clob-client + keys configured)

Docs: https://docs.polymarket.com/
SDK:  https://github.com/Polymarket/py-clob-client
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp

from shared.config_loader import get_config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PolymarketEvent:
    """A Polymarket event (container of markets)."""
    event_id: str
    slug: str
    title: str
    description: str = ""
    active: bool = True
    closed: bool = False
    markets: List[Dict[str, Any]] = field(default_factory=list)
    volume: float = 0.0
    volume_24hr: float = 0.0
    liquidity: float = 0.0
    tags: List[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "PolymarketEvent":
        """From api."""
        return cls(
            event_id=str(data.get("id", "")),
            slug=data.get("slug", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            active=data.get("active", True),
            closed=data.get("closed", False),
            markets=data.get("markets", []),
            volume=float(data.get("volume", 0) or 0),
            volume_24hr=float(data.get("volume_24hr", 0) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
            tags=[t.get("label", "") for t in data.get("tags", [])],
            start_date=data.get("startDate", ""),
            end_date=data.get("endDate", ""),
        )


@dataclass
class PolymarketMarket:
    """A single binary outcome market."""
    condition_id: str
    question: str
    slug: str = ""
    yes_token_id: str = ""
    no_token_id: str = ""
    yes_price: float = 0.5
    no_price: float = 0.5
    volume: float = 0.0
    liquidity: float = 0.0
    active: bool = True

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "PolymarketMarket":
        """From api."""
        tokens = data.get("clobTokenIds", "") or data.get("tokens", "")
        yes_id, no_id = "", ""
        if isinstance(tokens, list) and len(tokens) >= 2:
            yes_id = str(tokens[0]) if isinstance(tokens[0], str) else str(tokens[0].get("token_id", ""))
            no_id = str(tokens[1]) if isinstance(tokens[1], str) else str(tokens[1].get("token_id", ""))
        elif isinstance(tokens, str) and tokens:
            parts = tokens.split(",")
            if len(parts) >= 2:
                yes_id, no_id = parts[0].strip(), parts[1].strip()

        yes_price = float(data.get("outcomePrices", "0.5,0.5").split(",")[0]) if isinstance(data.get("outcomePrices"), str) else 0.5
        no_price = 1.0 - yes_price

        return cls(
            condition_id=data.get("conditionId", "") or data.get("condition_id", ""),
            question=data.get("question", ""),
            slug=data.get("slug", ""),
            yes_token_id=yes_id,
            no_token_id=no_id,
            yes_price=yes_price,
            no_price=no_price,
            volume=float(data.get("volume", 0) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
            active=data.get("active", True),
        )


@dataclass
class ArbitrageOpportunity:
    """A detected prediction market arbitrage opportunity."""
    market_question: str
    condition_id: str
    yes_price: float
    no_price: float
    spread: float
    overround: float  # YES+NO prices — anything < 1.0 is pure arb
    estimated_edge_pct: float
    detected_at: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════
# POLYMARKET AGENT
# ═══════════════════════════════════════════════════════════════════════════

class PolymarketAgent:
    """
    Autonomous agent for monitoring and trading on Polymarket.

    Capabilities:
      - Discover trending/highest-volume events and markets
      - Monitor real-time prices and order books via CLOB API
      - Detect overround arbitrage (YES+NO < 1.0)
      - Extract crowd probabilities for use by other AAC strategies
      - Place orders via py-clob-client SDK (when configured)
      - Paper-trade prediction markets via local SQLite ledger
    """

    def __init__(self):
        self.config = get_config()
        self._session: Optional[aiohttp.ClientSession] = None
        self._clob_client = None  # Lazy-init py-clob-client
        self._scan_interval = 300  # seconds between scans
        self._events_cache: List[PolymarketEvent] = []
        self._markets_cache: List[PolymarketMarket] = []
        self._opportunities: List[ArbitrageOpportunity] = []
        self._last_scan: Optional[datetime] = None

    # ─── HTTP session management ─────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def close(self):
        """Close."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, url: str, params: Optional[Dict] = None) -> Any:
        """Execute a GET request and return JSON."""
        session = await self._get_session()
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ─── Gamma API — events & markets ────────────────────────────────────

    async def get_trending_events(self, limit: int = 50) -> List[PolymarketEvent]:
        """Fetch top events sorted by 24-hour volume."""
        data = await self._get(f"{GAMMA_API}/events", params={
            "active": "true",
            "closed": "false",
            "order": "volume_24hr",
            "ascending": "false",
            "limit": limit,
        })
        events = [PolymarketEvent.from_api(e) for e in (data if isinstance(data, list) else [])]
        self._events_cache = events
        return events

    async def get_event_by_slug(self, slug: str) -> Optional[PolymarketEvent]:
        """Fetch a single event by its Polymarket URL slug."""
        data = await self._get(f"{GAMMA_API}/events/slug/{slug}")
        if isinstance(data, dict) and data.get("id"):
            return PolymarketEvent.from_api(data)
        if isinstance(data, list) and data:
            return PolymarketEvent.from_api(data[0])
        return None

    async def get_active_markets(self, limit: int = 100,
                                  offset: int = 0) -> List[PolymarketMarket]:
        """Fetch active markets from Gamma API."""
        data = await self._get(f"{GAMMA_API}/markets", params={
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
        })
        markets = [PolymarketMarket.from_api(m) for m in (data if isinstance(data, list) else [])]
        self._markets_cache = markets
        return markets

    async def search_markets(self, keyword: str, limit: int = 20) -> List[PolymarketMarket]:
        """Text search for markets containing keyword."""
        data = await self._get(f"{GAMMA_API}/markets", params={
            "slug": keyword,
            "limit": limit,
        })
        return [PolymarketMarket.from_api(m) for m in (data if isinstance(data, list) else [])]

    async def get_tags(self) -> List[Dict[str, Any]]:
        """Get all available tags (categories)."""
        data = await self._get(f"{GAMMA_API}/tags")
        return data if isinstance(data, list) else []

    # ─── CLOB API — prices & orderbook ───────────────────────────────────

    async def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token."""
        data = await self._get(f"{CLOB_API}/midpoint", params={"token_id": token_id})
        mid = data.get("mid") if isinstance(data, dict) else None
        return float(mid) if mid is not None else None

    async def get_spread(self, token_id: str) -> Dict[str, Optional[float]]:
        """Get bid-ask spread for a token."""
        data = await self._get(f"{CLOB_API}/spread", params={"token_id": token_id})
        if isinstance(data, dict):
            return {
                "bid": float(data["bid"]) if data.get("bid") else None,
                "ask": float(data["ask"]) if data.get("ask") else None,
                "spread": float(data["spread"]) if data.get("spread") else None,
            }
        return {"bid": None, "ask": None, "spread": None}

    async def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """Get full order book for a token."""
        return await self._get(f"{CLOB_API}/book", params={"token_id": token_id})

    async def get_last_trade_price(self, token_id: str) -> Optional[float]:
        """Get last trade price."""
        data = await self._get(f"{CLOB_API}/last-trade-price", params={"token_id": token_id})
        price = data.get("price") if isinstance(data, dict) else None
        return float(price) if price is not None else None

    async def get_price_history(self, token_id: str, interval: str = "1d",
                                 fidelity: int = 60) -> List[Dict[str, Any]]:
        """Get historical price timeseries."""
        data = await self._get(f"{CLOB_API}/prices-history", params={
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity,
        })
        return data.get("history", []) if isinstance(data, dict) else []

    # ─── Data API — trades, activity, leaderboard ────────────────────────

    async def get_recent_trades(self, condition_id: str,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for a market."""
        data = await self._get(f"{DATA_API}/trades", params={
            "market": condition_id,
            "limit": limit,
        })
        return data if isinstance(data, list) else []

    async def get_leaderboard(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Get platform leaderboard."""
        data = await self._get(f"{DATA_API}/leaderboard", params={"limit": limit})
        return data if isinstance(data, list) else []

    # ─── Arbitrage scanning ──────────────────────────────────────────────

    async def scan_for_arbitrage(self, min_edge_pct: float = 0.5) -> List[ArbitrageOpportunity]:
        """
        Scan all active markets for overround arbitrage.

        If YES + NO prices sum to < 1.0, there's a risk-free profit opportunity
        by buying both sides.  If > 1.0, the house edge exists.
        """
        logger.info("Scanning Polymarket for arbitrage opportunities...")
        markets = await self.get_active_markets(limit=200)
        opportunities: List[ArbitrageOpportunity] = []

        for mkt in markets:
            overround = mkt.yes_price + mkt.no_price
            edge_pct = (1.0 - overround) * 100 if overround < 1.0 else 0.0

            if edge_pct >= min_edge_pct:
                opp = ArbitrageOpportunity(
                    market_question=mkt.question,
                    condition_id=mkt.condition_id,
                    yes_price=mkt.yes_price,
                    no_price=mkt.no_price,
                    spread=abs(mkt.yes_price - mkt.no_price),
                    overround=overround,
                    estimated_edge_pct=edge_pct,
                )
                opportunities.append(opp)
                logger.info(
                    f"ARB: {mkt.question[:60]}... YES={mkt.yes_price:.3f} "
                    f"NO={mkt.no_price:.3f} edge={edge_pct:.2f}%"
                )

        self._opportunities = opportunities
        self._last_scan = datetime.now()
        logger.info(f"Scan complete: {len(opportunities)} opportunities found")
        return opportunities

    # ─── Probability extraction (for other AAC strategies) ───────────────

    async def get_crowd_probabilities(self, slugs: List[str]) -> Dict[str, float]:
        """
        Extract crowd-implied probabilities for a set of events.

        Returns {slug: probability_yes} dictionary.
        Useful for macro/sentiment overlays in AAC trading strategies.
        """
        probabilities: Dict[str, float] = {}
        for slug in slugs:
            try:
                event = await self.get_event_by_slug(slug)
                if event and event.markets:
                    # Take the first market's YES price as implied probability
                    mkt_data = event.markets[0]
                    prices = mkt_data.get("outcomePrices", "0.5,0.5")
                    if isinstance(prices, str):
                        yes_price = float(prices.split(",")[0])
                    else:
                        yes_price = 0.5
                    probabilities[slug] = yes_price
            except Exception as e:
                logger.warning(f"Failed to get probability for {slug}: {e}")
        return probabilities

    # ─── Trading (requires py-clob-client + private key) ─────────────────

    def _init_clob_trading_client(self):
        """Lazy-initialize the official py-clob-client SDK for trading."""
        if self._clob_client is not None:
            return

        private_key = self.config.polymarket_private_key
        if not private_key:
            logger.warning("POLYMARKET_PRIVATE_KEY not set — trading disabled")
            return

        try:
            from py_clob_client.client import ClobClient
            self._clob_client = ClobClient(
                CLOB_API,
                key=private_key,
                chain_id=self.config.polymarket_chain_id,
                signature_type=0,  # EOA default
                funder=self.config.polymarket_funder or None,
            )
            self._clob_client.set_api_creds(
                self._clob_client.create_or_derive_api_creds()
            )
            logger.info("Polymarket CLOB trading client initialized")
        except ImportError:
            logger.warning(
                "py-clob-client not installed. "
                "Install with: pip install py-clob-client"
            )
        except Exception as e:
            logger.error(f"Failed to initialize CLOB trading client: {e}")

    def place_limit_order(self, token_id: str, price: float, size: float,
                          side: str = "BUY") -> Optional[Dict[str, Any]]:
        """Place a limit order on Polymarket (requires SDK + funded wallet)."""
        self._init_clob_trading_client()
        if self._clob_client is None:
            logger.error("Trading client not available")
            return None

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            order_side = BUY if side.upper() == "BUY" else SELL
            order = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=order_side,
            )
            signed = self._clob_client.create_order(order)
            result = self._clob_client.post_order(signed, OrderType.GTC)
            logger.info(f"Order placed: {side} {size}@{price} on {token_id[:16]}...")
            return result
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return None

    def place_market_order(self, token_id: str, amount: float,
                           side: str = "BUY") -> Optional[Dict[str, Any]]:
        """Place a market order (fill-or-kill) on Polymarket."""
        self._init_clob_trading_client()
        if self._clob_client is None:
            logger.error("Trading client not available")
            return None

        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            order_side = BUY if side.upper() == "BUY" else SELL
            mo = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=order_side,
                order_type=OrderType.FOK,
            )
            signed = self._clob_client.create_market_order(mo)
            result = self._clob_client.post_order(signed, OrderType.FOK)
            logger.info(f"Market order executed: {side} ${amount} on {token_id[:16]}...")
            return result
        except Exception as e:
            logger.error(f"Market order failed: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        self._init_clob_trading_client()
        if self._clob_client is None:
            return False
        try:
            self._clob_client.cancel(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        self._init_clob_trading_client()
        if self._clob_client is None:
            return False
        try:
            self._clob_client.cancel_all()
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")
            return False

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders."""
        self._init_clob_trading_client()
        if self._clob_client is None:
            return []
        try:
            from py_clob_client.clob_types import OpenOrderParams
            return self._clob_client.get_orders(OpenOrderParams())
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    # ─── Continuous monitoring loop ──────────────────────────────────────

    async def run_monitor(self, interval: int = 300,
                          min_edge_pct: float = 0.5):
        """
        Continuous monitoring loop.  Scans for arb opportunities every
        `interval` seconds and logs findings.
        """
        logger.info(f"Starting Polymarket monitor (interval={interval}s, min_edge={min_edge_pct}%)")
        try:
            while True:
                try:
                    opportunities = await self.scan_for_arbitrage(min_edge_pct)
                    if opportunities:
                        for opp in opportunities[:5]:
                            logger.info(
                                f"  >> {opp.market_question[:50]}... "
                                f"edge={opp.estimated_edge_pct:.2f}%"
                            )
                except Exception as e:
                    logger.error(f"Monitor scan error: {e}")
                await asyncio.sleep(interval)
        finally:
            await self.close()

    # ─── Status / diagnostics ────────────────────────────────────────────

    async def get_status(self) -> Dict[str, Any]:
        """Get agent status summary."""
        trading_available = bool(self.config.polymarket_private_key)
        try:
            sdk_installed = True
            import py_clob_client  # noqa: F401
        except ImportError:
            sdk_installed = False

        return {
            "agent": "PolymarketAgent",
            "apis": {
                "gamma": GAMMA_API,
                "data": DATA_API,
                "clob": CLOB_API,
            },
            "trading_enabled": trading_available and sdk_installed,
            "sdk_installed": sdk_installed,
            "private_key_configured": trading_available,
            "cached_events": len(self._events_cache),
            "cached_markets": len(self._markets_cache),
            "pending_opportunities": len(self._opportunities),
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLI / self-test
# ═══════════════════════════════════════════════════════════════════════════

async def _self_test():
    """Quick self-test: fetch trending events + scan for arb."""
    agent = PolymarketAgent()
    try:
        logger.info("=" * 60)
        logger.info("  POLYMARKET AGENT — SELF TEST")
        logger.info("=" * 60)

        # 1 — Status
        status = await agent.get_status()
        logger.info(f"\nTrading enabled: {status['trading_enabled']}")
        logger.info(f"SDK installed:   {status['sdk_installed']}")

        # 2 — Trending events
        logger.info("\n--- Top 10 Trending Events ---")
        events = await agent.get_trending_events(limit=10)
        for i, ev in enumerate(events, 1):
            logger.info(f"  {i}. {ev.title[:60]:<60}  vol_24h=${ev.volume_24hr:,.0f}")

        # 3 — Arbitrage scan
        logger.info("\n--- Arbitrage Scan ---")
        opps = await agent.scan_for_arbitrage(min_edge_pct=0.1)
        if opps:
            for opp in opps[:5]:
                print(
                    f"  >> {opp.market_question[:50]:<50}  "
                    f"YES={opp.yes_price:.3f}  NO={opp.no_price:.3f}  "
                    f"edge={opp.estimated_edge_pct:.2f}%"
                )
        else:
            logger.info("  No arbitrage opportunities found at 0.1% threshold")

        # 4 — Tags
        tags = await agent.get_tags()
        logger.info(f"\n--- Available Tags ({len(tags)}) ---")
        for tag in tags[:10]:
            label = tag.get("label", tag) if isinstance(tag, dict) else str(tag)
            logger.info(f"  - {label}")

        logger.info("\n[OK] Polymarket agent self-test complete")
    except Exception as e:
        logger.info(f"\n[FAIL] Self-test error: {e}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(_self_test())
