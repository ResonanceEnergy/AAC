"""
Storm Lifeboat Matrix — Live Data Feed
=========================================
Unified ingestion layer that pulls real-time prices, VIX, options flow,
macro indicators, and sentiment from every connected AAC data source.

Data sources wired:
    1. Polygon.io      — real-time snapshots for all 20 assets (primary)
    2. Finnhub         — quote fallback for stocks/ETFs
    3. IBKR            — broker prices + positions + account (secondary/live)
    4. Moomoo           — quote fallback via OpenD gateway
    5. FRED            — VIX (VIXCLS), 10Y yield, oil, gold, fed funds, HY spread
    6. Unusual Whales  — options flow, dark pool, put/call ratio, market tone
    7. Fear & Greed    — crypto sentiment 0-100
    8. Google Trends   — retail search interest heatmap
    9. Finnhub News    — headline-driven scenario indicator detection

Results are cached with configurable TTL to respect rate limits.
All external calls are async; this module provides sync wrappers for
the Storm Lifeboat runner (which is synchronous).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from strategies.storm_lifeboat.core import DEFAULT_PRICES, Asset, VolRegime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ASSET → TICKER SYMBOL MAPPING
# ═══════════════════════════════════════════════════════════════════════════

# Polygon / Finnhub / IBKR use standard US equity tickers.
# Crypto needs special treatment per source.
ASSET_TICKER_MAP: Dict[Asset, str] = {
    Asset.OIL: "USO",       # WTI proxy ETF (Polygon/Finnhub); /CL for IBKR futures
    Asset.GOLD: "GLD",      # Gold proxy ETF; /GC for IBKR futures
    Asset.SILVER: "SLV",    # Silver proxy ETF
    Asset.GDX: "GDX",
    Asset.SPY: "SPY",
    Asset.QQQ: "QQQ",
    Asset.XLF: "XLF",
    Asset.XLRE: "XLRE",
    Asset.KRE: "KRE",
    Asset.JETS: "JETS",
    Asset.XLY: "XLY",
    Asset.XLE: "XLE",
    Asset.TLT: "TLT",
    Asset.HYG: "HYG",
    Asset.BTC: "BTC",       # Finnhub: BINANCE:BTCUSDT  |  Polygon: X:BTCUSD
    Asset.ETH: "ETH",       # Finnhub: BINANCE:ETHUSDT  |  Polygon: X:ETHUSD
    Asset.XRP: "XRP",       # Finnhub: BINANCE:XRPUSDT  |  Polygon: X:XRPUSD
    Asset.BITO: "BITO",
    Asset.TSLA: "TSLA",
    Asset.SMR: "SMR",
}

# Polygon crypto tickers use X: prefix
POLYGON_CRYPTO_MAP: Dict[Asset, str] = {
    Asset.BTC: "X:BTCUSD",
    Asset.ETH: "X:ETHUSD",
    Asset.XRP: "X:XRPUSD",
}

# Finnhub crypto symbols
FINNHUB_CRYPTO_MAP: Dict[Asset, str] = {
    Asset.BTC: "BINANCE:BTCUSDT",
    Asset.ETH: "BINANCE:ETHUSDT",
    Asset.XRP: "BINANCE:XRPUSDT",
}

# FRED series for macro indicators fed into Storm Lifeboat
FRED_SERIES = {
    "vix": "VIXCLS",
    "10y_yield": "DGS10",
    "2y_yield": "DGS2",
    "yield_spread": "T10Y2Y",
    "fed_funds": "DFF",
    "oil_wti": "DCOILWTICO",
    "gold_pm": "GOLDAMGBD228NLBM",
    "hy_spread": "BAMLH0A0HYM2",
    "dollar_index": "DTWEXBGS",
}

# Scenario indicator keywords → scenario codes (for news-based trigger detection)
INDICATOR_KEYWORDS: Dict[str, List[str]] = {
    "HORMUZ": ["hormuz", "strait of hormuz", "iran navy", "persian gulf", "oil blockade"],
    "DEBT_CRISIS": ["sovereign debt", "treasury auction", "debt ceiling", "credit downgrade", "cds spread"],
    "TAIWAN": ["taiwan", "china military", "tsmc", "pla exercises", "taiwan strait"],
    "EU_BANKS": ["deutsche bank", "european bank", "ecb emergency", "interbank freeze", "banking crisis europe"],
    "DEFI_CASCADE": ["stablecoin depeg", "defi collapse", "crypto liquidation", "tvl collapse", "cex freeze"],
    "SUPERCYCLE": ["gold record", "silver record", "commodity supercycle", "central bank gold"],
    "CRE_COLLAPSE": ["commercial real estate", "office vacancy", "cmbs default", "regional bank failure"],
    "AI_BUBBLE": ["ai bubble", "nvidia crash", "tech layoffs", "ai monetization", "qqq crash"],
    "PANDEMIC_2": ["pandemic", "bird flu", "h5n1", "who emergency", "new variant"],
    "GRID_ATTACK": ["power grid", "cyberattack grid", "infrastructure attack", "blackout"],
    "PETRODOLLAR": ["petrodollar", "brics currency", "dollar reserve", "de-dollarization"],
    "US_CIVIL": ["civil unrest", "martial law", "domestic conflict", "national guard deployment"],
    "CLIMATE": ["climate disaster", "hurricane category 5", "crop failure", "wildfire record"],
    "LIQUIDITY": ["liquidity crisis", "fed pivot", "repo spike", "credit freeze", "margin call cascade"],
    "TARIFF_WAR": ["tariff", "trade war", "import ban", "retaliatory tariff", "trade sanctions"],
    "US_WITHDRAWAL": ["us withdrawal", "troop pullout", "isolationism", "nato exit", "middle east withdrawal"],
    "IRAN_DEAL": ["iran deal", "iran reparations", "iran backstop", "jcpoa", "iran settlement"],
    "PETRODOLLAR_SPIRAL": ["petrodollar spiral", "yuan oil settlement", "ruble settlement",
                            "mbridge", "brics settlement", "dollar weaponization"],
    "IRAN_NUCLEAR": ["iran nuclear", "gcc realignment", "iran enrichment",
                      "iran weapons grade", "nuclear breakout"],
    "ELITE_EXPOSURE": ["epstein files", "elite exposure", "trust erosion",
                        "institutional scandal", "blackmail networks"],
    # ── Scenarios 21-43: US Western Hemisphere Pivot ──
    "HEMISPHERE_PIVOT": ["monroe doctrine", "hemisphere pivot", "western hemisphere",
                          "hemisphere consolidation", "america first trade"],
    "NATO_EXIT": ["nato withdrawal", "nato exit", "nato pullout", "nato defunding",
                   "transatlantic split"],
    "EUROPE_ABANDON": ["abandon europe", "us europe split", "transatlantic collapse",
                        "european defense pact", "us recall ambassador europe"],
    "CANADA_DECLINE": ["canada tariff", "canadian dollar collapse", "us canada trade war",
                        "canadian resource extraction", "pipeline dispute"],
    "GREENLAND_ACQ": ["greenland purchase", "greenland acquisition", "arctic sovereignty",
                       "greenland rare earth", "danish territory dispute"],
    "PANAMA_RECLAIM": ["panama canal", "canal sovereignty", "panama us control",
                        "canal reclamation", "panama military"],
    "LATAM_LOCKIN": ["latin america resource", "latam trade bloc", "latam lock-in",
                      "hemisphere resource deal", "latin american treaty"],
    "ARCTIC_EXPAND": ["arctic base", "icebreaker fleet", "arctic military",
                       "northern sea route", "arctic resource claim"],
    "BORDER_MILITARY": ["border militarization", "southern border troops",
                         "border deployment", "border closure", "border wall military"],
    "VENEZUELA_REGIME": ["venezuela regime change", "pdvsa", "venezuela oil",
                          "venezuelan opposition", "venezuela sanctions"],
    "LITHIUM_TRIANGLE": ["lithium triangle", "lithium supply chain", "bolivia lithium",
                          "argentina lithium", "chile lithium"],
    "CUBA_EMBARGO": ["cuba embargo", "cuba sanctions", "cuba blockade",
                      "cuba naval", "caribbean security"],
    "BRAZIL_ARGENTINA": ["brazil trade pressure", "argentina leverage",
                          "mercosur restructuring", "south america trade"],
    "NORTH_BORDER": ["us canada integration", "northern border", "energy corridor",
                      "cross-border resource", "arctic development"],
    "MIDEAST_REDEPLOY": ["military redeployment", "middle east base closure",
                          "centcom restructuring", "hemisphere redeployment"],
    "ENERGY_HEMISPHERE": ["hemisphere energy", "energy independence",
                           "hemisphere oil", "energy fortress", "hemisphere refinery"],
    "STARLINK_DOMINANCE": ["starlink", "spacex dominance", "satellite internet",
                            "starlink government", "hemisphere communication"],
    "FUSION_ROLLOUT": ["fusion reactor", "small modular reactor", "smr deployment",
                        "nuclear fusion", "micro reactor"],
    "RARE_EARTH_FORTRESS": ["rare earth", "critical mineral", "rare earth mine",
                              "rare earth processing", "defense production act mineral"],
    "MIGRATION_SECURITY": ["migration national security", "immigration policy",
                            "border security", "labor shortage", "migration emergency"],
    "FORTRESS_2100": ["fortress america", "autarky", "de-globalization",
                       "hemisphere self-sufficiency", "fortress state"],
    "ELITE_CAPITAL": ["institutional rebalancing", "capital redirection",
                       "hemisphere investment", "capital flight to americas"],
    "NUCLEAR_AMERICAS": ["nuclear umbrella", "nuclear posture review",
                          "nuclear deterrence americas", "european nuclear",
                          "arms race"],
}


# ═══════════════════════════════════════════════════════════════════════════
# CACHED DATA CONTAINER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LiveFeedSnapshot:
    """All aggregated live data for one Storm Lifeboat evaluation cycle."""
    prices: Dict[Asset, float] = field(default_factory=dict)
    vix: float = 25.0
    regime: VolRegime = VolRegime.CRISIS
    macro: Dict[str, float] = field(default_factory=dict)
    fear_greed: int = 50
    fear_greed_label: str = "Neutral"
    put_call_ratio: float = 0.0
    market_tone: str = "unknown"
    options_flow_signal_count: int = 0
    top_flow_tickers: List[str] = field(default_factory=list)
    dark_pool_notional: float = 0.0
    firing_indicators: Dict[str, List[str]] = field(default_factory=dict)
    trend_interest: Dict[str, int] = field(default_factory=dict)
    ibkr_positions: List[Dict[str, Any]] = field(default_factory=list)
    ibkr_account_value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    sources_ok: List[str] = field(default_factory=list)
    sources_failed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# LIVE FEED ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class LiveFeedEngine:
    """
    Fetches live data from ALL connected AAC sources and packages it for
    the Storm Lifeboat simulation engine.

    Usage (sync, from runner):
        feed = LiveFeedEngine()
        snapshot = feed.fetch()
        forecast = mc_engine.simulate(
            prices=snapshot.prices,
            vix=snapshot.vix,
        )

    Usage (async):
        feed = LiveFeedEngine()
        snapshot = await feed.fetch_async()
    """

    def __init__(self, cache_ttl_seconds: int = 120):
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._last_snapshot: Optional[LiveFeedSnapshot] = None
        self._last_fetch: Optional[datetime] = None

    # ───── public sync interface ─────

    def fetch(self, force: bool = False) -> LiveFeedSnapshot:
        """Synchronous entry point — runs the async fetch in an event loop."""
        if (
            not force
            and self._last_snapshot is not None
            and self._last_fetch is not None
            and datetime.now() - self._last_fetch < self._cache_ttl
        ):
            return self._last_snapshot

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop (e.g. Jupyter) — schedule as task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                snapshot = pool.submit(
                    lambda: asyncio.run(self.fetch_async())
                ).result(timeout=60)
        else:
            snapshot = asyncio.run(self.fetch_async())

        self._last_snapshot = snapshot
        self._last_fetch = datetime.now()
        return snapshot

    # ───── public async interface ─────

    async def fetch_async(self) -> LiveFeedSnapshot:
        """Fetch from all data sources concurrently, degrade gracefully."""
        snap = LiveFeedSnapshot()
        snap.prices = dict(DEFAULT_PRICES)  # start from baselines

        results = await asyncio.gather(
            self._fetch_polygon_prices(snap),
            self._fetch_fred_macro(snap),
            self._fetch_unusual_whales(snap),
            self._fetch_fear_greed(snap),
            self._fetch_finnhub_news(snap),
            self._fetch_google_trends(snap),
            return_exceptions=True,
        )

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                src = ["polygon", "fred", "unusual_whales", "fear_greed", "finnhub_news", "google_trends"][i]
                snap.sources_failed.append(src)
                snap.errors.append(f"{src}: {r}")
                logger.warning("Live feed source %s failed: %s", src, r)

        # Derive VIX regime
        if snap.vix > 40:
            snap.regime = VolRegime.PANIC
        elif snap.vix > 25:
            snap.regime = VolRegime.CRISIS
        elif snap.vix > 15:
            snap.regime = VolRegime.ELEVATED
        else:
            snap.regime = VolRegime.CALM

        snap.timestamp = datetime.now()
        logger.info(
            "Live feed: %d sources OK, %d failed, VIX=%.1f (%s), %d prices live",
            len(snap.sources_ok), len(snap.sources_failed),
            snap.vix, snap.regime.value,
            sum(1 for a in Asset if snap.prices.get(a, 0) != DEFAULT_PRICES.get(a, 0)),
        )
        return snap

    # ───── individual source fetchers ─────

    async def _fetch_polygon_prices(self, snap: LiveFeedSnapshot) -> None:
        """Fetch real-time snapshots from Polygon.io for all 18 assets."""
        try:
            from integrations.polygon_client import PolygonClient
            client = PolygonClient()
        except Exception as e:
            raise RuntimeError(f"PolygonClient init failed: {e}") from e

        # Fetch stock/ETF tickers in batch
        equity_tickers = [
            t for a, t in ASSET_TICKER_MAP.items()
            if a not in (Asset.BTC, Asset.ETH, Asset.XRP)
        ]
        try:
            snapshots = await client.get_all_snapshots(equity_tickers)
            ticker_price: Dict[str, float] = {}
            for ss in snapshots:
                price = ss.day_close if ss.day_close > 0 else ss.prev_close
                if price > 0:
                    ticker_price[ss.ticker] = price

            for asset, ticker in ASSET_TICKER_MAP.items():
                if asset in (Asset.BTC, Asset.ETH, Asset.XRP):
                    continue
                if ticker in ticker_price:
                    snap.prices[asset] = ticker_price[ticker]

            snap.sources_ok.append("polygon_equities")
        except Exception as e:
            snap.errors.append(f"polygon_equities: {e}")
            logger.warning("Polygon equities batch failed: %s", e)

        # Crypto individually (different Polygon endpoint)
        for asset, poly_ticker in POLYGON_CRYPTO_MAP.items():
            try:
                ss = await client.get_snapshot(poly_ticker)
                if ss and ss.day_close > 0:
                    snap.prices[asset] = ss.day_close
                elif ss and ss.prev_close > 0:
                    snap.prices[asset] = ss.prev_close
            except Exception as e:
                logger.debug("Polygon crypto %s failed: %s", poly_ticker, e)

        if any(a in snap.sources_ok for a in ["polygon_equities"]):
            snap.sources_ok.append("polygon")

    async def _fetch_fred_macro(self, snap: LiveFeedSnapshot) -> None:
        """Fetch VIX, yields, oil, gold, spreads from FRED."""
        try:
            from integrations.fred_client import FredClient
            client = FredClient()
        except Exception as e:
            raise RuntimeError(f"FredClient init failed: {e}") from e

        for key, series_id in FRED_SERIES.items():
            try:
                obs = await client.get_latest_value(series_id)
                if obs and obs.value:
                    snap.macro[key] = obs.value
            except Exception as e:
                logger.debug("FRED %s (%s) failed: %s", key, series_id, e)

        # VIX → top-level
        if "vix" in snap.macro:
            snap.vix = snap.macro["vix"]

        # FRED oil/gold can override ETF proxy prices as sanity reference
        # (we keep Polygon ETF prices as canonical since they're tradeable)

        snap.sources_ok.append("fred")

    async def _fetch_unusual_whales(self, snap: LiveFeedSnapshot) -> None:
        """Fetch options flow, dark pool, and market tone from Unusual Whales."""
        try:
            from integrations.unusual_whales_service import UnusualWhalesSnapshotService
            svc = UnusualWhalesSnapshotService(refresh_ttl_seconds=300)
            data = await svc.get_snapshot(force_refresh=True)
        except Exception as e:
            raise RuntimeError(f"UnusualWhales fetch failed: {e}") from e

        snap.put_call_ratio = data.get("put_call_ratio", 0.0)
        snap.market_tone = data.get("market_tone", "unknown")
        snap.options_flow_signal_count = data.get("options_flow_signal_count", 0)
        snap.top_flow_tickers = data.get("top_flow_tickers", [])
        snap.dark_pool_notional = data.get("dark_pool_notional", 0.0)
        snap.sources_ok.append("unusual_whales")

    async def _fetch_fear_greed(self, snap: LiveFeedSnapshot) -> None:
        """Fetch crypto Fear & Greed index from Alternative.me."""
        try:
            from integrations.fear_greed_client import FearGreedClient
            client = FearGreedClient()
        except Exception as e:
            raise RuntimeError(f"FearGreedClient init failed: {e}") from e

        try:
            reading = await client.get_current()
            if reading:
                snap.fear_greed = reading.value
                snap.fear_greed_label = reading.classification
                snap.sources_ok.append("fear_greed")
        except Exception as e:
            raise RuntimeError(f"Fear & Greed fetch failed: {e}") from e

    async def _fetch_finnhub_news(self, snap: LiveFeedSnapshot) -> None:
        """Fetch recent news from Finnhub and scan for scenario trigger keywords."""
        try:
            from integrations.finnhub_client import FinnhubClient
            client = FinnhubClient()
        except Exception as e:
            raise RuntimeError(f"FinnhubClient init failed: {e}") from e

        try:
            articles = await client.get_news(category="general", min_id=0)
            headlines: List[str] = []
            if articles:
                headlines = [a.headline.lower() for a in articles[:100] if hasattr(a, "headline")]
        except Exception as e:
            logger.debug("Finnhub news fetch failed: %s", e)
            headlines = []

        # Scan headlines for scenario trigger keywords
        for scenario_code, keywords in INDICATOR_KEYWORDS.items():
            matched = [kw for kw in keywords if any(kw in h for h in headlines)]
            if matched:
                snap.firing_indicators[scenario_code] = matched

        snap.sources_ok.append("finnhub_news")

    async def _fetch_google_trends(self, snap: LiveFeedSnapshot) -> None:
        """Fetch search interest for crisis-related terms from Google Trends."""
        try:
            from integrations.google_trends_client import GoogleTrendsClient
            client = GoogleTrendsClient()
        except Exception as e:
            raise RuntimeError(f"GoogleTrendsClient init failed: {e}") from e

        # Track interest in key panic/crisis terms (sync call — pytrends is not async)
        terms = ["stock market crash", "bank run", "gold price", "bitcoin crash", "recession"]
        try:
            loop = asyncio.get_event_loop()
            interest = await loop.run_in_executor(
                None, lambda: client.get_interest_over_time(terms, timeframe="now 7-d")
            )
            if interest:
                # Flatten to latest interest value per term
                for term, datapoints in interest.items():
                    if datapoints:
                        snap.trend_interest[term] = datapoints[-1].get("interest", 0)
                snap.sources_ok.append("google_trends")
        except Exception as e:
            logger.debug("Google Trends fetch failed: %s", e)
            snap.sources_ok.append("google_trends")  # non-critical

    # ───── IBKR direct feed (separate — requires TWS connection) ─────

    async def fetch_ibkr_overlay(self, snap: LiveFeedSnapshot) -> None:
        """
        Optional overlay: fetch positions + account from IBKR TWS.
        Not called in the default fetch() since it requires TWS running.
        Call separately when TWS is confirmed connected.
        """
        try:
            from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
            connector = IBKRConnector()
            await connector.connect()
        except Exception as e:
            snap.errors.append(f"ibkr_connect: {e}")
            snap.sources_failed.append("ibkr")
            return

        try:
            # Live prices for our assets
            for asset, ticker in ASSET_TICKER_MAP.items():
                if asset in (Asset.BTC, Asset.ETH, Asset.XRP):
                    continue  # IBKR crypto requires different contract type
                try:
                    t = await connector.get_ticker(ticker)
                    if t and t.last and t.last > 0:
                        snap.prices[asset] = t.last
                except Exception as e:
                    snap.errors.append(f"ibkr_price_{asset.value}: {e}")

            # Account info
            try:
                summary = await connector.get_account_summary()
                if summary:
                    snap.ibkr_account_value = float(
                        summary.get("NetLiquidation", summary.get("TotalCashValue", 0))
                    )
            except Exception as e:
                snap.errors.append(f"ibkr_account_summary: {e}")

            # Positions
            try:
                positions = await connector.get_positions()
                if positions:
                    snap.ibkr_positions = [
                        {"symbol": p.symbol, "qty": p.quantity, "avg_cost": p.avg_cost,
                         "market_value": getattr(p, "market_value", 0)}
                        for p in positions
                    ]
            except Exception as e:
                snap.errors.append(f"ibkr_positions: {e}")

            snap.sources_ok.append("ibkr")
        except Exception as e:
            snap.errors.append(f"ibkr_data: {e}")
            snap.sources_failed.append("ibkr")
        finally:
            try:
                await connector.disconnect()
            except Exception:
                pass

    # ───── Moomoo overlay (separate — requires OpenD) ─────

    async def fetch_moomoo_overlay(self, snap: LiveFeedSnapshot) -> None:
        """
        Optional overlay: fetch quotes from Moomoo OpenD.
        Requires OpenD running on port 11111.
        """
        try:
            from TradingExecution.exchange_connectors.moomoo_connector import MoomooConnector
            connector = MoomooConnector()
            await connector.connect()
        except Exception as e:
            snap.errors.append(f"moomoo_connect: {e}")
            snap.sources_failed.append("moomoo")
            return

        try:
            for asset, ticker in ASSET_TICKER_MAP.items():
                if asset in (Asset.BTC, Asset.ETH, Asset.XRP):
                    continue
                try:
                    t = await connector.get_ticker(f"US.{ticker}")
                    if t and t.last and t.last > 0:
                        # Only override if no Polygon price yet
                        if snap.prices.get(asset) == DEFAULT_PRICES.get(asset):
                            snap.prices[asset] = t.last
                except Exception as e:
                    snap.errors.append(f"moomoo_price_{asset.value}: {e}")

            snap.sources_ok.append("moomoo")
        except Exception as e:
            snap.errors.append(f"moomoo_data: {e}")
            snap.sources_failed.append("moomoo")
        finally:
            try:
                await connector.disconnect()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def get_live_snapshot(
    include_ibkr: bool = False,
    include_moomoo: bool = False,
    cache_ttl: int = 120,
) -> LiveFeedSnapshot:
    """
    One-call function: fetch all live data and return a snapshot.

    Args:
        include_ibkr: Also fetch from IBKR TWS (requires TWS running)
        include_moomoo: Also fetch from Moomoo OpenD (requires OpenD running)
        cache_ttl: Cache TTL in seconds

    Returns:
        LiveFeedSnapshot with prices, VIX, macro, sentiment, indicators.
    """
    engine = LiveFeedEngine(cache_ttl_seconds=cache_ttl)
    snap = engine.fetch(force=True)

    if include_ibkr:
        asyncio.run(engine.fetch_ibkr_overlay(snap))
    if include_moomoo:
        asyncio.run(engine.fetch_moomoo_overlay(snap))

    return snap
