"""
War Room Live Data Feeds
========================
Fetches live market data from CoinGecko, Unusual Whales, MetaMask (on-chain),
and NDAX exchange, then patches IndicatorState and SPOT_PRICES in war_room_engine.

Usage:
    from strategies.war_room_live_feeds import update_all_live_data
    indicators = await update_all_live_data()  # async
    indicators = update_all_live_data_sync()    # sync wrapper
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import sys

# -- UTF-8 stdout fix for Windows cp1252 terminals --
if hasattr(sys, "stdout") and sys.stdout is not None:
    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("WarRoomLiveFeeds")


# ============================================================================
# RESULT CONTAINER
# ============================================================================

@dataclass
class LiveFeedResult:
    """Aggregated result from all live data feeds."""
    timestamp: str = ""
    # CoinGecko prices
    btc_price: Optional[float] = None
    eth_price: Optional[float] = None
    xrp_price: Optional[float] = None
    # CoinGecko global
    total_market_cap_usd: Optional[float] = None
    btc_dominance: Optional[float] = None
    defi_market_cap: Optional[float] = None
    total_volume_24h: Optional[float] = None
    # Unusual Whales intelligence
    put_call_ratio: Optional[float] = None
    market_tone: Optional[str] = None
    options_flow_signals: int = 0
    total_options_premium: float = 0.0
    dark_pool_trades: int = 0
    dark_pool_notional: float = 0.0
    congress_trades: int = 0
    top_flow_tickers: List[str] = field(default_factory=list)
    # MetaMask on-chain balances
    metamask_matic: Optional[float] = None
    metamask_usdc_polygon: Optional[float] = None
    metamask_eth: Optional[float] = None
    metamask_usdc_eth: Optional[float] = None
    metamask_address: str = ""
    # NDAX balances (CAD exchange)
    ndax_balances: Dict[str, float] = field(default_factory=dict)
    ndax_net_cad: float = 0.0
    # Finnhub equity prices
    spy_price: Optional[float] = None
    # FRED macro indicators
    gold_price_oz: Optional[float] = None
    oil_price_wti: Optional[float] = None
    fed_rate: Optional[float] = None
    dxy_index: Optional[float] = None
    hy_spread_bp_live: Optional[float] = None
    # Stablecoin depeg (from CoinGecko USDT/USDC)
    stablecoin_depeg_pct: Optional[float] = None
    # Fear & Greed (alternative.me)
    fear_greed_value: Optional[int] = None
    # News severity (NewsAPI)
    news_severity_score: Optional[float] = None
    news_headline_count: int = 0
    # X/Twitter sentiment
    x_sentiment_score: Optional[float] = None
    # Council intel (YouTube + X via Grok)
    council_scenario_signals: Dict[str, float] = field(default_factory=dict)
    council_topics: List[str] = field(default_factory=list)
    council_emerging: List[str] = field(default_factory=list)
    council_x_posts: int = 0
    council_yt_videos: int = 0
    # IBKR live data
    ibkr_connected: bool = False
    ibkr_net_liquidation: Optional[float] = None
    ibkr_buying_power: Optional[float] = None
    ibkr_total_cash: Optional[float] = None
    ibkr_unrealized_pnl: Optional[float] = None
    ibkr_realized_pnl: Optional[float] = None
    ibkr_maint_margin: Optional[float] = None
    ibkr_positions: List[Dict[str, Any]] = field(default_factory=list)
    ibkr_vix: Optional[float] = None
    ibkr_spy_price: Optional[float] = None
    # DeFi TVL % change (from CoinGecko defi_market_cap vs baseline)
    defi_tvl_change_pct: Optional[float] = None
    # BDC NAV discount (from yfinance BDC basket price-to-book)
    bdc_nav_discount_live: Optional[float] = None
    # BDC non-accrual proxy (estimated from BDC P/B + HY spread)
    bdc_nonaccrual_proxy: Optional[float] = None
    # Alpha engine intelligence (from council + doctrine combined)
    alpha_signal: Optional[float] = None
    alpha_weights: Dict[str, float] = field(default_factory=dict)
    # Errors (non-fatal -- partial data is still useful)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary for logging."""
        parts = []
        if self.btc_price:
            parts.append(f"BTC ${self.btc_price:,.0f}")
        if self.eth_price:
            parts.append(f"ETH ${self.eth_price:,.0f}")
        if self.spy_price:
            parts.append(f"SPY ${self.spy_price:,.0f}")
        if self.gold_price_oz:
            parts.append(f"Gold ${self.gold_price_oz:,.0f}")
        if self.put_call_ratio:
            parts.append(f"P/C {self.put_call_ratio:.2f}")
        if self.market_tone:
            parts.append(f"Tone:{self.market_tone}")
        if self.fear_greed_value is not None:
            parts.append(f"FGI:{self.fear_greed_value}")
        if self.metamask_usdc_polygon is not None:
            parts.append(f"MM-USDC ${self.metamask_usdc_polygon:,.2f}")
        if self.ndax_net_cad > 0:
            parts.append(f"NDAX C${self.ndax_net_cad:,.2f}")
        if self.ibkr_connected:
            parts.append(f"IBKR NetLiq=${self.ibkr_net_liquidation or 0:,.0f}")
            parts.append(f"{len(self.ibkr_positions)} pos")
        if self.ibkr_vix:
            parts.append(f"VIX:{self.ibkr_vix:.1f}")
        if self.errors:
            parts.append(f"({len(self.errors)} errors)")
        return " | ".join(parts) if parts else "No data fetched"


# ============================================================================
# 1. COINGECKO — Live crypto prices + global market data
# ============================================================================

# CoinGecko IDs for our tracked assets
_CG_COIN_MAP = {
    "bitcoin": "btc",
    "ethereum": "eth",
    "ripple": "xrp",
    "tether": "usdt",
    "usd-coin": "usdc",
}

# ── DeFi TVL baseline for % change computation ──────────────────────────
# Reference defi_market_cap from CoinGecko (rolling baseline).
# On first run, use this reference value. Subsequent runs update it.
# DeFi market cap ~$90B as of early 2026 (CoinGecko /global endpoint).
_DEFI_MARKET_CAP_BASELINE: float = 90_000_000_000.0  # $90B reference
_defi_market_cap_prev: Optional[float] = None


def _compute_defi_tvl_change(current_defi_cap: float) -> float:
    """Compute DeFi TVL % change vs rolling baseline.

    Uses module-level state: first call uses hardcoded baseline,
    subsequent calls compare against previously fetched value.
    Returns negative = DeFi stress (TVL declining).
    """
    global _defi_market_cap_prev
    baseline = _defi_market_cap_prev if _defi_market_cap_prev else _DEFI_MARKET_CAP_BASELINE
    if baseline <= 0:
        return 0.0
    change_pct = ((current_defi_cap - baseline) / baseline) * 100.0
    _defi_market_cap_prev = current_defi_cap
    return round(change_pct, 2)


# ── BDC basket for NAV discount + non-accrual proxy ─────────────────────
_BDC_BASKET = ["ARCC", "MAIN", "FSK", "OBDC"]


async def fetch_bdc_data(result: LiveFeedResult) -> None:
    """Fetch BDC NAV discount and non-accrual proxy from yfinance.

    BDC NAV Discount: average (1 - price/bookValue) across BDC basket.
    Non-Accrual Proxy: estimated from price-to-book stress level.
      - P/B < 0.75 → non-accrual ~6%+ (severe stress)
      - P/B 0.75-0.85 → non-accrual ~4-6% (moderate stress)
      - P/B 0.85-0.95 → non-accrual ~2-4% (mild stress)
      - P/B > 0.95 → non-accrual ~1-2% (healthy)
    """
    try:
        import yfinance as yf

        loop = asyncio.get_event_loop()
        discounts = []
        p2b_ratios = []

        def _fetch_bdc_basket() -> tuple[list[float], list[float]]:
            disc: list[float] = []
            p2b: list[float] = []
            for sym in _BDC_BASKET:
                try:
                    ticker = yf.Ticker(sym)
                    info = ticker.info
                    price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
                    book = info.get("bookValue", 0)
                    if price and price > 0 and book and book > 0:
                        pb_ratio = price / book
                        p2b.append(pb_ratio)
                        nav_discount = (1.0 - pb_ratio) * 100.0
                        disc.append(nav_discount)
                except Exception:
                    continue
            return disc, p2b

        discounts, p2b_ratios = await loop.run_in_executor(None, _fetch_bdc_basket)

        if discounts:
            avg_discount = sum(discounts) / len(discounts)
            result.bdc_nav_discount_live = round(avg_discount, 2)
            logger.info("BDC NAV discount: %.1f%% (from %d BDCs)", avg_discount, len(discounts))

        if p2b_ratios:
            avg_pb = sum(p2b_ratios) / len(p2b_ratios)
            # Map P/B to non-accrual estimate (piecewise linear)
            if avg_pb < 0.75:
                nonaccrual = 6.0 + (0.75 - avg_pb) * 20.0  # severe: 6-10%
            elif avg_pb < 0.85:
                nonaccrual = 4.0 + (0.85 - avg_pb) * 20.0  # moderate: 4-6%
            elif avg_pb < 0.95:
                nonaccrual = 2.0 + (0.95 - avg_pb) * 20.0  # mild: 2-4%
            else:
                nonaccrual = max(0.5, 2.0 - (avg_pb - 0.95) * 10.0)  # healthy: 0.5-2%
            result.bdc_nonaccrual_proxy = round(min(nonaccrual, 10.0), 2)
            logger.info("BDC non-accrual proxy: %.1f%% (avg P/B=%.3f)", result.bdc_nonaccrual_proxy, avg_pb)

    except Exception as exc:
        msg = f"BDC data error: {exc}"
        logger.warning(msg)
        result.errors.append(msg)


async def fetch_coingecko_data(result: LiveFeedResult) -> None:
    """Fetch live crypto prices and global data from CoinGecko."""
    try:
        from shared.data_sources import CoinGeckoClient
        cg = CoinGeckoClient()
        await cg.connect()
        try:
            # Batch price fetch -- single API call for all 3 coins
            ticks = await cg.get_prices_batch(
                list(_CG_COIN_MAP.keys()), vs_currency="usd"
            )
            usdt_price = None
            usdc_price = None
            for tick in ticks:
                # tick.symbol is "BITCOIN/USD" etc.
                coin_id = tick.symbol.split("/")[0].lower()
                if coin_id == "bitcoin":
                    result.btc_price = tick.price
                elif coin_id == "ethereum":
                    result.eth_price = tick.price
                elif coin_id == "ripple":
                    result.xrp_price = tick.price
                elif coin_id == "tether":
                    usdt_price = tick.price
                elif coin_id in ("usd-coin", "usd coin"):
                    usdc_price = tick.price

            # Stablecoin depeg: max deviation from $1.00
            depegs = []
            if usdt_price is not None:
                depegs.append(abs(usdt_price - 1.0))
            if usdc_price is not None:
                depegs.append(abs(usdc_price - 1.0))
            if depegs:
                result.stablecoin_depeg_pct = max(depegs) * 100

            # Global market data (BTC dominance, volume, DeFi cap)
            # NOTE: get_global_data() already unwraps the "data" key.
            global_data = await cg.get_global_data()
            if global_data:
                result.total_market_cap_usd = global_data.get("total_market_cap", {}).get("usd")
                result.btc_dominance = global_data.get("market_cap_percentage", {}).get("btc")
                result.total_volume_24h = global_data.get("total_volume", {}).get("usd")
                # DeFi market cap — compute TVL change vs baseline
                defi_cap = global_data.get("defi_market_cap")
                if defi_cap:
                    result.defi_market_cap = float(defi_cap)
                    result.defi_tvl_change_pct = _compute_defi_tvl_change(float(defi_cap))

            logger.info("CoinGecko: BTC=$%s ETH=$%s XRP=$%s DeFi=$%s",
                        result.btc_price, result.eth_price, result.xrp_price,
                        result.defi_market_cap)
        finally:
            await cg.disconnect()
    except Exception as exc:
        msg = f"CoinGecko error: {exc}"
        logger.warning(msg)
        result.errors.append(msg)


# ============================================================================
# 2. UNUSUAL WHALES — Options flow, dark pool, congressional trades
# ============================================================================

async def fetch_unusual_whales_data(result: LiveFeedResult) -> None:
    """Fetch options flow intelligence from Unusual Whales."""
    try:
        from integrations.unusual_whales_service import get_unusual_whales_snapshot_service
        service = get_unusual_whales_snapshot_service()
        snapshot = await service.get_snapshot(force_refresh=True)

        if snapshot.get("status") == "healthy":
            result.put_call_ratio = snapshot.get("put_call_ratio", 0.0)
            result.market_tone = snapshot.get("market_tone", "unknown")
            result.options_flow_signals = snapshot.get("options_flow_signal_count", 0)
            result.total_options_premium = snapshot.get("total_options_premium", 0.0)
            result.dark_pool_trades = snapshot.get("dark_pool_trade_count", 0)
            result.dark_pool_notional = snapshot.get("dark_pool_notional", 0.0)
            result.congress_trades = snapshot.get("congress_trade_count", 0)
            result.top_flow_tickers = snapshot.get("top_flow_tickers", [])
            logger.info("Unusual Whales: P/C=%.2f Tone=%s Flow=%d DarkPool=%d Congress=%d",
                        result.put_call_ratio, result.market_tone,
                        result.options_flow_signals, result.dark_pool_trades,
                        result.congress_trades)
        elif snapshot.get("status") == "unconfigured":
            result.errors.append("Unusual Whales: API key not configured")
        else:
            err = snapshot.get("error", "unknown error")
            result.errors.append(f"Unusual Whales: {err}")
    except Exception as exc:
        msg = f"Unusual Whales error: {exc}"
        logger.warning(msg)
        result.errors.append(msg)


# ============================================================================
# 3. METAMASK — On-chain balances (Polygon + Ethereum)
# ============================================================================

_POLYGON_RPCS = [
    "https://1rpc.io/matic",
    "https://polygon-bor-rpc.publicnode.com",
    "https://polygon-rpc.com",
    "https://rpc.ankr.com/polygon",
]
_ETH_RPCS = [
    "https://eth.llamarpc.com",
    "https://rpc.ankr.com/eth",
    "https://1rpc.io/eth",
]
_USDC_POLYGON = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
_USDC_ETH = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
_ERC20_BALANCE_ABI = [
    {"constant": True, "inputs": [{"name": "_owner", "type": "address"}],
     "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}],
     "type": "function"},
]


async def fetch_metamask_balances(result: LiveFeedResult) -> None:
    """Check on-chain balances: MATIC, ETH, USDC on Polygon + Ethereum."""
    try:
        from web3 import Web3

        eoa = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "")
        if not eoa:
            pk = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
            if pk:
                from eth_account import Account
                eoa = Account.from_key(pk).address
        if not eoa:
            result.errors.append("MetaMask: No wallet address (set POLYMARKET_FUNDER_ADDRESS)")
            return

        result.metamask_address = eoa

        # Polygon balances
        for rpc in _POLYGON_RPCS:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
                if w3.is_connected():
                    # MATIC/POL native balance
                    matic_wei = w3.eth.get_balance(eoa)
                    result.metamask_matic = float(w3.from_wei(matic_wei, "ether"))
                    # USDC.e on Polygon (6 decimals)
                    usdc_contract = w3.eth.contract(
                        address=Web3.to_checksum_address(_USDC_POLYGON),
                        abi=_ERC20_BALANCE_ABI,
                    )
                    usdc_raw = usdc_contract.functions.balanceOf(eoa).call()
                    result.metamask_usdc_polygon = float(usdc_raw) / 1e6
                    break
            except Exception:
                continue

        # Ethereum balances
        for rpc in _ETH_RPCS:
            try:
                w3e = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
                if w3e.is_connected():
                    eth_wei = w3e.eth.get_balance(eoa)
                    result.metamask_eth = float(w3e.from_wei(eth_wei, "ether"))
                    usdc_eth_contract = w3e.eth.contract(
                        address=Web3.to_checksum_address(_USDC_ETH),
                        abi=_ERC20_BALANCE_ABI,
                    )
                    usdc_eth_raw = usdc_eth_contract.functions.balanceOf(eoa).call()
                    result.metamask_usdc_eth = float(usdc_eth_raw) / 1e6
                    break
            except Exception:
                continue

        logger.info("MetaMask: MATIC=%.4f USDC_Poly=%.2f ETH=%.6f USDC_Eth=%.2f",
                    result.metamask_matic or 0, result.metamask_usdc_polygon or 0,
                    result.metamask_eth or 0, result.metamask_usdc_eth or 0)
    except ImportError:
        result.errors.append("MetaMask: web3 package not installed")
    except Exception as exc:
        msg = f"MetaMask error: {exc}"
        logger.warning(msg)
        result.errors.append(msg)


# ============================================================================
# 4. NDAX — Canadian crypto exchange balances via ccxt
# ============================================================================

async def fetch_ndax_balances(result: LiveFeedResult) -> None:
    """Fetch NDAX exchange balances via ccxt (sync, run in executor)."""
    try:
        import ccxt
        api_key = os.getenv("NDAX_API_KEY", "")
        api_secret = os.getenv("NDAX_API_SECRET", "")
        uid = os.getenv("NDAX_USER_ID", "")

        if not api_key:
            result.errors.append("NDAX: NDAX_API_KEY not set")
            return

        # ccxt NDAX requires uid + login + password
        exchange = ccxt.ndax({
            "apiKey": api_key,
            "secret": api_secret,
            "uid": uid,
            "login": uid,
            "password": api_secret,
        })

        # Run sync ccxt in thread to avoid blocking event loop
        loop = asyncio.get_running_loop()
        bal = await loop.run_in_executor(None, exchange.fetch_balance)

        total = bal.get("total", {})
        for asset, amount in total.items():
            if amount and float(amount) > 0:
                result.ndax_balances[asset] = float(amount)

        result.ndax_net_cad = sum(result.ndax_balances.values())
        logger.info("NDAX: %s (net C$%.2f)", result.ndax_balances, result.ndax_net_cad)
    except ImportError:
        result.errors.append("NDAX: ccxt package not installed")
    except Exception as exc:
        msg = f"NDAX error: {exc}"
        logger.warning(msg)
        result.errors.append(msg)


# ============================================================================
# 5. FINNHUB — Live equity quotes (SPY)
# ============================================================================

async def _finnhub_quote(symbol: str) -> Optional[float]:
    """Fetch a single quote from Finnhub. Returns current price or None."""
    import aiohttp
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return None
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = data.get("c", 0)
                    return price if price and price > 0 else None
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as e:
        logging.getLogger(__name__).debug("Finnhub quote fetch failed for %s: %s", symbol, e)
    return None


async def fetch_finnhub_prices(result: LiveFeedResult) -> None:
    """Fetch live SPY and GLD prices from Finnhub REST API."""
    try:
        api_key = os.getenv("FINNHUB_API_KEY", "")
        if not api_key:
            result.errors.append("Finnhub: FINNHUB_API_KEY not set")
            return

        # Fetch SPY and GLD concurrently
        spy_price, gld_price = await asyncio.gather(
            _finnhub_quote("SPY"),
            _finnhub_quote("GLD"),
        )

        if spy_price:
            result.spy_price = spy_price
            logger.info("Finnhub SPY: $%.2f", spy_price)
        else:
            result.errors.append("Finnhub SPY: no price (market closed?)")

        # GLD ETF tracks gold at ~1/10.9 oz per share
        # Convert GLD price to gold spot price per troy ounce
        if gld_price:
            result.gold_price_oz = round(gld_price * 10.9, 2)
            logger.info("Finnhub Gold: $%.2f/oz (GLD=$%.2f)", result.gold_price_oz, gld_price)

    except Exception as exc:
        result.errors.append(f"Finnhub error: {exc}")


# ============================================================================
# 6. FRED — Macro indicators (gold, fed rate, DXY, HY spread)
# ============================================================================

async def _fred_latest(series_id: str) -> Optional[float]:
    """Fetch latest value from a FRED series via direct REST API."""
    import aiohttp
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        return None
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&limit=1&sort_order=desc"
        f"&api_key={api_key}&file_type=json"
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    obs = data.get("observations", [])
                    if obs and obs[0].get("value", ".") != ".":
                        return float(obs[0]["value"])
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as e:
        logging.getLogger(__name__).debug("FRED fetch failed for %s: %s", series_id, e)
    return None


async def fetch_fred_indicators(result: LiveFeedResult) -> None:
    """Fetch gold price, fed funds rate, DXY proxy, and HY spread from FRED."""
    try:
        api_key = os.getenv("FRED_API_KEY", "")
        if not api_key:
            result.errors.append("FRED: FRED_API_KEY not set")
            return

        # Run all FRED fetches concurrently (4 series, 4 HTTP calls)
        oil_task = _fred_latest("DCOILWTICO")
        fed_task = _fred_latest("DFF")
        dxy_task = _fred_latest("DTWEXBGS")
        hy_task = _fred_latest("BAMLH0A0HYM2")

        oil, fed, dtwexbgs, hy = await asyncio.gather(
            oil_task, fed_task, dxy_task, hy_task
        )

        # WTI Crude Oil (daily) -- maps to IndicatorState.oil_price
        if oil and oil > 0:
            result.oil_price_wti = oil
            logger.info("FRED Oil (WTI): $%.2f", oil)

        # Fed funds effective rate (daily) -- maps to IndicatorState.fed_funds_rate
        if fed is not None:
            result.fed_rate = fed
            logger.info("FRED Fed Rate: %.2f%%", fed)

        # Broad Dollar Index (proxy for DXY) -- rescale to DXY range
        # DTWEXBGS range ~110-140, DXY range ~90-115. Scale factor ~0.82
        if dtwexbgs and dtwexbgs > 0:
            result.dxy_index = round(dtwexbgs * 0.82, 2)
            logger.info("FRED DXY proxy: %.2f (raw DTWEXBGS=%.2f)", result.dxy_index, dtwexbgs)

        # HY spread (FRED gives in %, convert to basis points)
        if hy is not None and hy > 0:
            result.hy_spread_bp_live = hy * 100
            logger.info("FRED HY spread: %.0f bp", result.hy_spread_bp_live)

    except Exception as exc:
        result.errors.append(f"FRED error: {exc}")


# ============================================================================
# 7. FEAR & GREED INDEX — alternative.me (free, no key required)
# ============================================================================

async def fetch_fear_greed(result: LiveFeedResult) -> None:
    """Fetch crypto Fear & Greed Index from alternative.me (free API)."""
    try:
        import aiohttp
        url = "https://api.alternative.me/fng/?limit=1"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    fng_data = data.get("data", [])
                    if fng_data:
                        val = int(fng_data[0].get("value", 50))
                        classification = fng_data[0].get("value_classification", "")
                        result.fear_greed_value = val
                        logger.info("Fear & Greed: %d (%s)", val, classification)
                else:
                    result.errors.append(f"Fear&Greed: HTTP {resp.status}")
    except Exception as exc:
        result.errors.append(f"Fear&Greed error: {exc}")


# ============================================================================
# 8. NEWS SEVERITY — NewsAPI headline crisis scanning
# ============================================================================

_CRISIS_KEYWORDS = {
    "crash": 0.15, "collapse": 0.15, "recession": 0.12, "crisis": 0.12,
    "war": 0.10, "default": 0.10, "panic": 0.10, "plunge": 0.08,
    "contagion": 0.08, "black swan": 0.15, "bank run": 0.12,
    "liquidation": 0.08, "margin call": 0.08, "sell-off": 0.06,
    "selloff": 0.06, "downgrade": 0.05, "tariff": 0.05,
    "sanctions": 0.05, "inflation": 0.04, "layoffs": 0.04,
}


async def fetch_news_severity(result: LiveFeedResult) -> None:
    """Fetch business headlines from NewsAPI and score crisis severity."""
    try:
        import aiohttp
        api_key = os.getenv("NEWS_API_KEY", "")
        if not api_key:
            result.errors.append("News: NEWS_API_KEY not set")
            return

        url = (
            f"https://newsapi.org/v2/top-headlines"
            f"?country=us&category=business&pageSize=30&apiKey={api_key}"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    articles = data.get("articles", [])
                    if not articles:
                        return
                    severity = 0.0
                    hits = 0
                    for article in articles:
                        title = (article.get("title") or "").lower()
                        desc = (article.get("description") or "").lower()
                        text = f"{title} {desc}"
                        for keyword, weight in _CRISIS_KEYWORDS.items():
                            if keyword in text:
                                severity += weight
                                hits += 1
                    # Normalize: divide by 3.0 baseline so normal news ~0.3-0.4,
                    # heavy crisis day ~0.7-0.9, black swan ~1.0
                    result.news_severity_score = round(min(1.0, severity / 3.0), 3)
                    result.news_headline_count = len(articles)
                    logger.info("News: severity=%.3f from %d articles (%d keyword hits)",
                                result.news_severity_score, len(articles), hits)
                elif resp.status == 401:
                    result.errors.append("News: Invalid API key")
                elif resp.status == 429:
                    result.errors.append("News: Rate limit exceeded (100/day free tier)")
                else:
                    result.errors.append(f"News: HTTP {resp.status}")
    except Exception as exc:
        result.errors.append(f"News error: {exc}")


# ============================================================================
# 9. X/TWITTER SENTIMENT — v2 recent search with keyword scoring
# ============================================================================

_BEARISH_TERMS = ["crash", "selloff", "sell off", "collapse", "recession", "bear market",
                   "panic", "margin call", "black monday", "circuit breaker", "plunge"]
_BULLISH_TERMS = ["rally", "breakout", "all time high", "ath", "moon", "bull market",
                   "buying the dip", "recovery", "ripping", "green candle"]


async def fetch_x_sentiment(result: LiveFeedResult) -> None:
    """Fetch X/Twitter sentiment via council X (Grok-powered search).

    Replaces the broken direct Twitter API v2 (HTTP 402) with the working
    councils/xai pipeline that uses Grok's built-in X search capability.
    Falls back to neutral 0.5 if council fails.
    """
    try:
        from strategies.war_room_council_feeds import fetch_x_intel

        x_result = await fetch_x_intel()

        if x_result.x_posts_analyzed > 0:
            result.x_sentiment_score = x_result.x_sentiment_score
            result.council_x_posts = x_result.x_posts_analyzed
            result.council_topics.extend(x_result.x_key_themes[:10])
            result.council_emerging.extend(x_result.x_emerging_topics[:5])

            # Merge scenario signals
            for sc, strength in x_result.scenario_signals.items():
                existing = result.council_scenario_signals.get(sc, 0.0)
                result.council_scenario_signals[sc] = max(existing, strength)

            # Blend X severity into news severity
            if x_result.x_severity_score > 0:
                existing = result.news_severity_score or 0.0
                result.news_severity_score = round(existing * 0.6 + x_result.x_severity_score * 0.4, 3)

            logger.info("X Council Sentiment: %.3f (%d posts, %d themes, %d scenarios)",
                        result.x_sentiment_score, x_result.x_posts_analyzed,
                        len(x_result.x_key_themes), len(x_result.scenario_signals))
        else:
            result.x_sentiment_score = 0.5
            logger.info("X Council: no posts analyzed, defaulting to neutral 0.5")

    except ImportError:
        result.errors.append("X Council: war_room_council_feeds not importable")
        result.x_sentiment_score = 0.5
    except Exception as exc:
        result.errors.append(f"X Council error: {exc}")
        result.x_sentiment_score = 0.5


# ============================================================================
# 10. IBKR — Live positions, account summary, VIX, SPY via TWS/Gateway
# ============================================================================

async def fetch_ibkr_data(result: LiveFeedResult) -> None:
    """Fetch live account data from IBKR via ib_insync.

    Pulls: account summary (net liquidation, buying power, margin, P&L),
    all positions with current market values, and live VIX + SPY quotes.
    Requires TWS or IB Gateway running on IBKR_HOST:IBKR_PORT.
    """
    try:
        from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
    except ImportError:
        result.errors.append("IBKR: ibkr_connector not importable")
        return

    connector = None
    try:
        connector = IBKRConnector()
        connected = await connector.connect()
        if not connected:
            result.errors.append("IBKR: Failed to connect to TWS/Gateway")
            return

        result.ibkr_connected = True

        # --- Account Summary ---
        try:
            summary = await connector.get_account_summary()
            result.ibkr_net_liquidation = summary.get("NetLiquidation_USD") or summary.get("NetLiquidation_BASE")
            result.ibkr_buying_power = summary.get("BuyingPower_USD") or summary.get("BuyingPower_BASE")
            result.ibkr_total_cash = summary.get("TotalCashValue_USD") or summary.get("TotalCashValue_BASE")
            result.ibkr_unrealized_pnl = summary.get("UnrealizedPnL_USD") or summary.get("UnrealizedPnL_BASE")
            result.ibkr_realized_pnl = summary.get("RealizedPnL_USD") or summary.get("RealizedPnL_BASE")
            result.ibkr_maint_margin = summary.get("MaintMarginReq_USD") or summary.get("MaintMarginReq_BASE")
            logger.info("IBKR Account: NetLiq=$%.2f BuyPow=$%.2f Cash=$%.2f uPnL=$%.2f",
                        result.ibkr_net_liquidation or 0, result.ibkr_buying_power or 0,
                        result.ibkr_total_cash or 0, result.ibkr_unrealized_pnl or 0)
        except Exception as exc:
            result.errors.append(f"IBKR account summary: {exc}")

        # --- Positions ---
        try:
            positions = await connector.get_positions()
            result.ibkr_positions = positions
            logger.info("IBKR Positions: %d open", len(positions))
            for pos in positions:
                logger.info("  %s %s qty=%s avgCost=%.2f mktVal=%.2f",
                            pos.get("symbol"), pos.get("sec_type"),
                            pos.get("quantity"), pos.get("avg_cost", 0),
                            pos.get("market_value", 0))
        except Exception as exc:
            result.errors.append(f"IBKR positions: {exc}")

        # --- Live VIX + SPY via raw ib_insync (Index contract, not Stock) ---
        try:
            from ib_insync import Index, Stock
            ib = connector._ib
            if ib and ib.isConnected():
                # VIX is an Index on CBOE, not a Stock
                vix_contract = Index('VIX', 'CBOE')
                ib.qualifyContracts(vix_contract)
                vix_ticker = ib.reqMktData(vix_contract)
                # SPY is a Stock on SMART
                spy_contract = Stock('SPY', 'SMART', 'USD')
                ib.qualifyContracts(spy_contract)
                spy_ticker = ib.reqMktData(spy_contract)
                await asyncio.sleep(2)  # Wait for market data

                # Extract VIX
                vix_val = vix_ticker.last if (vix_ticker.last and vix_ticker.last > 0) else None
                if not vix_val:
                    vix_val = vix_ticker.close if (getattr(vix_ticker, 'close', None) and vix_ticker.close > 0) else None
                if vix_val:
                    result.ibkr_vix = vix_val
                    logger.info("IBKR VIX: %.2f", result.ibkr_vix)

                # Extract SPY
                spy_val = spy_ticker.last if (spy_ticker.last and spy_ticker.last > 0) else None
                if not spy_val:
                    spy_val = spy_ticker.close if (getattr(spy_ticker, 'close', None) and spy_ticker.close > 0) else None
                if spy_val:
                    result.ibkr_spy_price = spy_val
                    logger.info("IBKR SPY: $%.2f", result.ibkr_spy_price)

                # Clean up
                ib.cancelMktData(vix_contract)
                ib.cancelMktData(spy_contract)
        except Exception as exc:
            logger.debug("IBKR VIX/SPY ticker: %s", exc)

    except Exception as exc:
        msg = f"IBKR error: {exc}"
        logger.warning(msg)
        result.errors.append(msg)
    finally:
        if connector:
            try:
                await connector.disconnect()
            except Exception:
                pass


# ============================================================================
# 11. VIX — FRED VIXCLS (fallback if IBKR VIX unavailable)
# ============================================================================

async def fetch_vix_fred(result: LiveFeedResult) -> None:
    """Fetch VIX from FRED VIXCLS series as fallback."""
    if result.ibkr_vix is not None:
        return  # Already got VIX from IBKR, skip FRED
    try:
        vix = await _fred_latest("VIXCLS")
        if vix and vix > 0:
            result.ibkr_vix = vix  # Store in same field — source doesn't matter
            logger.info("FRED VIX (VIXCLS): %.2f", vix)
    except Exception as exc:
        result.errors.append(f"FRED VIX error: {exc}")


# ============================================================================
# MASTER ORCHESTRATOR — Fetch all feeds, patch IndicatorState + SPOT_PRICES
# ============================================================================

async def fetch_all_live_data() -> LiveFeedResult:
    """Fetch all live data feeds in parallel. Non-fatal -- partial data is fine."""
    result = LiveFeedResult(timestamp=datetime.now().isoformat())

    # Phase 1: IBKR first (VIX fallback depends on knowing if IBKR got VIX)
    await fetch_ibkr_data(result)

    # Phase 2: All other feeds concurrently (including VIX FRED fallback)
    await asyncio.gather(
        fetch_coingecko_data(result),
        fetch_unusual_whales_data(result),
        fetch_metamask_balances(result),
        fetch_ndax_balances(result),
        fetch_finnhub_prices(result),
        fetch_fred_indicators(result),
        fetch_fear_greed(result),
        fetch_news_severity(result),
        fetch_x_sentiment(result),          # Now uses council X (Grok-powered)
        fetch_vix_fred(result),
        fetch_bdc_data(result),
        return_exceptions=True,
    )

    # Phase 3: YouTube council intel (supplements news severity + scenario signals)
    try:
        from strategies.war_room_council_feeds import fetch_youtube_intel
        yt_result = await fetch_youtube_intel()
        if yt_result.yt_videos_processed > 0:
            result.council_yt_videos = yt_result.yt_videos_processed
            result.council_topics.extend(yt_result.yt_key_topics[:10])
            # Merge YouTube scenario signals
            for sc, strength in yt_result.scenario_signals.items():
                existing = result.council_scenario_signals.get(sc, 0.0)
                result.council_scenario_signals[sc] = max(existing, strength)
            # Blend YouTube severity into news severity
            if yt_result.yt_severity_score > 0:
                existing = result.news_severity_score or 0.0
                result.news_severity_score = round(existing * 0.7 + yt_result.yt_severity_score * 0.3, 3)
            logger.info("YouTube Council: %d videos, sentiment=%.2f, %d scenario hits",
                        yt_result.yt_videos_processed, yt_result.yt_sentiment_score,
                        len(yt_result.scenario_signals))
    except ImportError:
        result.errors.append("YouTube Council: war_room_council_feeds not importable")
    except Exception as exc:
        result.errors.append(f"YouTube Council error: {exc}")

    return result


def _patch_spot_prices(result: LiveFeedResult) -> Dict[str, float]:
    """Patch SPOT_PRICES dict with live CoinGecko data. Returns updated prices."""
    import strategies.war_room_engine as wre

    patched = {}
    if result.btc_price and result.btc_price > 0:
        wre.SPOT_PRICES["btc"] = result.btc_price
        patched["btc"] = result.btc_price
    if result.eth_price and result.eth_price > 0:
        wre.SPOT_PRICES["eth"] = result.eth_price
        patched["eth"] = result.eth_price
    if result.xrp_price and result.xrp_price > 0:
        # XRP isn't in SPOT_PRICES by default as a traded asset but
        # it's useful for the indicator state
        patched["xrp"] = result.xrp_price

    return patched


# ── BDC/Credit arm symbol mapping ──────────────────────────────────────
_BDC_SYMBOLS = {"ARCC", "PFF", "MAIN", "BKLN", "FSK", "TCPC", "OBDC", "OWL"}
_CREDIT_SYMBOLS = {"LQD", "EMB", "JNK", "HYG", "KRE"}
_OIL_SYMBOLS = {"XLE", "XOP", "USO", "OIL", "XLF"}
_METALS_SYMBOLS = {"GLD", "SLV", "GDX", "GDXJ", "IAU"}


def _classify_arm(symbol: str, sec_type: str) -> str:
    """Classify an IBKR position into an ArmType string."""
    sym = symbol.upper()
    if sym in _BDC_SYMBOLS:
        return "bdc_nonaccrual"
    elif sym in _CREDIT_SYMBOLS:
        return "tradfi_rotate"
    elif sym in _OIL_SYMBOLS:
        return "iran_oil"
    elif sym in _METALS_SYMBOLS:
        return "crypto_metals"
    elif sec_type == "CRYPTO":
        return "defi_yield"
    else:
        return "tradfi_rotate"


def _update_ibkr_positions(ibkr_positions: List[Dict[str, Any]]) -> None:
    """Update CURRENT_POSITIONS in war_room_engine with live IBKR data.

    Only replaces positions where account="IBKR". Preserves Moomoo and
    WealthSimple positions since those aren't in this connector.
    """
    import strategies.war_room_engine as wre

    # Keep non-IBKR positions (Moomoo, WealthSimple, etc.)
    non_ibkr = [p for p in wre.CURRENT_POSITIONS if p.account != "IBKR"]

    # Build new IBKR positions from live data
    new_ibkr = []
    for pos in ibkr_positions:
        sym = pos.get("symbol", "")
        sec_type = pos.get("sec_type", "STK")
        qty = pos.get("quantity", 0)
        avg_cost = pos.get("avg_cost", 0.0)
        mkt_val = pos.get("market_value", 0.0)
        mkt_price = pos.get("market_price", 0.0)

        if qty == 0:
            continue

        # Determine position type from sec_type
        if sec_type == "OPT":
            contract_qty = abs(int(qty))
            pos_type = "put" if qty < 0 else "call"
            # market_price from portfolio() is per-share for options
            current_price = mkt_price if mkt_price else (abs(mkt_val / (contract_qty * 100)) if contract_qty > 0 else 0.0)
            # avg_cost from portfolio() is total cost per contract for options
            entry = avg_cost / 100 if avg_cost > 1 else avg_cost
        else:
            contract_qty = abs(int(qty))
            pos_type = "long" if qty > 0 else "short"
            current_price = mkt_price if mkt_price else (abs(mkt_val / contract_qty) if contract_qty > 0 else 0.0)
            entry = avg_cost

        arm_str = _classify_arm(sym, sec_type)
        try:
            arm = wre.ArmType(arm_str)
        except ValueError:
            arm = wre.ArmType.TRADFI_ROTATE

        new_ibkr.append(wre.Position(
            arm=arm,
            symbol=sym,
            position_type=pos_type,
            quantity=contract_qty,
            entry_price=entry,
            current_price=current_price,
            account="IBKR",
        ))

    # Replace CURRENT_POSITIONS (IBKR portion only)
    wre.CURRENT_POSITIONS = non_ibkr + new_ibkr
    logger.info("IBKR positions updated: %d IBKR + %d other = %d total",
                len(new_ibkr), len(non_ibkr),
                len(wre.CURRENT_POSITIONS))


def apply_live_data_to_indicators(
    result: LiveFeedResult,
    indicators: Optional[Any] = None,
) -> Any:
    """Apply fetched live data to an IndicatorState instance.

    Patches crypto prices from CoinGecko and injects Unusual Whales
    intelligence as supplementary signals. Also patches module-level
    SPOT_PRICES for Monte Carlo.

    Returns the updated IndicatorState.
    """
    from strategies.war_room_engine import IndicatorState

    ind = indicators if indicators is not None else IndicatorState()

    # -- CoinGecko: patch crypto spot prices --
    if result.btc_price and result.btc_price > 0:
        ind.btc_price = result.btc_price
    if result.eth_price and result.eth_price > 0:
        # ETH feeds into stablecoin depeg monitoring indirectly
        pass  # ETH isn't a direct IndicatorState field but is in SPOT_PRICES

    # Patch module-level SPOT_PRICES for Monte Carlo use
    _patch_spot_prices(result)

    # -- Finnhub: SPY live price (IBKR overrides if available) --
    if result.ibkr_spy_price and result.ibkr_spy_price > 0:
        ind.spy_price = result.ibkr_spy_price  # IBKR is authoritative during market hours
    elif result.spy_price and result.spy_price > 0:
        ind.spy_price = result.spy_price  # Finnhub fallback

    # -- IBKR: VIX (IBKR live → FRED VIXCLS fallback) --
    if result.ibkr_vix and result.ibkr_vix > 0:
        ind.vix = result.ibkr_vix

    # -- FRED: gold, oil, fed rate, DXY, HY spread --
    if result.gold_price_oz and result.gold_price_oz > 0:
        ind.gold_price = result.gold_price_oz
    if result.oil_price_wti and result.oil_price_wti > 0:
        ind.oil_price = result.oil_price_wti
    if result.fed_rate is not None:
        ind.fed_funds_rate = result.fed_rate
    if result.dxy_index and result.dxy_index > 0:
        ind.dxy = result.dxy_index
    if result.hy_spread_bp_live and result.hy_spread_bp_live > 0:
        ind.hy_spread_bp = result.hy_spread_bp_live

    # -- CoinGecko: stablecoin depeg --
    if result.stablecoin_depeg_pct is not None:
        ind.stablecoin_depeg_pct = result.stablecoin_depeg_pct

    # -- CoinGecko: DeFi TVL change --
    if result.defi_tvl_change_pct is not None:
        ind.defi_tvl_change_pct = result.defi_tvl_change_pct

    # -- yfinance: BDC NAV discount --
    if result.bdc_nav_discount_live is not None:
        ind.bdc_nav_discount = result.bdc_nav_discount_live

    # -- yfinance: BDC non-accrual proxy --
    if result.bdc_nonaccrual_proxy is not None:
        ind.bdc_nonaccrual_pct = result.bdc_nonaccrual_proxy

    # -- Fear & Greed Index --
    if result.fear_greed_value is not None:
        ind.fear_greed_index = float(result.fear_greed_value)

    # -- X/Twitter sentiment --
    if result.x_sentiment_score is not None:
        ind.x_sentiment = result.x_sentiment_score

    # -- NewsAPI: news severity --
    if result.news_severity_score is not None:
        # Blend NewsAPI severity with existing (don't overwrite UW data)
        ind.news_severity = max(ind.news_severity, result.news_severity_score)

    # -- Unusual Whales: enhance composite with put-call ratio + dark pool --
    # The put-call ratio is a key sentiment signal. If UW data is available,
    # we adjust the news_severity to incorporate institutional flow signals.
    # >1.2 P/C = bearish institutional positioning, <0.7 = bullish
    if result.put_call_ratio and result.put_call_ratio > 0:
        # Map P/C ratio to crisis signal: 0.5=neutral, >1.2=high stress
        if result.put_call_ratio > 1.5:
            pcr_signal = 0.9  # extreme bearish flow
        elif result.put_call_ratio > 1.2:
            pcr_signal = 0.6  # bearish flow
        elif result.put_call_ratio > 0.9:
            pcr_signal = 0.3  # mild bearish
        elif result.put_call_ratio > 0.7:
            pcr_signal = 0.15  # neutral
        else:
            pcr_signal = 0.05  # bullish flow

        # Blend UW signal with existing news severity (UW weighs 40%)
        existing_severity = ind.news_severity
        ind.news_severity = round(existing_severity * 0.6 + pcr_signal * 0.4, 3)

    # Dark pool activity: extreme volume = institutional repositioning
    if result.dark_pool_notional > 10_000_000_000:  # >$10B notional = unusual
        # Bump fear slightly — institutions moving big blocks
        ind.news_severity = min(1.0, ind.news_severity + 0.05)

    # -- Alpha engine: council + doctrine combined signal --
    if result.alpha_signal is not None:
        ind.alpha_signal = result.alpha_signal

    # -- IBKR: Update account balance + positions in war_room_engine --
    if result.ibkr_connected:
        import strategies.war_room_engine as wre
        # Update ACCOUNTS dict with live IBKR data
        if result.ibkr_net_liquidation is not None:
            wre.ACCOUNTS["IBKR"]["balance_usd"] = result.ibkr_net_liquidation
            wre.ACCOUNTS["IBKR"]["note"] = (
                f"LIVE {result.timestamp[:10]} — NetLiq=${result.ibkr_net_liquidation:,.2f}, "
                f"BuyPow=${result.ibkr_buying_power or 0:,.2f}, "
                f"uPnL=${result.ibkr_unrealized_pnl or 0:,.2f}"
            )

        # Update CURRENT_POSITIONS from live IBKR portfolio
        if result.ibkr_positions:
            _update_ibkr_positions(result.ibkr_positions)

    return ind


async def update_all_live_data(
    indicators: Optional[Any] = None,
) -> Any:
    """Master function: fetch all live feeds + apply to IndicatorState.

    Call this before compute_composite_score() for a fully live mandate.

    Returns updated IndicatorState with:
    - Live BTC/ETH/XRP from CoinGecko
    - Live put-call ratio + dark pool intelligence from Unusual Whales
    - Module-level SPOT_PRICES patched for Monte Carlo
    - MetaMask + NDAX balances available in the result

    The LiveFeedResult is stored at module level for inspection.
    """
    global _last_feed_result

    # Fetch all feeds
    result = await fetch_all_live_data()
    _last_feed_result = result

    # Apply to indicators
    ind = apply_live_data_to_indicators(result, indicators)

    # Process Council scenario transitions (if any council data arrived)
    if result.council_scenario_signals:
        try:
            from strategies.war_room_council_feeds import update_scenario_statuses
            transitions = update_scenario_statuses(result.council_scenario_signals)
            if transitions:
                logger.info("Council scenario transitions: %s", transitions)
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("Council scenario update failed: %s", exc)

    logger.info("Live feeds applied: %s", result.summary())
    return ind


def update_all_live_data_sync(
    indicators: Optional[Any] = None,
) -> Any:
    """Synchronous wrapper for update_all_live_data().

    Safe to call from CLI or non-async code.
    """
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, update_all_live_data(indicators))
            return future.result(timeout=45)
    except RuntimeError:
        return asyncio.run(update_all_live_data(indicators))


def get_last_feed_result() -> Optional[LiveFeedResult]:
    """Get the most recent LiveFeedResult for inspection/display."""
    return _last_feed_result


# Module-level storage for last fetch
_last_feed_result: Optional[LiveFeedResult] = None


# ============================================================================
# CLI — standalone test
# ============================================================================

def main():
    """Run all feeds and print results."""
    import json
    from dataclasses import asdict

    print("=" * 72)
    print("WAR ROOM LIVE FEEDS -- Fetching all data sources...")
    print("=" * 72)

    result = asyncio.run(fetch_all_live_data())

    print(f"\nTimestamp: {result.timestamp}")
    print(f"\n--- CoinGecko Crypto Prices ---")
    print(f"  BTC: ${result.btc_price:,.2f}" if result.btc_price else "  BTC: N/A")
    print(f"  ETH: ${result.eth_price:,.2f}" if result.eth_price else "  ETH: N/A")
    print(f"  XRP: ${result.xrp_price:,.4f}" if result.xrp_price else "  XRP: N/A")
    if result.total_market_cap_usd:
        print(f"  Total Market Cap: ${result.total_market_cap_usd:,.0f}")
    if result.btc_dominance:
        print(f"  BTC Dominance: {result.btc_dominance:.1f}%")

    print(f"\n--- Unusual Whales Intelligence ---")
    if result.put_call_ratio:
        print(f"  Put/Call Ratio: {result.put_call_ratio:.2f}")
        print(f"  Market Tone: {result.market_tone}")
        print(f"  Options Flow Signals: {result.options_flow_signals}")
        print(f"  Total Premium: ${result.total_options_premium:,.0f}")
        print(f"  Dark Pool Trades: {result.dark_pool_trades} (${result.dark_pool_notional:,.0f} notional)")
        print(f"  Congress Trades: {result.congress_trades}")
        if result.top_flow_tickers:
            print(f"  Top Flow Tickers: {', '.join(result.top_flow_tickers)}")
    else:
        print("  Not available")

    print(f"\n--- MetaMask On-Chain ---")
    if result.metamask_address:
        print(f"  Address: {result.metamask_address[:10]}...{result.metamask_address[-6:]}")
        print(f"  MATIC: {result.metamask_matic:.4f}" if result.metamask_matic else "  MATIC: N/A")
        print(f"  USDC (Polygon): ${result.metamask_usdc_polygon:,.2f}" if result.metamask_usdc_polygon is not None else "  USDC (Polygon): N/A")
        print(f"  ETH: {result.metamask_eth:.6f}" if result.metamask_eth else "  ETH: N/A")
        print(f"  USDC (Ethereum): ${result.metamask_usdc_eth:,.2f}" if result.metamask_usdc_eth is not None else "  USDC (Ethereum): N/A")
    else:
        print("  No wallet configured")

    print(f"\n--- NDAX Exchange ---")
    if result.ndax_balances:
        for asset, amount in result.ndax_balances.items():
            print(f"  {asset}: {amount:.4f}")
        print(f"  Net Liquidation: C${result.ndax_net_cad:,.2f}")
    else:
        print("  No balances (account may be liquidated)")

    if result.errors:
        print(f"\n--- Errors ({len(result.errors)}) ---")
        for err in result.errors:
            print(f"  ! {err}")

    print(f"\nSummary: {result.summary()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
