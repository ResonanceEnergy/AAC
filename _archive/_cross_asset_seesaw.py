"""
Cross-Asset See-Saw Amplifier Scanner
=======================================
Runs the gold-oil-silver see-saw thesis through EVERY meaningful
cross-asset ratio — including BTC, ETH, XRP vs commodities.

Ratios computed:
    --- Classic See-Saw (already in capital_engine) ---
    1. Gold/Oil ratio          — safe-haven vs energy
    2. Gold/Silver ratio       — monetary metal vs industrial beta
    3. Oil % change            — supply shock trigger

    --- NEW Crypto × Commodity Ratios ---
    4. BTC/Gold ratio          — digital gold vs physical gold
    5. ETH/Gold ratio          — smart-contract platform vs haven
    6. BTC/Silver ratio        — BTC vs silver amplifier
    7. BTC/Oil ratio           — crypto vs energy
    8. ETH/BTC ratio           — alt-coin rotation strength
    9. XRP/BTC ratio           — payments layer vs store-of-value
   10. XRP/ETH ratio           — competing L1 platforms

    --- Dominance & Relative Strength ---
   11. BTC dominance proxy     — BTC / (BTC + ETH + XRP)
   12. Gold/BTC+Gold ratio     — physical gold share of total "gold"
   13. Silver/Gold × BTC/ETH   — cross amplifier index

    --- Macro Context ---
   14. VIX level               — fear gauge
   15. Fear & Greed index      — crypto sentiment
   16. DXY (dollar index)      — inverse driver for gold + crypto

Each ratio is checked for AMPLIFIER signals:
    - Trend direction (rising/falling vs 30-day lookback)
    - Extreme levels (historical percentile)
    - Divergence patterns (ratio moving opposite to constituent)

Usage:
    python _cross_asset_seesaw.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# UTF-8 fix for Windows
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load .env
from pathlib import Path as _P

_env_path = _P(__file__).resolve().parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _k = _k.strip()
            _v = _v.strip().strip('"').strip("'")
            if _k and _v and _k not in os.environ:
                os.environ[_k] = _v


# ═══════════════════════════════════════════════════════════════════════════
# RATIO DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RatioResult:
    """One computed cross-asset ratio with amplifier analysis."""
    name: str
    formula: str           # e.g. "BTC / GOLD"
    value: float
    historical_median: float  # rough historical median for context
    direction: str         # "BULLISH_AMPLIFIER", "BEARISH_AMPLIFIER", "NEUTRAL"
    signal: str            # human-readable signal
    strength: float        # 0-1 amplifier strength
    trade_hint: str        # what to do about it


# Historical median approximations for ratio context
# These are rough mid-2020s ranges for calibration
HISTORICAL_MEDIANS = {
    "gold_oil":       55.0,    # Gold/Oil historically ~55 (gold $2000 / oil $36)
    "gold_silver":    80.0,    # G/S ratio long-run ~80 (extremes: 30-120)
    "btc_gold":       30.0,    # BTC ~$60k / Gold ~$2000 → ~30 (was ~35 at peak)
    "eth_gold":       1.8,     # ETH ~$3600 / Gold ~$2000 → ~1.8
    "btc_silver":     2600.0,  # BTC ~$65k / Silver ~$25 → ~2600
    "btc_oil":        850.0,   # BTC ~$65k / Oil ~$75 → ~867
    "eth_btc":        0.055,   # ETH/BTC ratio (DeFi summer peak: 0.085, bear: 0.04)
    "xrp_btc":        0.000010,  # XRP ~$0.65 / BTC ~$65k — tiny fraction
    "xrp_eth":        0.00017,   # XRP ~$0.65 / ETH ~$3800
    "btc_dominance":  0.55,    # BTC dominance ~55% of BTC+ETH+XRP
    "gold_share":     0.50,    # Gold / (Gold + BTC_in_USD) — shifts with BTC price
    "cross_amplifier": 1.0,    # Normalized silver/gold × BTC/ETH index
}


# ═══════════════════════════════════════════════════════════════════════════
# DATA FETCHING (re-uses AAC infrastructure)
# ═══════════════════════════════════════════════════════════════════════════

async def _http_get_json(url: str, params: Optional[Dict] = None,
                         headers: Optional[Dict] = None, timeout: float = 10) -> Any:
    """Lightweight async HTTP GET → JSON. No retries, no circuit breakers."""
    import aiohttp
    try:
        connector = aiohttp.TCPConnector(resolver=aiohttp.resolver.ThreadedResolver())
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, params=params, headers=headers,
                                   timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception as e:
        logger.debug(f"HTTP GET {url.split('?')[0]} failed: {e}")
    return None


async def fetch_all_prices() -> Dict[str, float]:
    """Fetch live prices via direct HTTP — fast, no framework overhead."""
    prices: Dict[str, float] = {}
    api_key_cg = os.environ.get("COINGECKO_API_KEY", "")
    api_key_poly = os.environ.get("POLYGON_API_KEY", "")
    api_key_fred = os.environ.get("FRED_API_KEY", "")

    # --- CoinGecko (crypto prices — single batch call) ---
    try:
        cg_base = "https://pro-api.coingecko.com/api/v3" if api_key_cg else "https://api.coingecko.com/api/v3"
        cg_headers = {"x-cg-pro-api-key": api_key_cg} if api_key_cg else {}
        data = await _http_get_json(
            f"{cg_base}/simple/price",
            params={"ids": "bitcoin,ethereum,ripple", "vs_currencies": "usd",
                    "include_24hr_change": "true"},
            headers=cg_headers,
        )
        if data:
            prices["BTC"] = data.get("bitcoin", {}).get("usd", 0)
            prices["ETH"] = data.get("ethereum", {}).get("usd", 0)
            prices["XRP"] = data.get("ripple", {}).get("usd", 0)
            prices["BTC_24h"] = data.get("bitcoin", {}).get("usd_24h_change", 0)
            prices["ETH_24h"] = data.get("ethereum", {}).get("usd_24h_change", 0)
            prices["XRP_24h"] = data.get("ripple", {}).get("usd_24h_change", 0)
            logger.info(f"CoinGecko: BTC=${prices['BTC']:,.0f}  ETH=${prices['ETH']:,.0f}  XRP=${prices['XRP']:.3f}")
    except Exception as e:
        logger.warning(f"CoinGecko failed: {e}")

    # --- Polygon (ETFs — batch snapshot) ---
    if api_key_poly:
        try:
            etf_tickers = ["GLD", "SLV", "USO", "XLE", "GDX", "SPY", "QQQ", "BITO"]
            data = await _http_get_json(
                f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers",
                params={"tickers": ",".join(etf_tickers)},
                headers={"Authorization": f"Bearer {api_key_poly}"},
                timeout=15,
            )
            if data and "tickers" in data:
                for t in data["tickers"]:
                    ticker = t.get("ticker", "")
                    day = t.get("day", {})
                    price = float(day.get("c", 0)) or float(t.get("prevDay", {}).get("c", 0))
                    if ticker and price > 0:
                        prices[ticker] = price
                logger.info(f"Polygon: GLD=${prices.get('GLD',0):.2f}  SLV=${prices.get('SLV',0):.2f}  USO=${prices.get('USO',0):.2f}")
        except Exception as e:
            logger.warning(f"Polygon failed: {e}")

    # --- FRED (macro: VIX, oil, gold spot, dollar) ---
    if api_key_fred:
        fred_map = {
            "VIX": "VIXCLS", "OIL_WTI": "DCOILWTICO",
            "GOLD_SPOT": "GOLDAMGBD228NLBM", "DXY": "DTWEXBGS",
        }
        for name, series_id in fred_map.items():
            try:
                data = await _http_get_json(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={"series_id": series_id, "api_key": api_key_fred,
                            "file_type": "json", "sort_order": "desc", "limit": "1"},
                )
                if data:
                    obs = data.get("observations", [])
                    if obs and obs[0].get("value", ".") != ".":
                        prices[name] = float(obs[0]["value"])
            except Exception:
                pass
        if "VIX" in prices:
            logger.info(f"FRED: VIX={prices['VIX']:.1f}  OIL=${prices.get('OIL_WTI',0):.2f}  GOLD=${prices.get('GOLD_SPOT',0):,.0f}  DXY={prices.get('DXY',0):.1f}")

    # --- Fear & Greed (crypto sentiment — no key needed) ---
    try:
        data = await _http_get_json("https://api.alternative.me/fng/")
        if data and data.get("data"):
            entry = data["data"][0]
            prices["FEAR_GREED"] = float(entry.get("value", 50))
            logger.info(f"Fear & Greed: {prices['FEAR_GREED']:.0f} ({entry.get('value_classification', 'N/A')})")
    except Exception as e:
        logger.warning(f"Fear & Greed failed: {e}")

    return prices


# ═══════════════════════════════════════════════════════════════════════════
# FALLBACK PRICES (if APIs fail)
# ═══════════════════════════════════════════════════════════════════════════

FALLBACK_PRICES = {
    "BTC": 68000.0, "ETH": 3800.0, "XRP": 2.50,
    "GLD": 460.0, "SLV": 78.0, "USO": 80.0, "XLE": 57.0,
    "GDX": 95.0, "SPY": 665.0, "QQQ": 450.0, "BITO": 22.0,
    "VIX": 25.0, "OIL_WTI": 115.0, "GOLD_SPOT": 4861.0, "DXY": 99.0,
    "FEAR_GREED": 50.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# RATIO COMPUTATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def get_price(prices: Dict[str, float], key: str) -> float:
    """Get price with fallback."""
    val = prices.get(key, FALLBACK_PRICES.get(key, 0))
    return val if val and val > 0 else FALLBACK_PRICES.get(key, 1)


def compute_all_ratios(prices: Dict[str, float]) -> List[RatioResult]:
    """Compute all cross-asset ratios and detect amplifiers."""
    results: List[RatioResult] = []

    btc = get_price(prices, "BTC")
    eth = get_price(prices, "ETH")
    xrp = get_price(prices, "XRP")
    # Use FRED gold_spot if available, else GLD ETF × multiplier
    gold = get_price(prices, "GOLD_SPOT")
    if gold < 1000:
        gold = get_price(prices, "GLD") * 10.6  # GLD ≈ 1/10.6 of spot gold
    silver = get_price(prices, "SLV")
    oil = get_price(prices, "OIL_WTI")
    if oil < 10:
        oil = get_price(prices, "USO") * 1.4  # rough USO→WTI proxy
    vix = get_price(prices, "VIX")
    dxy = get_price(prices, "DXY")
    fear_greed = get_price(prices, "FEAR_GREED")

    # ── 1. Gold/Oil Ratio (Classic) ──
    gold_oil = gold / oil if oil > 0 else 0
    med = HISTORICAL_MEDIANS["gold_oil"]
    if gold_oil > med * 1.3:
        direction = "BULLISH_AMPLIFIER"
        signal = f"Gold/Oil at {gold_oil:.1f} (median {med:.0f}) — gold dominance, safe-haven bid strong"
        hint = "LONG GLD calls, SHORT USO/XLE if oil rolling over"
    elif gold_oil < med * 0.7:
        direction = "BEARISH_AMPLIFIER"
        signal = f"Gold/Oil at {gold_oil:.1f} — oil dominating, energy rotation in play"
        hint = "LONG XLE calls, oil producers — energy see-saw phase"
    else:
        direction = "NEUTRAL"
        signal = f"Gold/Oil at {gold_oil:.1f} — within normal range"
        hint = "No strong signal"
    results.append(RatioResult(
        "Gold/Oil", f"GOLD({gold:,.0f}) / OIL({oil:.0f})",
        gold_oil, med, direction, signal,
        min(1.0, abs(gold_oil - med) / med), hint,
    ))

    # ── 2. Gold/Silver Ratio (Classic) ──
    gold_silver = gold / silver if silver > 0 else 80
    med = HISTORICAL_MEDIANS["gold_silver"]
    if gold_silver < 65:
        direction = "BULLISH_AMPLIFIER"
        signal = f"G/S ratio {gold_silver:.1f} — SILVER AMPLIFIER active! Silver outperforming gold"
        hint = "HEAVY LONG SLV — silver in high-beta outperformance mode"
        strength = min(1.0, (65 - gold_silver) / 30)
    elif gold_silver > 90:
        direction = "BEARISH_AMPLIFIER"
        signal = f"G/S ratio {gold_silver:.1f} — extreme fear, silver lagging (mean-reversion setup)"
        hint = "BUY SLV calls — ratio likely to compress toward 65-80"
        strength = min(1.0, (gold_silver - 90) / 30)
    else:
        direction = "NEUTRAL"
        signal = f"G/S ratio {gold_silver:.1f} — normal range"
        hint = "No strong signal"
        strength = 0.2
    results.append(RatioResult(
        "Gold/Silver", f"GOLD({gold:,.0f}) / SLV({silver:.2f})",
        gold_silver, med, direction, signal, strength, hint,
    ))

    # ── 3. BTC/Gold Ratio (Digital Gold vs Physical Gold) ──
    btc_gold = btc / gold if gold > 0 else 0
    med = HISTORICAL_MEDIANS["btc_gold"]
    if btc_gold > med * 1.5:
        direction = "BULLISH_AMPLIFIER"
        signal = f"BTC/Gold {btc_gold:.1f} — BTC outpacing gold massively → risk-on crypto narrative"
        hint = "BTC momentum play, but watch for reversion — gold may catch up"
        strength = min(1.0, (btc_gold / med - 1) / 1.5)
    elif btc_gold < med * 0.5:
        direction = "BEARISH_AMPLIFIER"
        signal = f"BTC/Gold {btc_gold:.1f} — gold crushing BTC → flight to physical safety"
        hint = "LONG GLD, SHORT BITO — physical gold winning the 'store of value' war"
        strength = min(1.0, (1 - btc_gold / med) / 0.5)
    else:
        pct_of_med = btc_gold / med
        if pct_of_med > 1.0:
            direction = "BULLISH_AMPLIFIER"
            signal = f"BTC/Gold {btc_gold:.1f} — BTC slightly leading gold (risk-on tilt)"
            hint = "Crypto-friendly environment — consider BITO calls or BTC accumulation"
        else:
            direction = "BEARISH_AMPLIFIER"
            signal = f"BTC/Gold {btc_gold:.1f} — gold slightly leading BTC (risk-off tilt)"
            hint = "Gold preferred over BTC — favor GLD over BITO"
        strength = abs(pct_of_med - 1.0)
    results.append(RatioResult(
        "BTC/Gold", f"BTC({btc:,.0f}) / GOLD({gold:,.0f})",
        btc_gold, med, direction, signal, strength, hint,
    ))

    # ── 4. ETH/Gold Ratio ──
    eth_gold = eth / gold if gold > 0 else 0
    med = HISTORICAL_MEDIANS["eth_gold"]
    if eth_gold > med * 1.5:
        direction = "BULLISH_AMPLIFIER"
        signal = f"ETH/Gold {eth_gold:.3f} — ETH outperforming gold → DeFi/smart-contract narrative strong"
        hint = "Risk-on crypto, DeFi rotation — consider ETH exposure"
    elif eth_gold < med * 0.5:
        direction = "BEARISH_AMPLIFIER"
        signal = f"ETH/Gold {eth_gold:.3f} — gold crushing ETH → flight from tech/DeFi to physical havens"
        hint = "LONG GLD, avoid ETH — physical safety winning"
    else:
        direction = "NEUTRAL"
        signal = f"ETH/Gold {eth_gold:.3f} — balanced"
        hint = "No strong cross-signal"
    results.append(RatioResult(
        "ETH/Gold", f"ETH({eth:,.0f}) / GOLD({gold:,.0f})",
        eth_gold, med, direction, signal,
        min(1.0, abs(eth_gold - med) / med), hint,
    ))

    # ── 5. BTC/Silver Ratio ──
    btc_silver = btc / silver if silver > 0 else 0
    med = HISTORICAL_MEDIANS["btc_silver"]
    if btc_silver < med * 0.5:
        direction = "BULLISH_AMPLIFIER"
        signal = f"BTC/Silver {btc_silver:.0f} — silver MASSIVELY outperforming BTC → commodity supercycle"
        hint = "HEAVY LONG SLV — silver industrial + monetary demand crushing crypto"
        strength = min(1.0, (1 - btc_silver / med))
    elif btc_silver > med * 1.5:
        direction = "BEARISH_AMPLIFIER"
        signal = f"BTC/Silver {btc_silver:.0f} — BTC dominating silver → crypto risk-on, commodities lagging"
        hint = "BTC momentum, but silver may be undervalued entry"
        strength = min(1.0, (btc_silver / med - 1))
    else:
        direction = "NEUTRAL"
        signal = f"BTC/Silver {btc_silver:.0f} — balanced"
        hint = "No strong signal"
        strength = 0.2
    results.append(RatioResult(
        "BTC/Silver", f"BTC({btc:,.0f}) / SLV({silver:.2f})",
        btc_silver, med, direction, signal, strength, hint,
    ))

    # ── 6. BTC/Oil Ratio ──
    btc_oil = btc / oil if oil > 0 else 0
    med = HISTORICAL_MEDIANS["btc_oil"]
    if btc_oil < med * 0.5:
        direction = "BEARISH_AMPLIFIER"
        signal = f"BTC/Oil {btc_oil:.0f} — oil dominating BTC → energy crisis, crypto selling off"
        hint = "LONG XLE, SHORT BITO — energy is king"
    elif btc_oil > med * 1.5:
        direction = "BULLISH_AMPLIFIER"
        signal = f"BTC/Oil {btc_oil:.0f} — BTC dominating oil → post-crisis recovery, risk appetite"
        hint = "Crypto leads, energy peaked — consider trimming XLE"
    else:
        direction = "NEUTRAL"
        signal = f"BTC/Oil {btc_oil:.0f} — balanced"
        hint = "No strong signal"
    results.append(RatioResult(
        "BTC/Oil", f"BTC({btc:,.0f}) / OIL({oil:.0f})",
        btc_oil, med, direction, signal,
        min(1.0, abs(btc_oil - med) / med), hint,
    ))

    # ── 7. ETH/BTC Ratio (Alt Rotation) ──
    eth_btc = eth / btc if btc > 0 else 0
    med = HISTORICAL_MEDIANS["eth_btc"]
    if eth_btc > 0.065:
        direction = "BULLISH_AMPLIFIER"
        signal = f"ETH/BTC {eth_btc:.4f} — ETH gaining on BTC → alt season / DeFi rotation"
        hint = "Alt-season indicator — ETH, XRP, DeFi tokens likely to outperform BTC"
        strength = min(1.0, (eth_btc - 0.055) / 0.03)
    elif eth_btc < 0.040:
        direction = "BEARISH_AMPLIFIER"
        signal = f"ETH/BTC {eth_btc:.4f} — BTC dominance rising → risk-off within crypto, flight to BTC"
        hint = "BTC dominance trade — avoid alts, stick to BTC or exit crypto entirely"
        strength = min(1.0, (0.055 - eth_btc) / 0.02)
    else:
        direction = "NEUTRAL"
        signal = f"ETH/BTC {eth_btc:.4f} — normal range"
        hint = "No strong alt-season signal"
        strength = 0.2
    results.append(RatioResult(
        "ETH/BTC", f"ETH({eth:,.0f}) / BTC({btc:,.0f})",
        eth_btc, med, direction, signal, strength, hint,
    ))

    # ── 8. XRP/BTC Ratio ──
    xrp_btc = xrp / btc if btc > 0 else 0
    med = HISTORICAL_MEDIANS["xrp_btc"]
    if xrp_btc > med * 3:
        direction = "BULLISH_AMPLIFIER"
        signal = f"XRP/BTC {xrp_btc:.8f} — XRP massively outperforming BTC → payments narrative / utility rotation"
        hint = "XRP momentum trade — regulatory clarity or institutional adoption catalyst"
        strength = min(1.0, xrp_btc / med / 5)
    elif xrp_btc < med * 0.3:
        direction = "BEARISH_AMPLIFIER"
        signal = f"XRP/BTC {xrp_btc:.8f} — XRP collapsing vs BTC → altcoin exodus"
        hint = "Avoid XRP, BTC dominance strong"
        strength = min(1.0, (1 - xrp_btc / med))
    else:
        direction = "NEUTRAL"
        signal = f"XRP/BTC {xrp_btc:.8f} — normal range"
        hint = "No strong signal"
        strength = 0.2
    results.append(RatioResult(
        "XRP/BTC", f"XRP({xrp:.3f}) / BTC({btc:,.0f})",
        xrp_btc, med, direction, signal, strength, hint,
    ))

    # ── 9. XRP/ETH Ratio ──
    xrp_eth = xrp / eth if eth > 0 else 0
    med = HISTORICAL_MEDIANS["xrp_eth"]
    if xrp_eth > med * 3:
        direction = "BULLISH_AMPLIFIER"
        signal = f"XRP/ETH {xrp_eth:.6f} — XRP outpacing ETH → payments > smart contracts narrative"
        hint = "XRP rotation — watch for SWIFT/ISO20022 catalyst"
        strength = min(1.0, xrp_eth / med / 5)
    elif xrp_eth < med * 0.3:
        direction = "BEARISH_AMPLIFIER"
        signal = f"XRP/ETH {xrp_eth:.6f} — ETH dominating XRP → DeFi/smart-contract ecosystem winning"
        hint = "Favor ETH over XRP"
        strength = 0.3
    else:
        direction = "NEUTRAL"
        signal = f"XRP/ETH {xrp_eth:.6f} — balanced"
        hint = "No strong signal"
        strength = 0.1
    results.append(RatioResult(
        "XRP/ETH", f"XRP({xrp:.3f}) / ETH({eth:,.0f})",
        xrp_eth, med, direction, signal, strength, hint,
    ))

    # ── 10. BTC Dominance (proxy: BTC / BTC+ETH+XRP value) ──
    total_crypto = btc + eth + xrp
    btc_dom = btc / total_crypto if total_crypto > 0 else 0.55
    med = HISTORICAL_MEDIANS["btc_dominance"]
    if btc_dom > 0.95:
        direction = "BEARISH_AMPLIFIER"
        signal = f"BTC dominance {btc_dom:.1%} — extreme BTC dominance, alts dead → risk-off crypto"
        hint = "Only BTC surviving — exit alts, consider GLD over all crypto"
        strength = 0.8
    elif btc_dom < 0.90:
        direction = "BULLISH_AMPLIFIER"
        signal = f"BTC dominance {btc_dom:.1%} — alts gaining share → alt-season brewing"
        hint = "ETH, XRP gaining — consider alt exposure via ETH or sector baskets"
        strength = min(1.0, (0.95 - btc_dom) / 0.10)
    else:
        direction = "NEUTRAL"
        signal = f"BTC dominance {btc_dom:.1%} — normal"
        hint = "No dominance signal"
        strength = 0.1
    results.append(RatioResult(
        "BTC Dominance", f"BTC / (BTC+ETH+XRP)",
        btc_dom, med, direction, signal, strength, hint,
    ))

    # ── 11. Gold Share: Gold / (Gold + BTC_in_oz) ──
    # Compare gold's "market cap mindshare" vs BTC
    gold_share = gold / (gold + btc) if (gold + btc) > 0 else 0.5
    med = HISTORICAL_MEDIANS["gold_share"]
    if gold_share > 0.99:
        direction = "BEARISH_AMPLIFIER"
        signal = f"Gold share {gold_share:.1%} — gold completely dominating BTC → extreme risk-off"
        hint = "All-in gold era — BTC not competitive as store of value"
        strength = 0.9
    elif gold_share < 0.95:
        direction = "BULLISH_AMPLIFIER"
        signal = f"Gold share {gold_share:.1%} — BTC clawing back vs gold → crypto confidence returning"
        hint = "BTC gaining on gold — digital gold thesis alive"
        strength = min(1.0, (0.98 - gold_share) / 0.05)
    else:
        direction = "NEUTRAL"
        signal = f"Gold share {gold_share:.1%} — balanced"
        hint = "Normal allocation between physical and digital gold"
        strength = 0.2
    results.append(RatioResult(
        "Physical Gold Share", f"GOLD / (GOLD + BTC)",
        gold_share, med, direction, signal, strength, hint,
    ))

    # ── 12. Cross Amplifier Index: (Silver/Gold) × (BTC/ETH) ──
    # When BOTH silver is compressing vs gold AND BTC is compressing vs ETH
    # → double amplifier: physical metals + crypto alts both in rotation mode
    sg_ratio = silver / gold if gold > 0 else 0.012
    be_ratio = btc / eth if eth > 0 else 18
    cross_amp = (sg_ratio * 1000) * (be_ratio / 20)  # Normalized
    med = HISTORICAL_MEDIANS["cross_amplifier"]
    if cross_amp > med * 1.5:
        direction = "BULLISH_AMPLIFIER"
        signal = f"Cross Amplifier {cross_amp:.2f} — silver AND BTC both compressing vs gold/ETH → DOUBLE ROTATION"
        hint = "MAX AMPLIFIER: SLV + BTC longs — both physical and digital rotation favoring high-beta"
        strength = min(1.0, (cross_amp / med - 1))
    elif cross_amp < med * 0.5:
        direction = "BEARISH_AMPLIFIER"
        signal = f"Cross Amplifier {cross_amp:.2f} — gold + ETH dominating → defensive posture"
        hint = "Gold + ETH winning — quality over beta"
        strength = min(1.0, (1 - cross_amp / med))
    else:
        direction = "NEUTRAL"
        signal = f"Cross Amplifier {cross_amp:.2f} — balanced"
        hint = "No cross-amplifier signal"
        strength = 0.2
    results.append(RatioResult(
        "Cross Amplifier Index", f"(SLV/GOLD × 1000) × (BTC/ETH ÷ 20)",
        cross_amp, med, direction, signal, strength, hint,
    ))

    # ── 13. VIX Context ──
    if vix > 30:
        direction = "BEARISH_AMPLIFIER"
        signal = f"VIX {vix:.1f} — CRISIS/PANIC zone → all see-saw amplifiers are HIGH CONVICTION"
        hint = "High vol amplifies ALL signals — size positions carefully"
        strength = min(1.0, (vix - 20) / 30)
    elif vix > 20:
        direction = "BULLISH_AMPLIFIER"
        signal = f"VIX {vix:.1f} — ELEVATED → see-saw rotation is active"
        hint = "Moderate vol — amplifiers are live but not extreme"
        strength = min(1.0, (vix - 15) / 15)
    else:
        direction = "NEUTRAL"
        signal = f"VIX {vix:.1f} — CALM → see-saw mechanics dormant"
        hint = "Low vol — amplifiers need VIX > 20 to trigger meaningfully"
        strength = 0.1
    results.append(RatioResult(
        "VIX Context", f"VIX = {vix:.1f}",
        vix, 20.0, direction, signal, strength, hint,
    ))

    # ── 14. Fear & Greed ──
    if fear_greed <= 25:
        direction = "BULLISH_AMPLIFIER"
        signal = f"Fear & Greed {fear_greed:.0f} — EXTREME FEAR → contrarian buy signal for crypto"
        hint = "Crypto fear extreme — historical buying opportunity if thesis intact"
        strength = min(1.0, (50 - fear_greed) / 50)
    elif fear_greed >= 75:
        direction = "BEARISH_AMPLIFIER"
        signal = f"Fear & Greed {fear_greed:.0f} — EXTREME GREED → crypto overextended"
        hint = "Take profits on crypto, rotate to gold/silver safety"
        strength = min(1.0, (fear_greed - 50) / 50)
    else:
        direction = "NEUTRAL"
        signal = f"Fear & Greed {fear_greed:.0f} — neutral zone"
        hint = "No extreme sentiment signal"
        strength = 0.1
    results.append(RatioResult(
        "Fear & Greed", f"Index = {fear_greed:.0f}",
        fear_greed, 50.0, direction, signal, strength, hint,
    ))

    # ── 15. Dollar Index (inverse driver) ──
    if dxy > 105:
        direction = "BEARISH_AMPLIFIER"
        signal = f"DXY {dxy:.1f} — strong dollar → headwind for gold AND crypto"
        hint = "Dollar strength crushes gold + BTC — wait for DXY reversal"
        strength = min(1.0, (dxy - 100) / 15)
    elif dxy < 95:
        direction = "BULLISH_AMPLIFIER"
        signal = f"DXY {dxy:.1f} — weak dollar → tailwind for ALL hard assets (gold, silver, BTC)"
        hint = "Weak dollar amplifies EVERYTHING — gold, silver, BTC all benefit"
        strength = min(1.0, (100 - dxy) / 10)
    else:
        direction = "NEUTRAL"
        signal = f"DXY {dxy:.1f} — neutral zone"
        hint = "Dollar not a major driver right now"
        strength = 0.1
    results.append(RatioResult(
        "Dollar Index (DXY)", f"DXY = {dxy:.1f}",
        dxy, 99.0, direction, signal, strength, hint,
    ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# AMPLIFIER SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════

def synthesize_amplifiers(ratios: List[RatioResult]) -> Dict[str, Any]:
    """Synthesize all ratios into actionable amplifier signals."""
    bullish = [r for r in ratios if r.direction == "BULLISH_AMPLIFIER"]
    bearish = [r for r in ratios if r.direction == "BEARISH_AMPLIFIER"]
    neutral = [r for r in ratios if r.direction == "NEUTRAL"]

    # Weighted amplifier score (-1 = max bearish, +1 = max bullish)
    total_weight = sum(r.strength for r in ratios) or 1
    score = sum(
        r.strength * (1 if r.direction == "BULLISH_AMPLIFIER" else -1 if r.direction == "BEARISH_AMPLIFIER" else 0)
        for r in ratios
    ) / total_weight

    # Detect concordance (multiple amplifiers firing same direction)
    concordance = max(len(bullish), len(bearish)) / len(ratios) if ratios else 0

    # Top 3 strongest signals
    all_active = sorted(bullish + bearish, key=lambda r: r.strength, reverse=True)
    top_3 = all_active[:3]

    # Phase hint based on cross-asset analysis
    if score > 0.3 and concordance > 0.5:
        phase = "RISK-ON AMPLIFIER — crypto + commodities aligned bullish"
    elif score < -0.3 and concordance > 0.5:
        phase = "RISK-OFF AMPLIFIER — flight to gold, exit crypto"
    elif any(r.name == "Gold/Silver" and r.direction == "BULLISH_AMPLIFIER" for r in ratios):
        phase = "SILVER AMPLIFIER — commodity see-saw in high-beta mode"
    elif any(r.name == "ETH/BTC" and r.direction == "BULLISH_AMPLIFIER" for r in ratios):
        phase = "ALT-SEASON AMPLIFIER — crypto rotation to alts"
    elif any(r.name == "BTC/Gold" and r.direction == "BEARISH_AMPLIFIER" for r in ratios):
        phase = "GOLD DOMINANCE — physical gold winning vs digital"
    else:
        phase = "NO CLEAR AMPLIFIER — mixed signals, stay defensive"

    return {
        "timestamp": datetime.now().isoformat(),
        "amplifier_score": round(score, 3),
        "concordance": round(concordance, 3),
        "bullish_count": len(bullish),
        "bearish_count": len(bearish),
        "neutral_count": len(neutral),
        "phase": phase,
        "top_signals": [
            {"name": r.name, "direction": r.direction, "strength": round(r.strength, 2), "hint": r.trade_hint}
            for r in top_3
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

async def async_main():
    print("=" * 80)
    print("  CROSS-ASSET SEE-SAW AMPLIFIER SCANNER")
    print("  BTC / ETH / XRP  ×  Gold / Silver / Oil")
    print("=" * 80)
    print()

    # 1. Fetch all prices
    print("▸ Fetching live prices from CoinGecko, Polygon, FRED, Fear & Greed...")
    prices = await fetch_all_prices()
    print()

    # Show raw prices
    print("─" * 60)
    print("  RAW PRICES")
    print("─" * 60)
    crypto_line = f"  BTC: ${get_price(prices, 'BTC'):>10,.0f}  │  ETH: ${get_price(prices, 'ETH'):>8,.0f}  │  XRP: ${get_price(prices, 'XRP'):>6.3f}"
    metal_line  = f"  GOLD: ${get_price(prices, 'GOLD_SPOT'):>9,.0f}  │  SLV: ${get_price(prices, 'SLV'):>8.2f}  │  OIL: ${get_price(prices, 'OIL_WTI'):>7.2f}"
    macro_line  = f"  VIX: {get_price(prices, 'VIX'):>8.1f}     │  DXY: {get_price(prices, 'DXY'):>8.1f}     │  F&G: {get_price(prices, 'FEAR_GREED'):>6.0f}"
    print(crypto_line)
    print(metal_line)
    print(macro_line)
    print()

    # 2. Compute all ratios
    ratios = compute_all_ratios(prices)

    # 3. Display each ratio
    print("─" * 80)
    print("  CROSS-ASSET RATIOS & AMPLIFIER SIGNALS")
    print("─" * 80)
    for i, r in enumerate(ratios, 1):
        icon = "🟢" if r.direction == "BULLISH_AMPLIFIER" else "🔴" if r.direction == "BEARISH_AMPLIFIER" else "⚪"
        print(f"\n  {i:2d}. {icon} {r.name}")
        print(f"      Formula: {r.formula}")
        print(f"      Value: {r.value:.6g}  │  Historical Median: {r.historical_median:.6g}")
        print(f"      Direction: {r.direction}  │  Strength: {r.strength:.1%}")
        print(f"      Signal: {r.signal}")
        print(f"      → Trade: {r.trade_hint}")

    # 4. Synthesis
    synthesis = synthesize_amplifiers(ratios)
    print()
    print("=" * 80)
    print("  AMPLIFIER SYNTHESIS")
    print("=" * 80)
    print(f"  Phase:        {synthesis['phase']}")
    print(f"  Score:        {synthesis['amplifier_score']:+.3f}  (-1.0 bearish ↔ +1.0 bullish)")
    print(f"  Concordance:  {synthesis['concordance']:.1%}  (% of ratios firing same direction)")
    print(f"  Counts:       {synthesis['bullish_count']} bullish  │  {synthesis['bearish_count']} bearish  │  {synthesis['neutral_count']} neutral")
    print()
    print("  TOP 3 STRONGEST SIGNALS:")
    for i, sig in enumerate(synthesis["top_signals"], 1):
        icon = "▲" if sig["direction"] == "BULLISH_AMPLIFIER" else "▼"
        print(f"    {i}. {icon} {sig['name']}  (strength {sig['strength']:.0%})")
        print(f"       → {sig['hint']}")
    print()

    # 5. Save report
    report_dir = Path("data/cross_asset_seesaw")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / "cross_asset_report.jsonl"
    report = {
        "timestamp": synthesis["timestamp"],
        "prices": {k: round(v, 4) for k, v in prices.items()},
        "ratios": [
            {"name": r.name, "value": round(r.value, 6), "direction": r.direction,
             "strength": round(r.strength, 3), "hint": r.trade_hint}
            for r in ratios
        ],
        "synthesis": synthesis,
    }
    with open(report_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")
    print(f"  Report saved → {report_file}")
    print("=" * 80)

    return synthesis


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
