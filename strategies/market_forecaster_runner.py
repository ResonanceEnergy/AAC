"""
Market Forecaster Runner — AAC
================================
Unified CLI that:
  1. Pulls live macro data from FRED, Finnhub, CoinGecko, Fear & Greed
  2. Builds MacroSnapshot + CryptoSnapshot
  3. Runs RegimeEngine + StockForecaster + CryptoForecaster
  4. Prints a ranked battle plan for today's market

Usage:
    python strategies/market_forecaster_runner.py
    python strategies/market_forecaster_runner.py --crypto-only
    python strategies/market_forecaster_runner.py --stock-only
    python strategies/market_forecaster_runner.py --horizon medium
    python strategies/market_forecaster_runner.py --manual   # enter data manually

    # Inject manual override for today's macro context:
    python strategies/market_forecaster_runner.py \\
        --vix 21.5 --hy-spread 380 --oil 95.5 \\
        --core-pce 3.1 --gdp 0.7 \\
        --war --hormuz \\
        --private-redemptions 11.0

Live API feeds:
    FRED_API_KEY        — FRED macro data (yield curve, VIX, HY spread, oil, gold)
    FINNHUB_API_KEY     — sector ETF returns (HYG, KRE, JETS, ZIM)
    COINGECKO_PRO_API_KEY — BTC/ETH prices, dominance, market data
    (Fear & Greed: free, no key)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.crypto_forecaster import (
    CryptoForecaster,
    CryptoRegimeState,
    CryptoSnapshot,
    snapshot_from_coingecko,
)
from strategies.regime_engine import (
    MacroSnapshot,
    Regime,
    RegimeEngine,
    RegimeState,
    snapshot_from_fred,
)
from strategies.stock_forecaster import (
    Horizon,
    Industry,
    IndustryForecast,
    StockForecaster,
    TradeOpportunity,
    print_industry_regime_matrix,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("market_forecaster")


# ═══════════════════════════════════════════════════════════════════════════
# LIVE DATA FETCHERS (lightweight — no SDK dependencies)
# ═══════════════════════════════════════════════════════════════════════════

def _http_get(url: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """Simple urllib JSON fetch. Returns None on any error."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AAC-Forecaster/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        logger.debug("HTTP fetch failed %s: %s", url, exc)
        return None


def _fred_latest(series_ids: List[str], api_key: Optional[str] = None) -> Dict[str, float]:
    """
    Fetch latest values for multiple FRED series.
    Returns dict {series_id: float_value}.

    FRED series used:
        VIXCLS          — VIX daily close
        BAMLH0A0HYM2    — HY OAS (OA spread, not bps — multiply ×100 for bps)
        T10Y2Y          — 10Y-2Y spread (%)
        DCOILWTICO      — WTI crude oil
        GOLDAMGBD228NLBM — Gold AM fix
        T10YIE          — 10Y breakeven inflation
        DGS10           — 10Y treasury yield
        PCEPILFE        — Core PCE price index YoY
    """
    key_param = f"&api_key={api_key}" if api_key else ""
    base = "https://fred.stlouisfed.org/graph/fredgraph.csv?series_id={sid}{key}&vintage_dates="
    results: Dict[str, float] = {}

    for sid in series_ids:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={sid}&limit=5&sort_order=desc&file_type=json{key_param}"
        )
        data = _http_get(url)
        if data and "observations" in data:
            for obs in data["observations"]:
                try:
                    val = float(obs["value"])
                    results[sid] = val
                    break
                except (ValueError, KeyError):
                    continue
    return results


def _fetch_ticker_returns(tickers: List[str], finnhub_key: Optional[str] = None) -> Dict[str, float]:
    """
    Fetch 1-day % returns for a list of tickers via Finnhub quote endpoint.
    Falls back to a lightweight approach if no Finnhub key.
    Returns {ticker: pct_change_1d}.
    """
    if not finnhub_key:
        return {}

    results: Dict[str, float] = {}
    for ticker in tickers:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={finnhub_key}"
        data = _http_get(url)
        if data and "c" in data and "pc" in data and data["pc"] and data["pc"] != 0:
            results[ticker] = round((data["c"] - data["pc"]) / data["pc"] * 100, 3)
    return results


def _fetch_fear_greed() -> Optional[float]:
    """Fetch Alternative.me Fear & Greed Index (free, no key)."""
    data = _http_get("https://api.alternative.me/fng/?limit=1&format=json")
    if data and "data" in data:
        try:
            return float(data["data"][0]["value"])
        except (KeyError, IndexError, ValueError):
            pass
    return None


def _fetch_coingecko(pro_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch BTC/ETH prices, dominance, 24h changes from CoinGecko.
    Works without a key (free tier, rate-limited).
    """
    base = "https://pro-api.coingecko.com" if pro_key else "https://api.coingecko.com"
    key_param = f"?x_cg_pro_api_key={pro_key}" if pro_key else ""
    sep = "&" if key_param else "?"

    prices_url = (
        f"{base}/api/v3/simple/price"
        f"{key_param}{sep}ids=bitcoin,ethereum"
        f"&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
    )
    global_url = f"{base}/api/v3/global{key_param}"

    prices = _http_get(prices_url) or {}
    global_data = _http_get(global_url) or {}

    result: Dict[str, Any] = {}

    btc = prices.get("bitcoin", {})
    eth = prices.get("ethereum", {})
    result["btc_price"] = btc.get("usd")
    result["btc_change_24h"] = btc.get("usd_24h_change")
    result["eth_price"] = eth.get("usd")
    result["eth_change_24h"] = eth.get("usd_24h_change")

    gdata = global_data.get("data", {})
    result["btc_dominance"] = gdata.get("market_cap_percentage", {}).get("btc")
    result["total_market_cap"] = gdata.get("total_market_cap", {}).get("usd")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SNAPSHOT BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_macro_snapshot(
    args: argparse.Namespace,
    fred_data: Dict[str, float],
    ticker_returns: Dict[str, float],
    fear_greed: Optional[float],
) -> MacroSnapshot:
    """Merge CLI args + live data into a MacroSnapshot."""

    # Start from FRED data
    snap = snapshot_from_fred(fred_data)

    # Override/extend with CLI manual args
    if args.vix is not None:
        snap.vix = args.vix
    if args.hy_spread is not None:
        snap.hy_spread_bps = args.hy_spread
    if args.oil is not None:
        snap.oil_price = args.oil
    if args.core_pce is not None:
        snap.core_pce = args.core_pce
    if args.gdp is not None:
        snap.gdp_growth = args.gdp
    if args.private_redemptions is not None:
        snap.private_credit_redemption_pct = args.private_redemptions
    if args.yield_curve is not None:
        snap.yield_curve_10_2 = args.yield_curve

    snap.war_active = args.war or False
    snap.hormuz_blocked = args.hormuz or False

    # Ticker returns from Finnhub
    snap.hyg_return_1d = ticker_returns.get("HYG")
    snap.spy_return_1d = ticker_returns.get("SPY")
    snap.kre_return_1d = ticker_returns.get("KRE")
    snap.qqq_return_1d = ticker_returns.get("QQQ")
    snap.airlines_return_1d = ticker_returns.get("JETS")
    snap.shipping_return_1d = ticker_returns.get("ZIM")

    # Sentiment
    snap.fear_greed = fear_greed

    return snap


def build_crypto_snapshot(
    cg_data: Dict[str, Any],
    fear_greed: Optional[float],
    macro_snap: MacroSnapshot,
) -> CryptoSnapshot:
    """Merge CoinGecko + fear/greed + macro context into CryptoSnapshot."""
    snap = snapshot_from_coingecko(cg_data, fear_greed=fear_greed)

    # Macro context
    snap.spx_return_1d = macro_snap.spy_return_1d
    snap.dxy_change_1d = None  # would need DXY ticker data

    # Infer global liquidity trend from signals
    if macro_snap.vix and macro_snap.hy_spread_bps:
        if macro_snap.vix < 20 and macro_snap.hy_spread_bps < 300:
            snap.global_liquidity_trend = "expanding"
        elif macro_snap.hy_spread_bps > 400:
            snap.global_liquidity_trend = "contracting"
        else:
            snap.global_liquidity_trend = "neutral"

    return snap


# ═══════════════════════════════════════════════════════════════════════════
# REPORT PRINTER
# ═══════════════════════════════════════════════════════════════════════════

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║        AAC MARKET FORECASTER — REGIME & TRADE INTELLIGENCE      ║
║        IF X + Y → EXPECT Z  |  Short + Medium Horizon           ║
╚══════════════════════════════════════════════════════════════════╝"""


def print_full_report(
    snap: MacroSnapshot,
    state: RegimeState,
    short_forecast: IndustryForecast,
    medium_forecast: IndustryForecast,
    crypto_state: Optional[CryptoRegimeState],
) -> None:
    print(BANNER)
    print(f"\n  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  War: {'YES' if snap.war_active else 'no'}  |  Hormuz: {'BLOCKED' if snap.hormuz_blocked else 'open'}")
    if snap.vix:
        print(f"  VIX: {snap.vix:.1f}", end="")
    if snap.hy_spread_bps:
        print(f"  |  HY Spread: {snap.hy_spread_bps:.0f}bps", end="")
    if snap.oil_price:
        print(f"  |  Oil: ${snap.oil_price:.1f}/bbl", end="")
    if snap.gold_price:
        print(f"  |  Gold: ${snap.gold_price:.0f}", end="")
    print()

    # ── REGIME SECTION
    print("\n" + "━" * 70)
    print("  MACRO REGIME ENGINE")
    print("━" * 70)
    print(f"  {state.summary}")
    print(f"\n  Vol Shock Readiness: {state.vol_shock_readiness:.0f}/100", end="")
    if state.vol_shock_readiness >= 80:
        print("  ⚠️  SHOCK WINDOW OPEN — convexity NOW")
    elif state.vol_shock_readiness >= 60:
        print("  ⚡ ARMED — accumulation phase, cheapest options")
    elif state.vol_shock_readiness >= 40:
        print("  🟡 ELEVATED — watch signals")
    else:
        print("  🟢 LOW")

    # Formula readout
    fired = state.top_formulas
    if fired:
        print(f"\n  Fired Formulas ({len(fired)}):")
        for f in fired[:5]:
            print(f"    [{f.tag.value}]  conf={f.confidence:.0%}  |  {f.expected_outcome[:65]}...")
            if f.conditions_met:
                print(f"      ✓ {' | '.join(f.conditions_met[:2])}")

    # ── STOCK FORECASTS
    print("\n" + "━" * 70)
    print("  SHORT-TERM RANKED TRADE STACK (0-15 days)")
    print("━" * 70)
    _print_opportunity_table(short_forecast.opportunities[:8])

    print("\n" + "━" * 70)
    print("  MEDIUM-TERM RANKED TRADE STACK (1-6 months)")
    print("━" * 70)
    _print_opportunity_table(medium_forecast.opportunities[:6])

    # 2-trade stack callout
    credit, banks = StockForecaster().two_trade_stack(state)
    print("\n" + "━" * 70)
    print("  ★  TODAY'S 2-TRADE STACK  ★")
    print("━" * 70)
    if credit:
        print(f"  TRADE 1 — ANCHOR")
        print(f"    {credit.primary_ticker} ({credit.industry.value.upper()}) | {credit.expression.value.replace('_',' ').title()}")
        print(f"    {credit.thesis}")
        print(f"    Structure: {credit.structure_hint}")
    if banks:
        print(f"\n  TRADE 2 — CONTAGION ACCELERATOR")
        print(f"    {banks.primary_ticker} ({banks.industry.value.upper()}) | {banks.expression.value.replace('_',' ').title()}")
        print(f"    {banks.thesis}")
        print(f"    Structure: {banks.structure_hint}")

    # ── CRYPTO SECTION
    if crypto_state:
        print("\n" + "━" * 70)
        print("  CRYPTO REGIME ENGINE")
        print("━" * 70)
        print(f"  Regime: {crypto_state.primary_regime.value.upper().replace('_',' ')}")
        print(f"  Bias: {crypto_state.net_bias.value.upper()}"
              f"  |  Confidence: {crypto_state.regime_confidence:.0%}"
              f"  |  L:{crypto_state.long_signals} / S:{crypto_state.short_signals}")
        for r in crypto_state.formula_results:
            if r.fired:
                icon = "🟢" if r.direction.value == "long" else "🔴"
                print(f"\n  {icon}  [{r.formula.value}]")
                print(f"     → {r.expected_outcome}")
                print(f"     Expression: {r.expression.value} | {r.timeframe_str} | Risk: {r.risk_level}")
                if r.conditions_met:
                    print(f"     ✓ {' | '.join(r.conditions_met[:2])}")

    # ── MATRIX
    print("\n" + "━" * 70)
    print(print_industry_regime_matrix())

    print("\n" + "=" * 70)
    print("  END OF FORECAST REPORT")
    print("=" * 70 + "\n")


def _print_opportunity_table(opps: List[TradeOpportunity]) -> None:
    if not opps:
        print("  No opportunities for current regime.")
        return
    print(f"  {'#':3}  {'TICKER':6}  {'INDUSTRY':15}  {'EXPRESSION':17}  {'ROI':5}  {'SPD':5}  {'RSK':5}  {'SCORE':6}")
    print("  " + "─" * 68)
    for o in opps:
        stars = "★" * min(5, round(o.composite_score / 20))
        print(
            f"  {o.rank:>2}.  {o.primary_ticker:6}  {o.industry.value.upper():15}  "
            f"{o.expression.value.replace('_',' '):17}  {o.roi_score:5.0f}  {o.speed_score:5.0f}  "
            f"{o.risk_score:5.0f}  {o.composite_score:5.0f}  {stars}"
        )
        print(f"        Thesis: {o.thesis[:65]}")
        print(f"        {o.structure_hint[:70]}")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AAC Market Forecaster — Regime + Trade Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stock-only", action="store_true", help="Skip crypto analysis")
    p.add_argument("--crypto-only", action="store_true", help="Skip stock/regime analysis")
    p.add_argument(
        "--horizon",
        choices=["short", "medium", "both"],
        default="both",
        help="Forecast horizon (default: both)",
    )
    p.add_argument("--matrix", action="store_true", help="Print industry×regime matrix reference")
    p.add_argument("--top-n", type=int, default=8, help="Number of opportunities to show")

    # Manual macro overrides
    macro = p.add_argument_group("manual macro inputs (override live API)")
    macro.add_argument("--vix", type=float, help="VIX level")
    macro.add_argument("--hy-spread", type=float, dest="hy_spread", help="HY spread in bps (e.g. 380)")
    macro.add_argument("--oil", type=float, help="WTI crude oil price")
    macro.add_argument("--gold", type=float, help="Gold price $/oz")
    macro.add_argument("--core-pce", type=float, dest="core_pce", help="Core PCE %%")
    macro.add_argument("--gdp", type=float, help="GDP growth rate %%")
    macro.add_argument("--yield-curve", type=float, dest="yield_curve", help="10Y-2Y spread %%")
    macro.add_argument("--private-redemptions", type=float, dest="private_redemptions",
                       help="Private credit redemption %% of AUM")
    macro.add_argument("--war", action="store_true", help="War active flag")
    macro.add_argument("--hormuz", action="store_true", help="Hormuz blockade flag")

    # Ticker return overrides
    ret = p.add_argument_group("manual return overrides (1d %)")
    ret.add_argument("--hyg-ret", type=float, dest="hyg_ret")
    ret.add_argument("--spy-ret", type=float, dest="spy_ret")
    ret.add_argument("--kre-ret", type=float, dest="kre_ret")
    ret.add_argument("--jets-ret", type=float, dest="jets_ret")

    return p


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.matrix:
        print(print_industry_regime_matrix())
        return

    # ── Load API keys from .env
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    fred_key = os.environ.get("FRED_API_KEY")
    finnhub_key = os.environ.get("FINNHUB_API_KEY")
    cg_key = os.environ.get("COINGECKO_PRO_API_KEY")

    print("  Fetching live data...", end="", flush=True)

    # ── Fetch FRED macro data
    FRED_SERIES = [
        "VIXCLS", "BAMLH0A0HYM2", "T10Y2Y", "DCOILWTICO",
        "GOLDAMGBD228NLBM", "T10YIE", "DGS10", "PCEPILFE",
    ]
    fred_data: Dict[str, float] = {}
    if not args.crypto_only:
        fred_data = _fred_latest(FRED_SERIES, fred_key)
        print(f" FRED({len(fred_data)})", end="", flush=True)

    # ── Fetch ticker returns
    ticker_returns: Dict[str, float] = {}
    if not args.crypto_only and finnhub_key:
        TICKERS = ["HYG", "JNK", "KRE", "SPY", "QQQ", "JETS", "ZIM", "XLF"]
        ticker_returns = _fetch_ticker_returns(TICKERS, finnhub_key)
        print(f" Finnhub({len(ticker_returns)})", end="", flush=True)

    # Apply manual return overrides
    if args.hyg_ret is not None:
        ticker_returns["HYG"] = args.hyg_ret
    if args.spy_ret is not None:
        ticker_returns["SPY"] = args.spy_ret
    if args.kre_ret is not None:
        ticker_returns["KRE"] = args.kre_ret
    if args.jets_ret is not None:
        ticker_returns["JETS"] = args.jets_ret

    # ── Fear & Greed
    fear_greed: Optional[float] = None
    fear_greed = _fetch_fear_greed()
    print(f" F&G({fear_greed})", end="", flush=True)

    # ── CoinGecko
    cg_data: Dict[str, Any] = {}
    if not args.stock_only:
        cg_data = _fetch_coingecko(cg_key)
        print(f" CG({'ok' if cg_data.get('btc_price') else 'limited'})", end="", flush=True)

    print(" done.\n")

    # ── Build snapshots
    macro_snap = build_macro_snapshot(args, fred_data, ticker_returns, fear_greed)
    crypto_snap: Optional[CryptoSnapshot] = None
    if not args.stock_only:
        crypto_snap = build_crypto_snapshot(cg_data, fear_greed, macro_snap)

    # ── Run engines
    regime_engine = RegimeEngine()
    stock_engine = StockForecaster()
    crypto_engine = CryptoForecaster()

    state: Optional[RegimeState] = None
    short_forecast: Optional[IndustryForecast] = None
    medium_forecast: Optional[IndustryForecast] = None
    crypto_state_result: Optional[CryptoRegimeState] = None

    if not args.crypto_only:
        state = regime_engine.evaluate(macro_snap)
        if args.horizon in ("short", "both"):
            short_forecast = stock_engine.forecast(state, Horizon.SHORT, top_n=args.top_n)
        if args.horizon in ("medium", "both"):
            medium_forecast = stock_engine.forecast(state, Horizon.MEDIUM, top_n=args.top_n)
        # If only one horizon requested, replicate for display
        if short_forecast is None:
            short_forecast = stock_engine.forecast(state, Horizon.SHORT, top_n=0)
        if medium_forecast is None:
            medium_forecast = stock_engine.forecast(state, Horizon.MEDIUM, top_n=0)

    if not args.stock_only and crypto_snap is not None:
        crypto_state_result = crypto_engine.evaluate(crypto_snap)

    if state is None:
        # Crypto-only mode: print crypto only
        if crypto_state_result:
            print(crypto_state_result.print_plan())
        return

    # ── Print report
    print_full_report(
        snap=macro_snap,
        state=state,
        short_forecast=short_forecast,
        medium_forecast=medium_forecast,
        crypto_state=crypto_state_result,
    )


if __name__ == "__main__":
    main()
