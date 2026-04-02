#!/usr/bin/env python3
"""Quick test: all 15 indicators with live data."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv; load_dotenv()

from strategies.war_room_engine import IndicatorState, compute_composite_score
from strategies.war_room_live_feeds import apply_live_data_to_indicators, fetch_all_live_data


async def main():
    r = await fetch_all_live_data()
    print("=== FEED RESULTS ===")
    print(f"  SPY:        {r.spy_price}")
    print(f"  Gold:       {r.gold_price_oz}")
    print(f"  Fed Rate:   {r.fed_rate}")
    print(f"  DXY:        {r.dxy_index}")
    print(f"  HY Spread:  {r.hy_spread_bp_live}")
    print(f"  Stablecoin: {r.stablecoin_depeg_pct}")
    print(f"  FGI:        {r.fear_greed_value}")
    print(f"  News Score: {r.news_severity_score} ({r.news_headline_count} articles)")
    print(f"  X Sent:     {r.x_sentiment_score}")
    print(f"  BTC:        {r.btc_price}")
    print(f"  ETH:        {r.eth_price}")
    print(f"  Errors:     {r.errors}")

    ind = apply_live_data_to_indicators(r)
    print()
    print("=== INDICATOR STATE (after live patch) ===")
    fields = [
        "oil_price", "gold_price", "vix", "hy_spread_bp",
        "bdc_nav_discount", "bdc_nonaccrual_pct", "defi_tvl_change_pct",
        "stablecoin_depeg_pct", "btc_price", "fed_funds_rate", "dxy",
        "spy_price", "x_sentiment", "news_severity", "fear_greed_index",
    ]
    for f in fields:
        val = getattr(ind, f)
        print(f"  {f:25s} = {val}")

    print()
    result = compute_composite_score(ind)
    composite = result["composite_score"]
    regime = result["regime"]
    print(f"COMPOSITE: {composite}/100 [{regime}]")
    print()
    for k, v in sorted(result["individual_scores"].items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v:6.1f}")


asyncio.run(main())
