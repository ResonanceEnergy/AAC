"""Quick top-25 Polymarket dashboard."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import aiohttp
import aiohttp.resolver

GAMMA_API = "https://gamma-api.polymarket.com"


async def main():
    resolver = aiohttp.resolver.ThreadedResolver()
    connector = aiohttp.TCPConnector(resolver=resolver)
    async with aiohttp.ClientSession(
        connector=connector, timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        # Fetch top 25 markets by volume
        async with session.get(
            f"{GAMMA_API}/markets",
            params={
                "active": "true",
                "closed": "false",
                "order": "volume",
                "ascending": "false",
                "limit": "25",
            },
        ) as resp:
            resp.raise_for_status()
            markets = await resp.json()

        # Also fetch top 25 by 24h volume
        async with session.get(
            f"{GAMMA_API}/markets",
            params={
                "active": "true",
                "closed": "false",
                "order": "volume24hr",
                "ascending": "false",
                "limit": "25",
            },
        ) as resp2:
            if resp2.status == 200:
                hot_markets = await resp2.json()
            else:
                hot_markets = None

    def parse_price(m):
        prices = m.get("outcomePrices", "")
        if isinstance(prices, str) and prices:
            cleaned = prices.strip().strip("[]")
            parts = [p.strip().strip('"').strip("'") for p in cleaned.split(",")]
            try:
                return float(parts[0])
            except (ValueError, IndexError):
                return 0.5
        elif isinstance(prices, list) and prices:
            try:
                return float(prices[0])
            except (ValueError, TypeError):
                return 0.5
        return 0.5

    def fmt_vol(v):
        if v >= 1_000_000:
            return f"${v / 1_000_000:,.1f}M"
        elif v >= 1_000:
            return f"${v / 1_000:,.1f}K"
        else:
            return f"${v:,.0f}"

    # ─── TABLE 1: By Total Volume ─────────────────────────
    print()
    print("=" * 130)
    print("  POLYMARKET TOP 25 MARKETS — BY TOTAL VOLUME")
    print("=" * 130)
    print(f"{'#':>3}  {'YES':>7}  {'NO':>7}  {'Total Vol':>12}  {'Liq':>10}  {'Question'}")
    print("-" * 130)

    total_vol = 0
    total_liq = 0
    for i, m in enumerate(markets[:25], 1):
        yp = parse_price(m)
        np = 1.0 - yp
        vol = float(m.get("volume", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)
        q = m.get("question", "")[:75]
        total_vol += vol
        total_liq += liq

        print(f"{i:3d}  {yp*100:6.1f}c  {np*100:6.1f}c  {fmt_vol(vol):>12}  {fmt_vol(liq):>10}  {q}")

    print("-" * 130)
    print(f"  TOTALS: Vol {fmt_vol(total_vol)} | Liq {fmt_vol(total_liq)}")
    print("=" * 130)

    # ─── TABLE 2: By 24h Volume (hottest right now) ───────
    if hot_markets:
        print()
        print("=" * 130)
        print("  POLYMARKET TOP 25 MARKETS — BY 24H VOLUME (HOTTEST NOW)")
        print("=" * 130)
        print(f"{'#':>3}  {'YES':>7}  {'NO':>7}  {'Vol 24h':>12}  {'Total Vol':>12}  {'Liq':>10}  {'Question'}")
        print("-" * 130)

        for i, m in enumerate(hot_markets[:25], 1):
            yp = parse_price(m)
            np = 1.0 - yp
            vol = float(m.get("volume", 0) or 0)
            v24 = float(m.get("volume24hr", 0) or 0)
            liq = float(m.get("liquidity", 0) or 0)
            q = m.get("question", "")[:65]

            print(f"{i:3d}  {yp*100:6.1f}c  {np*100:6.1f}c  {fmt_vol(v24):>12}  {fmt_vol(vol):>12}  {fmt_vol(liq):>10}  {q}")

        print("=" * 130)

    # ─── THESIS-RELEVANT SCAN ─────────────────────────────
    thesis_keywords = [
        "iran", "oil", "gold", "dollar", "yuan", "fed", "rate cut",
        "bitcoin", "recession", "war", "nuclear", "sanctions", "inflation",
        "treasury", "debt", "credit", "brics", "saudi",
    ]
    all_mkts = markets[:25] + (hot_markets[:25] if hot_markets else [])
    seen = set()
    thesis_hits = []
    for m in all_mkts:
        cid = m.get("conditionId", "")
        if cid in seen:
            continue
        seen.add(cid)
        q_lower = m.get("question", "").lower()
        matched = [kw for kw in thesis_keywords if kw in q_lower]
        if matched:
            thesis_hits.append((m, matched))

    if thesis_hits:
        print()
        print("=" * 130)
        print("  THESIS-RELEVANT MARKETS (from top 25)")
        print("=" * 130)
        for m, kws in thesis_hits:
            yp = parse_price(m)
            vol = float(m.get("volume", 0) or 0)
            q = m.get("question", "")[:80]
            print(f"  {yp*100:6.1f}c YES | {fmt_vol(vol):>10} vol | [{', '.join(kws)}] | {q}")
        print("=" * 130)
    else:
        print()
        print("  No thesis-relevant markets in the top 25.")


if __name__ == "__main__":
    asyncio.run(main())
