"""Test Gamma events endpoint — looks like the best source of live markets."""
import asyncio

import aiohttp

GAMMA = "https://gamma-api.polymarket.com"

async def test():
    async with aiohttp.ClientSession() as s:
        # 1) Get more events with higher limit
        async with s.get(f"{GAMMA}/events", params={
            "closed": "false", "active": "true", "limit": "200"
        }) as r:
            events = await r.json()
            print(f"Gamma events (limit=200, active, !closed): {len(events)} events")
            total_mkts = sum(len(e.get("markets", [])) for e in events)
            print(f"  Total markets across events: {total_mkts}")
            for e in events[:10]:
                mkts = e.get("markets", [])
                print(f"  [{len(mkts)} mkts] {e.get('title','')[:70]}")

        # 2) Check if events have offset pagination
        async with s.get(f"{GAMMA}/events", params={
            "closed": "false", "active": "true", "limit": "200", "offset": "200"
        }) as r:
            events2 = await r.json()
            print(f"\nOffset=200: {len(events2)} more events")
            for e in events2[:5]:
                mkts = e.get("markets", [])
                print(f"  [{len(mkts)} mkts] {e.get('title','')[:70]}")

        # 3) Check market detail inside an event
        async with s.get(f"{GAMMA}/events", params={
            "closed": "false", "active": "true", "limit": "5"
        }) as r:
            events3 = await r.json()
            if events3:
                ev = events3[0]
                mkts = ev.get("markets", [])
                if mkts:
                    m = mkts[0]
                    print(f"\nSample market keys: {sorted(m.keys())}")
                    print(f"  question: {m.get('question','')[:80]}")
                    print(f"  condition_id: {m.get('conditionId','')[:20]}")
                    print(f"  active: {m.get('active')}")
                    print(f"  closed: {m.get('closed')}")
                    print(f"  outcomePrices: {m.get('outcomePrices')}")
                    print(f"  bestBid: {m.get('bestBid')}")
                    print(f"  bestAsk: {m.get('bestAsk')}")
                    print(f"  volume: {m.get('volume')}")
                    print(f"  clobTokenIds: {m.get('clobTokenIds')}")

        # 4) Try Gamma /markets directly with active filter
        async with s.get(f"{GAMMA}/markets", params={
            "closed": "false", "active": "true", "limit": "200"
        }) as r:
            mkts4 = await r.json()
            print(f"\nGamma /markets (active, !closed, limit=200): {len(mkts4)} markets")
            if mkts4:
                m = mkts4[0]
                print(f"  keys: {sorted(m.keys())[:15]}...")
                print(f"  q: {m.get('question','')[:70]}")
                print(f"  outcomePrices: {m.get('outcomePrices')}")
                print(f"  active: {m.get('active')}, closed: {m.get('closed')}")

asyncio.run(test())
