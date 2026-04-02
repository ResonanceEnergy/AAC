"""Quick test of CLOB API filter params."""
import asyncio

import aiohttp

CLOB = "https://clob.polymarket.com"

async def test():
    async with aiohttp.ClientSession() as s:
        # Test with active=true filter
        async with s.get(f"{CLOB}/markets", params={"limit": "10", "active": "true"}) as r:
            d = await r.json()
            print(f"active=true: count={d.get('count')}, cursor={d.get('next_cursor','')[:12]}")
            for m in d.get("data", [])[:3]:
                print(f"  a={m.get('active')} cl={m.get('closed')} acc={m.get('accepting_orders')} q={m.get('question','')[:70]}")

        # Test with closed=false
        async with s.get(f"{CLOB}/markets", params={"limit": "10", "closed": "false"}) as r:
            d = await r.json()
            print(f"\nclosed=false: count={d.get('count')}")
            for m in d.get("data", [])[:3]:
                print(f"  a={m.get('active')} cl={m.get('closed')} acc={m.get('accepting_orders')} q={m.get('question','')[:70]}")

        # Test active=true + closed=false + accepting_orders=true
        async with s.get(f"{CLOB}/markets", params={"limit": "500", "active": "true", "closed": "false", "accepting_orders": "true"}) as r:
            d = await r.json()
            n = d.get("count", 0)
            cursor = d.get("next_cursor", "")
            batch = d.get("data", [])
            print(f"\nactive+!closed+accepting: count={n}, batch_size={len(batch)}, cursor={cursor[:12]}")
            for m in batch[:5]:
                print(f"  a={m.get('active')} cl={m.get('closed')} acc={m.get('accepting_orders')} q={m.get('question','')[:70]}")

        # Another approach: try getting from Gamma events endpoint with proper search
        async with s.get(f"https://gamma-api.polymarket.com/events", params={"closed": "false", "limit": "50", "active": "true"}) as r:
            d = await r.json()
            print(f"\nGamma events active+!closed: {len(d)} events")
            for ev in d[:5]:
                mkts = ev.get("markets", [])
                print(f"  title={ev.get('title','')[:60]}  mkts={len(mkts)}")

asyncio.run(test())
