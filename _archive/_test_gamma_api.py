"""Polymarket API endpoint discovery — find the right search method."""
import asyncio
import json
import aiohttp

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

async def test():
    async with aiohttp.ClientSession() as s:
        kw = "tariff"

        # 1) Gamma /markets with various params
        for param in ["search", "tag", "slug_contains", "title", "question_contains"]:
            try:
                async with s.get(f"{GAMMA}/markets", params={param: kw, "limit": 3, "closed": "false"}) as r:
                    data = await r.json()
                    n = len(data) if isinstance(data, list) else 0
                    q0 = data[0].get("question","")[:60] if n > 0 and isinstance(data, list) else "-"
                    p0 = data[0].get("outcomePrices","") if n > 0 and isinstance(data, list) else "-"
                    print(f"markets {param}={kw} closed=false => {n}  q={q0}  p={p0}")
            except Exception as e:
                print(f"markets {param}={kw} => ERR: {e}")

        # 2) Gamma /events with search
        for param in ["search", "tag", "slug", "title"]:
            try:
                async with s.get(f"{GAMMA}/events", params={param: kw, "limit": 3, "closed": "false"}) as r:
                    data = await r.json()
                    n = len(data) if isinstance(data, list) else 0
                    t0 = data[0].get("title","")[:60] if n > 0 and isinstance(data, list) else "-"
                    print(f"events {param}={kw} => {n}  title={t0}")
            except Exception as e:
                print(f"events {param}={kw} => ERR: {e}")

        # 3) CLOB /markets sampling
        try:
            async with s.get(f"{CLOB}/markets", params={"limit": 5}) as r:
                data = await r.json()
                if isinstance(data, dict):
                    print(f"\nCLOB /markets keys: {list(data.keys())}")
                    markets = data.get("data", data.get("markets", []))
                    if isinstance(markets, list):
                        print(f"  count: {len(markets)}")
                        for m in markets[:2]:
                            print(f"  keys: {list(m.keys()) if isinstance(m, dict) else type(m)}")
                            if isinstance(m, dict):
                                print(f"  q: {m.get('question','')[:60]}")
                elif isinstance(data, list):
                    print(f"\nCLOB /markets: {len(data)} items")
                    if data:
                        print(f"  keys: {list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
        except Exception as e:
            print(f"CLOB /markets => ERR: {e}")

        # 4) Try Gamma search endpoint
        try:
            async with s.get(f"{GAMMA}/search", params={"query": kw, "limit": 5}) as r:
                text = await r.text()
                print(f"\nGamma /search?query={kw} => status={r.status} body={text[:200]}")
        except Exception as e:
            print(f"Gamma /search => ERR: {e}")

        # 5) Try events with tag_id
        try:
            async with s.get(f"{GAMMA}/events", params={"tag": "politics", "limit": 3, "closed": "false"}) as r:
                data = await r.json()
                n = len(data) if isinstance(data, list) else 0
                print(f"\nevents tag=politics closed=false => {n}")
                if n > 0 and isinstance(data, list):
                    for ev in data[:3]:
                        t = ev.get("title","")[:70]
                        mcount = len(ev.get("markets", []))
                        print(f"  {t}  ({mcount} mkts)")
        except Exception as e:
            print(f"events tag=politics => ERR: {e}")

asyncio.run(test())
