"""Count total active Gamma markets with pagination."""
import asyncio
import aiohttp

async def count():
    total = 0
    offset = 0
    async with aiohttp.ClientSession() as s:
        while True:
            params = {
                "closed": "false",
                "active": "true",
                "limit": "200",
                "offset": str(offset),
            }
            async with s.get(
                "https://gamma-api.polymarket.com/markets", params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                mkts = await r.json()

            if not mkts:
                break
            total += len(mkts)
            print(f"  offset={offset}: got {len(mkts)}, total={total}")
            offset += 200
            if len(mkts) < 200:
                break
            await asyncio.sleep(0.1)
    print(f"TOTAL active non-closed markets: {total}")

asyncio.run(count())
