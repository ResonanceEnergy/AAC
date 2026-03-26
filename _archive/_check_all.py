#!/usr/bin/env python3
"""Quick status check across all 3 exchanges."""
import asyncio
import sys
import os

sys.path.insert(0, r'c:\dev\AAC_fresh')
os.chdir(r'c:\dev\AAC_fresh')
from shared.config_loader import load_env_file
load_env_file()

async def check_ndax():
    print("=" * 50)
    print("NDAX (Crypto - CAD)")
    print("=" * 50)
    try:
        from TradingExecution.exchange_connectors.ndax_connector import NDAXConnector
        conn = NDAXConnector(testnet=False)
        await conn.connect()
        bals = await conn.get_balances()
        for asset, b in bals.items():
            if b.free > 0:
                print(f"  {b.asset}: {b.free}")
        t = await conn.get_ticker("XRP/CAD")
        print(f"  XRP/CAD: bid={t.bid} ask={t.ask} spread={t.spread_pct:.3f}%")
        t2 = await conn.get_ticker("ETH/CAD")
        print(f"  ETH/CAD: bid={t2.bid} ask={t2.ask} spread={t2.spread_pct:.3f}%")
        await conn.disconnect()
        print("  Status: CONNECTED")
    except Exception as e:
        print(f"  ERROR: {e}")


async def check_moomoo():
    print()
    print("=" * 50)
    print("MOOMOO (US Stocks - Simulate)")
    print("=" * 50)
    try:
        from TradingExecution.exchange_connectors.moomoo_connector import MoomooConnector
        mc = MoomooConnector(testnet=True)
        ok = await mc.connect()
        if not ok:
            print("  NOT CONNECTED")
            return
        bals = await mc.get_balances()
        for a, b in bals.items():
            if b.free > 0:
                print(f"  {b.asset}: ${b.free:,.2f}")
        orders = await mc.get_open_orders()
        print(f"  Open orders: {len(orders)}")
        for o in orders[:10]:
            print(f"    {o.side} {o.quantity}x {o.symbol} @ ${o.price} [{o.status}]")
        await mc.disconnect()
        print("  Status: CONNECTED")
    except Exception as e:
        print(f"  ERROR: {e}")


async def check_ibkr():
    print()
    print("=" * 50)
    print("IBKR (Options/Puts - Paper)")
    print("=" * 50)
    try:
        import nest_asyncio
        nest_asyncio.apply()
        from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
        ic = IBKRConnector()
        ok = await ic.connect()
        if not ok:
            print("  NOT CONNECTED (is TWS running?)")
            return
        bals = await ic.get_balances()
        for a, b in bals.items():
            if b.free != 0 or b.locked != 0:
                print(f"  {b.asset}: free=${b.free:,.2f} locked=${b.locked:,.2f}")
        orders = await ic.get_open_orders()
        print(f"  Open orders: {len(orders)}")
        for o in orders[:10]:
            print(f"    {o.side} {o.quantity}x {o.symbol} @ ${o.price} [{o.status}]")
        positions = await ic.get_positions()
        print(f"  Positions: {len(positions)}")
        for p in positions[:5]:
            sym = p.get("symbol", "?")
            qty = p.get("quantity", 0)
            pnl = p.get("unrealized_pnl", 0)
            print(f"    {sym}: {qty} (P&L: ${pnl:,.2f})")
        await ic.disconnect()
        print("  Status: CONNECTED")
    except Exception as e:
        print(f"  ERROR: {e}")


async def main():
    print("\n  AAC TRADING DASHBOARD - March 17, 2026\n")
    await check_ndax()
    await check_moomoo()
    await check_ibkr()
    print("\n" + "=" * 50)


asyncio.run(main())
