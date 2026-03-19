#!/usr/bin/env python3
"""Scout NDAX orderbook for market-making opportunity."""
import asyncio
import sys
import os

sys.path.insert(0, r"c:\dev\AAC_fresh")
os.chdir(r"c:\dev\AAC_fresh")
from shared.config_loader import load_env_file
load_env_file()
from TradingExecution.exchange_connectors.ndax_connector import NDAXConnector


async def scout():
    conn = NDAXConnector(testnet=False)
    await conn.connect()

    # XRP/CAD orderbook depth
    ob = await conn.get_orderbook("XRP/CAD", limit=10)
    print("=== XRP/CAD ORDERBOOK ===")
    print("ASKS (sellers):")
    for p, q in ob.asks[:5]:
        print(f"  {p:.4f} CAD x {q:.2f} XRP (= ${p*q:.2f} CAD)")
    print("BIDS (buyers):")
    for p, q in ob.bids[:5]:
        print(f"  {p:.4f} CAD x {q:.2f} XRP (= ${p*q:.2f} CAD)")
    mid = ob.mid_price
    spread_bps = ((ob.best_ask[0] - ob.best_bid[0]) / mid) * 10000
    print(f"Mid: {mid:.4f} | Spread: {spread_bps:.1f} bps")
    print()

    # ETH/CAD orderbook
    ob2 = await conn.get_orderbook("ETH/CAD", limit=5)
    print("=== ETH/CAD ORDERBOOK ===")
    print(f"Best bid: {ob2.best_bid[0]:.2f} x {ob2.best_bid[1]:.6f}")
    print(f"Best ask: {ob2.best_ask[0]:.2f} x {ob2.best_ask[1]:.6f}")
    mid2 = ob2.mid_price
    spread2 = ((ob2.best_ask[0] - ob2.best_bid[0]) / mid2) * 10000
    print(f"Mid: {mid2:.2f} | Spread: {spread2:.1f} bps")
    print()

    # Our balances
    bals = await conn.get_balances()
    print("=== OUR INVENTORY ===")
    for a, b in bals.items():
        if b.free > 0:
            print(f"  {b.asset}: {b.free}")
    print()

    # Fees
    fee = await conn.get_trade_fee("XRP/CAD")
    maker = fee["maker"] * 100
    taker = fee["taker"] * 100
    print(f"XRP/CAD fees: maker={maker}% taker={taker}%")

    # Profit calc: sell XRP at ask, buy back at bid
    sell_price = ob.best_bid[0]  # we'd hit the bid
    buy_price = ob.best_ask[0]   # we'd hit the ask
    # As maker: place limit sell above mid, limit buy below mid
    our_sell = mid + (ob.best_ask[0] - mid) * 0.5  # halfway between mid and ask
    our_buy = mid - (mid - ob.best_bid[0]) * 0.5   # halfway between bid and mid
    gross_spread_bps = ((our_sell - our_buy) / mid) * 10000
    net_spread_bps = gross_spread_bps - (fee["maker"] * 10000 * 2)  # maker on both sides
    print(f"\nMARKET-MAKING PLAN:")
    print(f"  Limit SELL XRP at {our_sell:.4f} CAD")
    print(f"  Limit BUY  XRP at {our_buy:.4f} CAD")
    print(f"  Gross spread: {gross_spread_bps:.1f} bps")
    print(f"  Net after fees: {net_spread_bps:.1f} bps")
    print(f"  Per 500 XRP round-trip: ${500 * mid * net_spread_bps / 10000:.2f} CAD profit")
    print(f"  We have 2070 XRP = can run 4 x 500 XRP lots")

    await conn.disconnect()


asyncio.run(scout())
