#!/usr/bin/env python3
"""
NDAX Market Sell — Liquidate XRP + ETH holdings.
THIS EXECUTES REAL TRADES ON LIVE NDAX EXCHANGE.
Proceeds in CAD for withdrawal to IBKR.
"""

import asyncio
import os
import sys

sys.path.insert(0, r"c:\dev\AAC_fresh")
os.chdir(r"c:\dev\AAC_fresh")

from shared.config_loader import load_env_file

load_env_file()
from TradingExecution.exchange_connectors.ndax_connector import NDAXConnector


async def main():
    conn = NDAXConnector(testnet=False)
    await conn.connect()
    print("Connected to NDAX (LIVE)\n")

    # ── Current balances ────────────────────────────────────────────
    bals = await conn.get_balances()
    print("=== CURRENT BALANCES ===")
    for asset, b in bals.items():
        if b.free > 0 or b.locked > 0:
            print(f"  {b.asset}: free={b.free}, locked={b.locked}")

    xrp_bal = bals.get("XRP")
    eth_bal = bals.get("ETH")
    cad_bal = bals.get("CAD")

    xrp_free = xrp_bal.free if xrp_bal else 0
    eth_free = eth_bal.free if eth_bal else 0
    cad_free = cad_bal.free if cad_bal else 0

    print(f"\nXRP to sell: {xrp_free}")
    print(f"ETH to sell: {eth_free}")
    print(f"CAD before: ${cad_free:.2f}")

    # ── Current prices ──────────────────────────────────────────────
    xrp_ob = await conn.get_orderbook("XRP/CAD", limit=5)
    eth_ob = await conn.get_orderbook("ETH/CAD", limit=5)

    xrp_bid = xrp_ob.best_bid[0] if xrp_ob.bids else 0
    eth_bid = eth_ob.best_bid[0] if eth_ob.bids else 0

    est_xrp_cad = xrp_free * xrp_bid
    est_eth_cad = eth_free * eth_bid
    est_total = est_xrp_cad + est_eth_cad

    print(f"\nXRP/CAD bid: ${xrp_bid:.4f}  → ~${est_xrp_cad:.2f} CAD")
    print(f"ETH/CAD bid: ${eth_bid:.2f}  → ~${est_eth_cad:.2f} CAD")
    print(f"Estimated total: ~${est_total:.2f} CAD")
    print()

    results = []

    # ── Sell XRP ────────────────────────────────────────────────────
    if xrp_free > 0:
        print(f"SELLING {xrp_free} XRP/CAD at MARKET...")
        try:
            order = await conn.create_order(
                symbol="XRP/CAD",
                side="sell",
                order_type="market",
                quantity=xrp_free,
            )
            print(f"  XRP SELL ORDER: id={order.order_id}, status={order.status}, "
                  f"filled={order.filled_quantity}, avg_price={order.average_price}")
            results.append(("XRP", order))
        except Exception as e:
            print(f"  XRP SELL FAILED: {e}")
    else:
        print("No XRP to sell.")

    # ── Sell ETH ────────────────────────────────────────────────────
    if eth_free > 0:
        print(f"SELLING {eth_free} ETH/CAD at MARKET...")
        try:
            order = await conn.create_order(
                symbol="ETH/CAD",
                side="sell",
                order_type="market",
                quantity=eth_free,
            )
            print(f"  ETH SELL ORDER: id={order.order_id}, status={order.status}, "
                  f"filled={order.filled_quantity}, avg_price={order.average_price}")
            results.append(("ETH", order))
        except Exception as e:
            print(f"  ETH SELL FAILED: {e}")
    else:
        print("No ETH to sell.")

    # ── Post-trade balances ─────────────────────────────────────────
    await asyncio.sleep(2)
    new_bals = await conn.get_balances()
    print("\n=== POST-TRADE BALANCES ===")
    for asset, b in new_bals.items():
        if b.free > 0 or b.locked > 0:
            print(f"  {b.asset}: free={b.free}, locked={b.locked}")

    new_cad = new_bals.get("CAD")
    new_cad_free = new_cad.free if new_cad else 0
    proceeds = new_cad_free - cad_free
    print(f"\nCAD proceeds: ${proceeds:.2f}")
    print(f"CAD total:    ${new_cad_free:.2f}")
    print(f"USD equiv:    ~${new_cad_free * 0.70:.2f} (at 0.70)")

    print("\n=== SUMMARY ===")
    for asset, order in results:
        print(f"  {asset}: order_id={order.order_id}, status={order.status}, "
              f"filled={order.filled_quantity}")
    print(f"\nNEXT: Withdraw ${new_cad_free:.2f} CAD via Interac e-Transfer or wire to bank.")
    print("THEN: Deposit to IBKR via EFT or wire.")

    await conn.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
