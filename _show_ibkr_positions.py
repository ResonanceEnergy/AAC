import asyncio
import os
import sys

sys.path.insert(0, ".")
from dotenv import load_dotenv

load_dotenv()


async def main():
    from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector

    port = 7497  # TWS is listening on 7497 per netstat
    account = os.getenv("IBKR_ACCOUNT", "U24346218")
    conn = IBKRConnector(
        host=os.getenv("IBKR_HOST", "127.0.0.1"),
        port=port,
        client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
        account=account,
    )
    print(f"Connecting to IBKR port {port} (account {account}) ...")
    ok = await conn.connect()
    if not ok:
        print("ERROR: Could not connect to IBKR TWS/Gateway.")
        return

    print("Connected.\n")

    # Positions — use raw ib_insync positions to avoid connector bug
    conn._ensure_connected()
    raw_positions = conn._conn.positions(conn.account)
    if not raw_positions:
        print("No open positions found.")
    else:
        print(f"{'SYMBOL':<12} {'TYPE':<6} {'QTY':>8} {'AVG COST':>12} {'CURRENCY':<6}")
        print("-" * 52)
        for pos in raw_positions:
            print(
                f"{pos.contract.symbol:<12} {pos.contract.secType:<6} {pos.position:>8.2f}"
                f" {pos.avgCost:>12.4f} {pos.contract.currency:<6}"
            )

    # Account summary
    print()
    print("--- ACCOUNT SUMMARY ---")
    try:
        summary = await conn.get_account_summary()
        for k, v in sorted(summary.items()):
            print(f"  {k:<35} {v:>12.2f}")
    except Exception as e:
        print(f"  (summary failed: {e})")

    await conn.disconnect()


asyncio.run(main())
