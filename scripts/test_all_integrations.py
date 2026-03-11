#!/usr/bin/env python3
"""Test all 5 integrations: IBKR, Moomoo, NDAX, CoinGecko, EQ Bank"""
import sys, asyncio, os, logging, socket
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING)

W = 70


def port_open(host, port, timeout=2):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


async def main():
    print("=" * W)
    print("  AAC — LIVE READINESS TEST: ALL 5 INTEGRATIONS")
    print("=" * W)

    # ── 1. IBKR ──────────────────────────────────────────────────────
    print("\n[1/5] INTERACTIVE BROKERS (IBKR)")
    print("-" * 40)
    try:
        from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
        from shared.config_loader import get_config
        cfg = get_config()
        acct = cfg.ibkr.account if cfg.ibkr.account else "(not set)"
        print(f"  Module import:  OK")
        print(f"  Host:           {cfg.ibkr.host}:{cfg.ibkr.port}")
        print(f"  Account:        {acct}")
        print(f"  Paper mode:     {cfg.ibkr.paper}")
        print(f"  Enabled:        {cfg.ibkr.enabled}")
        conn = IBKRConnector()
        print(f"  Connector name: {conn.name}")

        if port_open(cfg.ibkr.host, cfg.ibkr.port):
            try:
                result = await conn.connect()
                if result:
                    print("  Connection:     LIVE")
                    balances = await conn.get_balances()
                    print(f"  Balances:       {len(balances)} assets")
                    await conn.disconnect()
                else:
                    print("  Connection:     FAILED")
            except asyncio.TimeoutError:
                print("  Connection:     TIMEOUT")
            except Exception as e:
                print(f"  Connection:     OFFLINE - {str(e)[:80]}")
        else:
            print(f"  Gateway:        NOT RUNNING on {cfg.ibkr.host}:{cfg.ibkr.port}")
            print("  Action needed:  Start TWS or IB Gateway, then retest")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── 2. MOOMOO ────────────────────────────────────────────────────
    print("\n[2/5] MOOMOO / FUTU")
    print("-" * 40)
    try:
        from TradingExecution.exchange_connectors.moomoo_connector import MoomooConnector, MOOMOO_AVAILABLE
        cfg = get_config()
        print(f"  Module import:  OK")
        print(f"  SDK installed:  {MOOMOO_AVAILABLE}")
        print(f"  Host:           {cfg.moomoo.host}:{cfg.moomoo.port}")
        print(f"  Market:         {cfg.moomoo.market}")
        print(f"  Trade env:      {cfg.moomoo.trade_env}")
        print(f"  Enabled:        {cfg.moomoo.enabled}")
        conn = MoomooConnector()
        print(f"  Connector name: {conn.name}")

        if port_open(cfg.moomoo.host, cfg.moomoo.port):
            try:
                result = await conn.connect()
                if result:
                    print("  Connection:     LIVE")
                    ticker = await conn.get_ticker("AAPL/USD")
                    print(f"  AAPL last:      ${ticker.last:.2f}")
                    await conn.disconnect()
                else:
                    print("  Connection:     FAILED")
            except asyncio.TimeoutError:
                print("  Connection:     TIMEOUT")
            except Exception as e:
                print(f"  Connection:     OFFLINE - {str(e)[:80]}")
        else:
            print(f"  Gateway:        NOT RUNNING on {cfg.moomoo.host}:{cfg.moomoo.port}")
            print("  Action needed:  Install & start Moomoo OpenD, then retest")
            print("  Download:       https://www.moomoo.com/download/OpenD")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── 3. NDAX ──────────────────────────────────────────────────────
    print("\n[3/5] NDAX (Canadian Crypto Exchange)")
    print("-" * 40)
    try:
        from TradingExecution.exchange_connectors.ndax_connector import NDAXConnector, CCXT_AVAILABLE
        import ccxt
        print(f"  Module import:  OK")
        print(f"  ccxt installed: {CCXT_AVAILABLE}")
        print(f"  ccxt version:   {ccxt.__version__}")
        print(f"  ndax in ccxt:   {'ndax' in ccxt.exchanges}")

        conn = NDAXConnector(testnet=False)
        print(f"  Connector name: {conn.name}")

        try:
            result = await conn.connect()
            if result:
                print("  Connection:     LIVE")
                try:
                    cad_pairs = await conn.get_available_cad_pairs()
                    print(f"  CAD pairs:      {len(cad_pairs)} available")
                    if cad_pairs:
                        print(f"  Examples:       {cad_pairs[:5]}")
                except Exception:
                    print("  CAD pairs:      N/A")
                try:
                    ticker = await conn.get_ticker("BTC/CAD")
                    print(f"  BTC/CAD:        ${ticker.last:,.2f} CAD")
                except Exception:
                    try:
                        ticker = await conn.get_ticker("BTC/USDT")
                        print(f"  BTC/USDT:       ${ticker.last:,.2f}")
                    except Exception:
                        print("  Ticker test:    Skipped")
                await conn.disconnect()
            else:
                print("  Connection:     FAILED")
        except asyncio.TimeoutError:
            print("  Connection:     TIMEOUT")
        except Exception as e:
            print(f"  Connection:     OFFLINE - {str(e)[:80]}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── 4. COINGECKO ─────────────────────────────────────────────────
    print("\n[4/5] COINGECKO (Crypto Market Data)")
    print("-" * 40)
    try:
        from shared.data_sources import CoinGeckoClient
        print("  Module import:  OK")
        client = CoinGeckoClient()
        print("  Client created: OK")

        try:
            tick = await client.get_price("bitcoin")
            if tick and tick.price > 0:
                print("  Connection:     LIVE")
                print(f"  BTC price:      ${tick.price:,.2f} USD")
            else:
                print(f"  Connection:     OK (tick={tick})")

            eth_tick = await client.get_price("ethereum")
            if eth_tick:
                print(f"  ETH price:      ${eth_tick.price:,.2f} USD")

            if asyncio.iscoroutinefunction(getattr(client, 'get_trending', None)):
                trending = await client.get_trending()
                if trending:
                    names = [t.get("item", {}).get("name", "?") if isinstance(t, dict) else str(t) for t in trending[:3]]
                    print(f"  Trending:       {names}")
            await client.disconnect()
        except Exception as e:
            print(f"  Connection:     OFFLINE - {str(e)[:80]}")
            try:
                await client.disconnect()
            except Exception:
                pass
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── 5. EQ BANK ───────────────────────────────────────────────────
    print("\n[5/5] EQ BANK FUNDING MANAGER")
    print("-" * 40)
    try:
        from integrations.eq_bank_funding import EQBankFundingManager
        print("  Module import:  OK")

        mgr = EQBankFundingManager()
        print(f"  Enabled:        {mgr.enabled}")
        print(f"  Account:        {mgr.account_label}")
        print(f"  Linked brokers: {mgr.linked_brokers}")

        mgr.register_broker_account("ibkr", "DU12345", "IBKR Paper Account", "USD")
        mgr.register_broker_account("moomoo", "MOOMOO-001", "Moomoo US Equities", "USD")
        mgr.register_broker_account("ndax", "NDAX-001", "NDAX Crypto CAD", "CAD")
        print("  Registered:     3 broker accounts")

        t1 = mgr.record_transfer("ibkr", 5000.00, "CAD", "eft", notes="Initial IBKR funding")
        t2 = mgr.record_transfer("ndax", 2000.00, "CAD", "eft", notes="NDAX crypto funding")
        t3 = mgr.record_transfer("moomoo", 3000.00, "CAD", "eft", notes="Moomoo equities funding")
        print("  Test transfers: 3 recorded")

        mgr.update_transfer_status(t1.transfer_id, "completed")
        print(f"  Status update:  {t1.transfer_id} -> completed")

        summary = mgr.get_funding_summary()
        print(f"  Total funded:   ${summary['total_funded']:,.2f} (completed)")
        print(f"  Pending:        ${summary['pending_transfers']:,.2f}")
        print(f"  By broker:      {summary['by_broker']}")
        print("  Status:         OPERATIONAL")
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # ── EXECUTION ENGINE REGISTRY ────────────────────────────────────
    print()
    print("=" * W)
    print("  EXECUTION ENGINE CONNECTOR REGISTRY")
    print("=" * W)
    try:
        from TradingExecution.execution_engine import ExecutionEngine
        cfg = get_config()
        exchanges = ["binance", "coinbase", "kraken", "ibkr", "moomoo", "ndax"]
        print(f"  Registered exchanges: {len(exchanges)}")
        for ex in exchanges:
            try:
                engine = ExecutionEngine.__new__(ExecutionEngine)
                engine._connectors = {}
                engine.config = cfg
                engine.logger = logging.getLogger("test")
                c = engine._get_connector(ex)
                print(f"    {ex:12s} -> {type(c).__name__:20s} OK")
            except Exception as e:
                print(f"    {ex:12s} -> FAILED: {str(e)[:50]}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()
    print("=" * W)
    print("  ALL 5 INTEGRATION TESTS COMPLETE")
    print("=" * W)


if __name__ == "__main__":
    asyncio.run(main())
