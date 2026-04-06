#!/usr/bin/env python3
"""Test all 5 integrations: IBKR, Moomoo, NDAX, CoinGecko, EQ Bank"""
import asyncio
import logging
import os
import socket
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING)

W = 70


def port_open(host, port, timeout=2):
    """Port open."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


async def main():
    """Main."""
    logger.info("=" * W)
    logger.info("  AAC — LIVE READINESS TEST: ALL 5 INTEGRATIONS")
    logger.info("=" * W)

    # ── 1. IBKR ──────────────────────────────────────────────────────
    logger.info("\n[1/5] INTERACTIVE BROKERS (IBKR)")
    logger.info("-" * 40)
    try:
        from shared.config_loader import get_config
        from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
        cfg = get_config()
        acct = cfg.ibkr.account if cfg.ibkr.account else "(not set)"
        logger.info(f"  Module import:  OK")
        logger.info(f"  Host:           {cfg.ibkr.host}:{cfg.ibkr.port}")
        logger.info(f"  Account:        {acct}")
        logger.info(f"  Paper mode:     {cfg.ibkr.paper}")
        logger.info(f"  Enabled:        {cfg.ibkr.enabled}")
        conn = IBKRConnector()
        logger.info(f"  Connector name: {conn.name}")

        if port_open(cfg.ibkr.host, cfg.ibkr.port):
            try:
                result = await conn.connect()
                if result:
                    logger.info("  Connection:     LIVE")
                    balances = await conn.get_balances()
                    logger.info(f"  Balances:       {len(balances)} assets")
                    await conn.disconnect()
                else:
                    logger.info("  Connection:     FAILED")
            except asyncio.TimeoutError:
                logger.info("  Connection:     TIMEOUT")
            except Exception as e:
                logger.info(f"  Connection:     OFFLINE - {str(e)[:80]}")
        else:
            logger.info(f"  Gateway:        NOT RUNNING on {cfg.ibkr.host}:{cfg.ibkr.port}")
            logger.info("  Action needed:  Start TWS or IB Gateway, then retest")
    except Exception as e:
        logger.info(f"  ERROR: {e}")

    # ── 2. MOOMOO ────────────────────────────────────────────────────
    logger.info("\n[2/5] MOOMOO / FUTU")
    logger.info("-" * 40)
    try:
        from TradingExecution.exchange_connectors.moomoo_connector import (
            MOOMOO_AVAILABLE,
            MoomooConnector,
        )
        cfg = get_config()
        logger.info(f"  Module import:  OK")
        logger.info(f"  SDK installed:  {MOOMOO_AVAILABLE}")
        logger.info(f"  Host:           {cfg.moomoo.host}:{cfg.moomoo.port}")
        logger.info(f"  Market:         {cfg.moomoo.market}")
        logger.info(f"  Trade env:      {cfg.moomoo.trade_env}")
        logger.info(f"  Enabled:        {cfg.moomoo.enabled}")
        conn = MoomooConnector()
        logger.info(f"  Connector name: {conn.name}")

        if port_open(cfg.moomoo.host, cfg.moomoo.port):
            try:
                result = await conn.connect()
                if result:
                    logger.info("  Connection:     LIVE")
                    ticker = await conn.get_ticker("AAPL/USD")
                    logger.info(f"  AAPL last:      ${ticker.last:.2f}")
                    await conn.disconnect()
                else:
                    logger.info("  Connection:     FAILED")
            except asyncio.TimeoutError:
                logger.info("  Connection:     TIMEOUT")
            except Exception as e:
                logger.info(f"  Connection:     OFFLINE - {str(e)[:80]}")
        else:
            logger.info(f"  Gateway:        NOT RUNNING on {cfg.moomoo.host}:{cfg.moomoo.port}")
            logger.info("  Action needed:  Install & start Moomoo OpenD, then retest")
            logger.info("  Download:       https://www.moomoo.com/download/OpenD")
    except Exception as e:
        logger.info(f"  ERROR: {e}")

    # ── 3. NDAX ──────────────────────────────────────────────────────
    logger.info("\n[3/5] NDAX (Canadian Crypto Exchange)")
    logger.info("-" * 40)
    try:
        import ccxt

        from TradingExecution.exchange_connectors.ndax_connector import (
            CCXT_AVAILABLE,
            NDAXConnector,
        )
        logger.info(f"  Module import:  OK")
        logger.info(f"  ccxt installed: {CCXT_AVAILABLE}")
        logger.info(f"  ccxt version:   {ccxt.__version__}")
        logger.info(f"  ndax in ccxt:   {'ndax' in ccxt.exchanges}")

        conn = NDAXConnector(testnet=False)
        logger.info(f"  Connector name: {conn.name}")

        try:
            result = await conn.connect()
            if result:
                logger.info("  Connection:     LIVE")
                try:
                    cad_pairs = await conn.get_available_cad_pairs()
                    logger.info(f"  CAD pairs:      {len(cad_pairs)} available")
                    if cad_pairs:
                        logger.info(f"  Examples:       {cad_pairs[:5]}")
                except Exception:
                    logger.info("  CAD pairs:      N/A")
                try:
                    ticker = await conn.get_ticker("BTC/CAD")
                    logger.info(f"  BTC/CAD:        ${ticker.last:,.2f} CAD")
                except Exception:
                    try:
                        ticker = await conn.get_ticker("BTC/USDT")
                        logger.info(f"  BTC/USDT:       ${ticker.last:,.2f}")
                    except Exception:
                        logger.info("  Ticker test:    Skipped")
                await conn.disconnect()
            else:
                logger.info("  Connection:     FAILED")
        except asyncio.TimeoutError:
            logger.info("  Connection:     TIMEOUT")
        except Exception as e:
            logger.info(f"  Connection:     OFFLINE - {str(e)[:80]}")
    except Exception as e:
        logger.info(f"  ERROR: {e}")

    # ── 4. COINGECKO ─────────────────────────────────────────────────
    logger.info("\n[4/5] COINGECKO (Crypto Market Data)")
    logger.info("-" * 40)
    try:
        from shared.data_sources import CoinGeckoClient
        logger.info("  Module import:  OK")
        client = CoinGeckoClient()
        logger.info("  Client created: OK")

        try:
            tick = await client.get_price("bitcoin")
            if tick and tick.price > 0:
                logger.info("  Connection:     LIVE")
                logger.info(f"  BTC price:      ${tick.price:,.2f} USD")
            else:
                logger.info(f"  Connection:     OK (tick={tick})")

            eth_tick = await client.get_price("ethereum")
            if eth_tick:
                logger.info(f"  ETH price:      ${eth_tick.price:,.2f} USD")

            if asyncio.iscoroutinefunction(getattr(client, 'get_trending', None)):
                trending = await client.get_trending()
                if trending:
                    names = [t.get("item", {}).get("name", "?") if isinstance(t, dict) else str(t) for t in trending[:3]]
                    logger.info(f"  Trending:       {names}")
            await client.disconnect()
        except Exception as e:
            logger.info(f"  Connection:     OFFLINE - {str(e)[:80]}")
            try:
                await client.disconnect()
            except Exception as e:
                logger.exception("Unexpected error: %s", e)
    except Exception as e:
        logger.info(f"  ERROR: {e}")

    # ── 5. EQ BANK ───────────────────────────────────────────────────
    logger.info("\n[5/5] EQ BANK FUNDING MANAGER")
    logger.info("-" * 40)
    try:
        from integrations.eq_bank_funding import EQBankFundingManager
        logger.info("  Module import:  OK")

        mgr = EQBankFundingManager()
        logger.info(f"  Enabled:        {mgr.enabled}")
        logger.info(f"  Account:        {mgr.account_label}")
        logger.info(f"  Linked brokers: {mgr.linked_brokers}")

        mgr.register_broker_account("ibkr", "DU12345", "IBKR Paper Account", "USD")
        mgr.register_broker_account("moomoo", "MOOMOO-001", "Moomoo US Equities", "USD")
        mgr.register_broker_account("ndax", "NDAX-001", "NDAX Crypto CAD", "CAD")
        logger.info("  Registered:     3 broker accounts")

        t1 = mgr.record_transfer("ibkr", 5000.00, "CAD", "eft", notes="Initial IBKR funding")
        t2 = mgr.record_transfer("ndax", 2000.00, "CAD", "eft", notes="NDAX crypto funding")
        t3 = mgr.record_transfer("moomoo", 3000.00, "CAD", "eft", notes="Moomoo equities funding")
        logger.info("  Test transfers: 3 recorded")

        mgr.update_transfer_status(t1.transfer_id, "completed")
        logger.info(f"  Status update:  {t1.transfer_id} -> completed")

        summary = mgr.get_funding_summary()
        logger.info(f"  Total funded:   ${summary['total_funded']:,.2f} (completed)")
        logger.info(f"  Pending:        ${summary['pending_transfers']:,.2f}")
        logger.info(f"  By broker:      {summary['by_broker']}")
        logger.info("  Status:         OPERATIONAL")
    except Exception as e:
        import traceback
        logger.info(f"  ERROR: {e}")
        traceback.print_exc()

    # ── EXECUTION ENGINE REGISTRY ────────────────────────────────────
    logger.info("")
    logger.info("=" * W)
    logger.info("  EXECUTION ENGINE CONNECTOR REGISTRY")
    logger.info("=" * W)
    try:
        from TradingExecution.execution_engine import ExecutionEngine
        cfg = get_config()
        exchanges = ["binance", "coinbase", "kraken", "ibkr", "moomoo", "ndax"]
        logger.info(f"  Registered exchanges: {len(exchanges)}")
        for ex in exchanges:
            try:
                engine = ExecutionEngine.__new__(ExecutionEngine)
                engine._connectors = {}
                engine.config = cfg
                engine.logger = logging.getLogger("test")
                c = engine._get_connector(ex)
                logger.info(f"    {ex:12s} -> {type(c).__name__:20s} OK")
            except Exception as e:
                logger.info(f"    {ex:12s} -> FAILED: {str(e)[:50]}")
    except Exception as e:
        logger.info(f"  ERROR: {e}")

    logger.info("")
    logger.info("=" * W)
    logger.info("  ALL 5 INTEGRATION TESTS COMPLETE")
    logger.info("=" * W)


if __name__ == "__main__":
    asyncio.run(main())
