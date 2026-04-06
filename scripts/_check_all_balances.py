#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
AAC UNIFIED BALANCE SCANNER — All Accounts
═══════════════════════════════════════════════════════════════════════════
Scans: IBKR, Moomoo, NDAX, Polymarket, MetaMask/Polygon, EQ Bank, Wealthsimple

Usage:
    python _check_all_balances.py              # scan all accounts
    python _check_all_balances.py --json       # JSON output
    python _check_all_balances.py --update     # update doctrine after scan
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

# ─── Constants ────────────────────────────────────────────────────────────
POLYGON_RPCS = [
    "https://polygon-rpc.com",
    "https://rpc.ankr.com/polygon",
]
ETH_RPCS = [
    "https://eth.llamarpc.com",
    "https://rpc.ankr.com/eth",
]
USDC_POLYGON = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e on Polygon
USDC_ETH = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"     # USDC on Ethereum
ERC20_BALANCE_ABI = [{"constant": True, "inputs": [{"name": "_owner", "type": "address"}],
                      "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}],
                      "type": "function"}]

CAD_USD_RATE = 0.72  # approximate, used for NDAX CAD → USD normalization


def _header(name: str) -> str:
    return f"\n{'═' * 60}\n  {name}\n{'═' * 60}"


def _connect_ibkr_sync():
    """Connect to the first reachable IBKR endpoint and return session context."""
    from ib_insync import IB

    host = os.getenv("IBKR_HOST", "127.0.0.1")
    env_port = int(os.getenv("IBKR_PORT", "7496"))
    ports = []
    for port in (env_port, 7496, 7497):
        if port not in ports:
            ports.append(port)

    last_error = None
    for port in ports:
        ib = IB()
        try:
            ib.connect(host=host, port=port, clientId=99, timeout=10)
            if not ib.isConnected():
                raise RuntimeError(f"connect returned disconnected on port {port}")

            accounts = ib.managedAccounts()
            if not accounts:
                raise RuntimeError(f"no managed accounts on port {port}")

            preferred = os.getenv("IBKR_ACCOUNT", "")
            acct = preferred if preferred in accounts else next(
                (account for account in accounts if account.startswith("U")),
                accounts[0],
            )
            return ib, acct, port
        except Exception as exc:
            last_error = exc
            try:
                ib.disconnect()
            except Exception:
                pass

    raise RuntimeError(str(last_error) if last_error else "Could not connect to TWS/Gateway")


# ═══════════════════════════════════════════════════════════════════════════
# 1. IBKR — Interactive Brokers (ib_insync)
# ═══════════════════════════════════════════════════════════════════════════
async def scan_ibkr() -> dict:
    """Connect to TWS/Gateway, pull account summary + positions."""
    result = {"platform": "IBKR", "status": "error", "balances": {}, "positions": [], "net_liquidation": 0}
    try:
        ib, acct, port = await asyncio.to_thread(_connect_ibkr_sync)
        result["account"] = acct
        result["account_id"] = acct
        result["port"] = port

        summary = await asyncio.to_thread(ib.accountSummary, acct)
        base_currency = "USD"
        for item in summary:
            if item.tag == "NetLiquidation":
                base_currency = item.currency or base_currency
                result["currency"] = base_currency
                result["net_liquidation"] = float(item.value)
            elif item.tag in ("TotalCashValue", "CashBalance"):
                if item.currency == base_currency or item.currency == "BASE":
                    result["balances"]["TOTAL_CASH"] = float(item.value)
                else:
                    amt = float(item.value)
                    if abs(amt) > 0.01:
                        result["balances"][item.currency] = amt
            elif item.tag == "AvailableFunds":
                if item.currency == base_currency or item.currency == "BASE":
                    result["balances"]["AVAILABLE_FUNDS"] = float(item.value)
            elif item.tag == "BuyingPower":
                if item.currency == base_currency or item.currency == "BASE":
                    result["balances"]["BUYING_POWER"] = float(item.value)
            elif item.tag == "UnrealizedPnL" and (item.currency == base_currency or item.currency == "BASE"):
                result["balances"]["UNREALIZED_PNL"] = float(item.value)
            elif item.tag == "RealizedPnL" and (item.currency == base_currency or item.currency == "BASE"):
                result["balances"]["REALIZED_PNL"] = float(item.value)

        if "currency" not in result:
            result["currency"] = base_currency

        portfolio = await asyncio.to_thread(ib.portfolio, acct)
        for pos in portfolio:
            c = pos.contract
            result["positions"].append({
                "symbol": c.symbol,
                "secType": c.secType,
                "strike": getattr(c, "strike", None),
                "right": getattr(c, "right", None),
                "expiry": getattr(c, "lastTradeDateOrContractMonth", None),
                "qty": float(pos.position),
                "avgCost": float(pos.avgCost),
                "marketPrice": float(getattr(pos, "marketPrice", 0) or 0),
                "marketValue": float(getattr(pos, "marketValue", 0) or 0),
                "unrealizedPNL": float(getattr(pos, "unrealizedPNL", 0) or 0),
            })
        await asyncio.to_thread(ib.disconnect)
        result["status"] = "ok"
    except Exception as e:
        result["error"] = str(e)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 2. MOOMOO — Futu/Moomoo (moomoo-api via OpenD)
# ═══════════════════════════════════════════════════════════════════════════
async def scan_moomoo() -> dict:
    """Connect to OpenD, pull account info + positions."""
    result = {"platform": "Moomoo", "status": "error", "balances": {}, "positions": []}
    try:
        from moomoo import (
            RET_OK,
            Currency,
            OpenSecTradeContext,
            SecurityFirm,
            TrdEnv,
            TrdMarket,
        )
        host = os.getenv("MOOMOO_HOST", "127.0.0.1")
        port = int(os.getenv("MOOMOO_PORT", "11111"))
        env_str = os.getenv("MOOMOO_ENV", "REAL").upper()
        trd_env = TrdEnv.REAL if env_str == "REAL" else TrdEnv.SIMULATE
        security_firm = SecurityFirm.FUTUCA

        ctx = OpenSecTradeContext(
            host=host, port=port, filter_trdmarket=TrdMarket.US,
            security_firm=security_firm,
        )

        for currency in (Currency.USD, Currency.CAD):
            ret, data = ctx.accinfo_query(trd_env=trd_env, currency=currency)
            if ret == RET_OK and not data.empty:
                row = data.iloc[0]
                result["currency"] = getattr(currency, "name", str(currency))
                result["balances"]["cash"] = float(row.get("cash", 0) or 0)
                result["balances"]["market_val"] = float(row.get("market_val", 0) or 0)
                result["balances"]["total_assets"] = float(row.get("total_assets", 0) or 0)
                result["balances"]["frozen_cash"] = float(row.get("frozen_cash", 0) or 0)
                result["net_liquidation"] = float(row.get("total_assets", 0) or 0)
                break

        # Positions
        ret2, pos_data = ctx.position_list_query(trd_env=trd_env)
        if ret2 == RET_OK and not pos_data.empty:
            pos_market_val = 0
            for _, row in pos_data.iterrows():
                mv = float(row.get("market_val", 0) or 0)
                pos_market_val += mv
                result["positions"].append({
                    "symbol": row.get("code", ""),
                    "qty": float(row.get("qty", 0) or 0),
                    "cost_price": float(row.get("cost_price", 0) or 0),
                    "market_val": mv,
                    "pl_val": float(row.get("pl_val", 0) or 0),
                })
            # If accinfo didn't give total, use positions sum
            if not result.get("net_liquidation"):
                result["net_liquidation"] = pos_market_val + result["balances"].get("cash", 0)
            result["balances"]["positions_market_val"] = pos_market_val
        ctx.close()
        result["status"] = "ok"
        result["account_id"] = "FUTUCA"
    except Exception as e:
        result["error"] = str(e)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3. NDAX — National Digital Asset Exchange (ccxt)
# ═══════════════════════════════════════════════════════════════════════════
async def scan_ndax() -> dict:
    """Check NDAX balance via ccxt. Account was LIQUIDATED 2026-03-18."""
    result = {"platform": "NDAX", "status": "error", "balances": {}, "positions": [],
              "note": "LIQUIDATED 2026-03-18 — sold all XRP+ETH → $4,492 CAD"}
    try:
        import ccxt
        api_key = os.getenv("NDAX_API_KEY", "")
        api_secret = os.getenv("NDAX_API_SECRET", "")
        uid = os.getenv("NDAX_USER_ID", "")
        if not api_key:
            result["error"] = "NDAX_API_KEY not set"
            return result
        exchange = ccxt.ndax({
            "apiKey": api_key,
            "secret": api_secret,
            "uid": uid,
            "login": uid,
            "password": api_secret,
        })
        bal = exchange.fetch_balance()
        total = bal.get("total", {})
        for asset, amount in total.items():
            if amount and float(amount) > 0:
                result["balances"][asset] = float(amount)

        cad_cash = float(result["balances"].get("CAD", 0) or 0)
        positions = []
        mtm_cad = cad_cash
        for asset, qty in result["balances"].items():
            if asset == "CAD":
                continue
            quantity = float(qty or 0)
            if quantity <= 0:
                continue

            price_cad = 0.0
            try:
                ticker = exchange.fetch_ticker(f"{asset}/CAD")
                price_cad = float(ticker.get("last") or ticker.get("close") or 0.0)
            except Exception:
                price_cad = 0.0

            value_cad = quantity * price_cad if price_cad > 0 else 0.0
            mtm_cad += value_cad
            positions.append({
                "symbol": asset,
                "qty": quantity,
                "price_cad": price_cad,
                "market_val_cad": value_cad,
            })

        result["positions"] = positions
        result["net_liquidation"] = mtm_cad
        result["currency"] = "CAD"
        result["status"] = "ok"
    except Exception as e:
        result["error"] = str(e)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. POLYMARKET — CLOB + On-chain (py_clob_client + web3)
# ═══════════════════════════════════════════════════════════════════════════
async def scan_polymarket() -> dict:
    """Check Polymarket CLOB balance + on-chain USDC."""
    result = {"platform": "Polymarket", "status": "error", "balances": {}, "positions": []}
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

        key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
        funder = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "")
        chain_id = int(os.environ.get("POLYMARKET_CHAIN_ID", "137"))
        sig_type = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "1"))

        if not key:
            result["error"] = "POLYMARKET_PRIVATE_KEY not set"
            return result

        client = ClobClient(
            "https://clob.polymarket.com",
            key=key, chain_id=chain_id,
            signature_type=sig_type, funder=funder,
        )
        # Derive API creds
        try:
            creds = client.derive_api_key()
            client.set_api_creds(creds)
        except Exception:
            creds = client.create_api_key()
            client.set_api_creds(creds)

        # CLOB balance (collateral = USDC)
        ba = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        clob_bal = 0
        if isinstance(ba, dict):
            raw = ba.get("balance", 0)
            # Balance may already be in USDC units or in raw (1e6) units
            val = float(raw)
            clob_bal = val / 1e6 if val > 10000 else val  # heuristic: raw > 10000 means wei-like
        result["balances"]["CLOB_USDC"] = clob_bal
        result["net_liquidation"] = clob_bal

        # Check all sig type balances (only add extras, don't duplicate primary)
        for st in [0, 1, 2]:
            label = {0: "EOA", 1: "POLY_PROXY", 2: "GNOSIS_SAFE"}[st]
            try:
                c2 = ClobClient("https://clob.polymarket.com", key=key, chain_id=chain_id,
                                signature_type=st, funder=funder)
                cr2 = c2.derive_api_key()
                c2.set_api_creds(cr2)
                ba2 = c2.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
                if isinstance(ba2, dict):
                    raw2 = float(ba2.get("balance", 0))
                    val2 = raw2 / 1e6 if raw2 > 10000 else raw2
                    if val2 > 0:
                        result["balances"][f"CLOB_{label}"] = val2
            except Exception:
                pass

        result["status"] = "ok"
    except Exception as e:
        result["error"] = str(e)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5. METAMASK / POLYGON WALLET — Web3 on-chain balance
# ═══════════════════════════════════════════════════════════════════════════
async def scan_metamask() -> dict:
    """Check on-chain balances: MATIC, ETH, USDC on Polygon + Ethereum."""
    result = {"platform": "MetaMask/Polygon", "status": "error", "balances": {}}
    try:
        from web3 import Web3

        eoa = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "")
        if not eoa:
            # Derive from private key
            pk = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
            if pk:
                from eth_account import Account
                eoa = Account.from_key(pk).address
        if not eoa:
            result["error"] = "No wallet address found"
            return result

        result["address"] = eoa

        # Polygon balances
        for rpc in POLYGON_RPCS:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
                if w3.is_connected():
                    # MATIC/POL
                    matic_wei = w3.eth.get_balance(eoa)
                    result["balances"]["MATIC"] = float(w3.from_wei(matic_wei, "ether"))
                    # USDC.e on Polygon
                    usdc_contract = w3.eth.contract(
                        address=Web3.to_checksum_address(USDC_POLYGON),
                        abi=ERC20_BALANCE_ABI,
                    )
                    usdc_raw = usdc_contract.functions.balanceOf(eoa).call()
                    result["balances"]["USDC_POLYGON"] = float(usdc_raw) / 1e6
                    break
            except Exception:
                continue

        # Ethereum balances
        for rpc in ETH_RPCS:
            try:
                w3e = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
                if w3e.is_connected():
                    eth_wei = w3e.eth.get_balance(eoa)
                    result["balances"]["ETH"] = float(w3e.from_wei(eth_wei, "ether"))
                    usdc_eth_contract = w3e.eth.contract(
                        address=Web3.to_checksum_address(USDC_ETH),
                        abi=ERC20_BALANCE_ABI,
                    )
                    usdc_eth_raw = usdc_eth_contract.functions.balanceOf(eoa).call()
                    result["balances"]["USDC_ETH"] = float(usdc_eth_raw) / 1e6
                    break
            except Exception:
                continue

        result["status"] = "ok"
    except Exception as e:
        result["error"] = str(e)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6. EQ BANK — Manual (no API available)
# ═══════════════════════════════════════════════════════════════════════════
async def scan_eqbank() -> dict:
    """EQ Bank has no API. Read from .env manual entry or return stub."""
    result = {"platform": "EQ Bank", "status": "manual", "balances": {}, "currency": "CAD",
              "note": "No API — Canadian digital bank. Update EQ_BANK_BALANCE in .env manually."}
    bal_str = os.getenv("EQ_BANK_BALANCE", "0")
    try:
        bal = float(bal_str)
    except ValueError:
        bal = 0
    if bal > 0:
        result["balances"]["CAD_SAVINGS"] = bal
        result["net_liquidation"] = bal
        result["status"] = "ok"
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 7. WEALTHSIMPLE — via SnapTrade (not yet connected)
# ═══════════════════════════════════════════════════════════════════════════
async def scan_wealthsimple() -> dict:
    """Check Wealthsimple via SnapTrade API. Requires setup first."""
    result = {"platform": "Wealthsimple", "status": "not_connected", "balances": {}, "positions": [],
              "note": "Requires SnapTrade setup: dashboard.snaptrade.com → get clientId+consumerKey"}

    client_id = os.getenv("SNAPTRADE_CLIENT_ID", "")
    consumer_key = os.getenv("SNAPTRADE_CONSUMER_KEY", "")
    user_id = os.getenv("SNAPTRADE_USER_ID", "")
    user_secret = os.getenv("SNAPTRADE_USER_SECRET", "")

    if not client_id or not consumer_key:
        result["error"] = "SNAPTRADE_CLIENT_ID / SNAPTRADE_CONSUMER_KEY not set. Sign up at dashboard.snaptrade.com"
        return result

    if not user_id or not user_secret:
        result["error"] = "SNAPTRADE_USER_ID / SNAPTRADE_USER_SECRET not set. Run SnapTrade user registration first."
        result["setup_hint"] = "python _setup_snaptrade.py --register"
        return result

    try:
        from snaptrade_client import Configuration
        from snaptrade_client.client import SnapTrade
        cfg = Configuration(consumer_key=consumer_key, client_id=client_id)
        snaptrade = SnapTrade(configuration=cfg)

        # List accounts
        accounts_resp = snaptrade.account_information.list_user_accounts(
            user_id=user_id, user_secret=user_secret,
        )
        total_value = 0.0
        for acct in (accounts_resp.body or []):
            acct_id = str(getattr(acct, "id", "unknown"))
            acct_name = getattr(acct, "name", acct_id)
            # Get balance for each account
            try:
                bal_resp = snaptrade.account_information.get_user_account_balance(
                    user_id=user_id, user_secret=user_secret, account_id=acct_id,
                )
                for b in (bal_resp.body or []):
                    currency = getattr(b, "currency", "CAD")
                    total = float(getattr(b, "total", 0) or 0)
                    result["balances"][f"{acct_name}_{currency}"] = total
                    total_value += total
            except Exception:
                pass
            # Get positions for each account
            try:
                pos_resp = snaptrade.account_information.get_user_account_positions(
                    user_id=user_id, user_secret=user_secret, account_id=acct_id,
                )
                for pos in (pos_resp.body or []):
                    sym_obj = getattr(pos, "symbol", None)
                    sym = getattr(sym_obj, "symbol", "?") if sym_obj else "?"
                    qty = float(getattr(pos, "units", 0) or 0)
                    mv = float(getattr(pos, "market_value", 0) or 0)
                    result["positions"].append({
                        "account": acct_name,
                        "symbol": sym,
                        "qty": qty,
                        "market_value": mv,
                    })
            except Exception:
                pass

        result["net_liquidation"] = total_value
        result["status"] = "ok"
    except ImportError:
        result["error"] = "snaptrade-python-sdk not installed. Run: pip install snaptrade-python-sdk"
    except Exception as e:
        result["error"] = str(e)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════
async def scan_all() -> dict:
    """Run all scanners and aggregate results."""
    scanners = {
        "ibkr": scan_ibkr,
        "moomoo": scan_moomoo,
        "ndax": scan_ndax,
        "polymarket": scan_polymarket,
        "metamask": scan_metamask,
        "eqbank": scan_eqbank,
        "wealthsimple": scan_wealthsimple,
    }
    results = {}
    for name, fn in scanners.items():
        print(f"  Scanning {name}...", end=" ", flush=True)
        try:
            results[name] = await fn()
            status = results[name].get("status", "?")
            print(f"[{status.upper()}]")
        except Exception as e:
            results[name] = {"platform": name, "status": "error", "error": str(e)}
            print(f"[EXCEPTION: {e}]")

    # Compute totals
    total_usd = 0
    total_cad = 0
    for name, r in results.items():
        if name.startswith("_"):
            continue
        nl = r.get("net_liquidation", 0)
        currency = r.get("currency", "USD")
        if currency == "CAD":
            total_cad += nl
            total_usd += nl * CAD_USD_RATE
        else:
            total_usd += nl
        # Add on-chain USDC from MetaMask (not already in net_liquidation)
        if name == "metamask":
            for k, v in r.get("balances", {}).items():
                if "USDC" in k:
                    total_usd += v

    results["_summary"] = {
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "total_usd_approx": round(total_usd, 2),
        "total_cad_component": round(total_cad, 2),
        "cad_usd_rate": CAD_USD_RATE,
    }
    return results


def print_report(results: dict):
    """Pretty-print the balance report."""
    print(_header("AAC UNIFIED BALANCE REPORT"))
    print(f"  Scan Time: {results['_summary']['scan_time']}")
    print()

    grand_total = 0

    for key in ["ibkr", "moomoo", "ndax", "polymarket", "metamask", "eqbank", "wealthsimple"]:
        r = results.get(key, {})
        platform = r.get("platform", key.upper())
        status = r.get("status", "unknown")

        print(f"  ┌─ {platform} [{status.upper()}]")

        if r.get("error"):
            print(f"  │  ⚠ {r['error']}")
        if r.get("note"):
            print(f"  │  ℹ {r['note']}")
        if r.get("account"):
            print(f"  │  Account: {r['account']}")
        if r.get("address"):
            print(f"  │  Address: {r['address']}")

        balances = r.get("balances", {})
        if balances:
            print(f"  │  Balances:")
            for asset, amount in balances.items():
                currency = r.get("currency", "USD")
                print(f"  │    {asset}: {amount:,.2f} {currency}")

        positions = r.get("positions", [])
        if positions:
            print(f"  │  Positions ({len(positions)}):")
            for p in positions[:15]:  # cap display
                sym = p.get("symbol", "?")
                qty = p.get("qty", 0)
                sec = p.get("secType", "")
                strike = p.get("strike", "")
                right = p.get("right", "")
                expiry = p.get("expiry", "")
                mv = p.get("market_val", p.get("marketValue", 0))
                cost = p.get("avgCost", p.get("cost_price", 0))
                label = sym
                if sec == "OPT":
                    label = f"{sym} {strike}{right} {expiry}"
                print(f"  │    {label}: {qty:+.0f} @ ${cost:,.2f} (mv=${mv:,.2f})")

        nl = r.get("net_liquidation", 0)
        currency = r.get("currency", "USD")
        if nl:
            print(f"  │  Net Liquidation: ${nl:,.2f} {currency}")
            grand_total += nl * (CAD_USD_RATE if currency == "CAD" else 1)

        # Add on-chain USDC from MetaMask only
        if key == "metamask":
            for k, v in balances.items():
                if "USDC" in k and v > 0:
                    grand_total += v

        print(f"  └{'─' * 50}")
        print()

    print(_header("GRAND TOTAL"))
    print(f"  Approximate Total:  ${grand_total:,.2f} USD")
    print(f"  CAD component:      ${results['_summary']['total_cad_component']:,.2f} CAD")
    print(f"  CAD/USD rate used:  {CAD_USD_RATE}")
    print(f"  {'═' * 56}")


def save_json(results: dict, path: str = "balance_snapshot.json"):
    """Save results to JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
async def main():
    parser = argparse.ArgumentParser(description="AAC Unified Balance Scanner")
    parser.add_argument("--json", action="store_true", help="Output JSON to balance_snapshot.json")
    parser.add_argument("--update", action="store_true", help="Update doctrine with scanned balances")
    args = parser.parse_args()

    print(_header("AAC BALANCE SCANNER — Scanning All Accounts"))
    results = await scan_all()

    print_report(results)

    if args.json or args.update:
        save_json(results)

    if args.update:
        from config.account_balances import Balances
        print("\n  Syncing to central config (data/account_balances.json)...")
        report = Balances.sync_from_scan(results)
        for acct, status in report.items():
            print(f"    {acct}: {status}")
        print(f"  Total portfolio: ${Balances.total_portfolio_usd():,.2f} USD")
        print("  Central config updated. All modules will see new values.")

    return results


if __name__ == "__main__":
    asyncio.run(main())
