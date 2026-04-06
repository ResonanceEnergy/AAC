"""Live balance check across all APIs."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import dotenv_values

env = dotenv_values('.env')

# === 1. IBKR via ib_insync ===
print("=== IBKR (port 7496) ===")
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass
try:
    from ib_insync import IB
    ib = IB()
    ib.connect("127.0.0.1", 7496, clientId=99, timeout=5)
    acct = ib.managedAccounts()[0]
    summary = ib.accountSummary(acct)
    cash = next((s.value for s in summary if s.tag == "TotalCashValue" and s.currency == "USD"), "N/A")
    nlv = next((s.value for s in summary if s.tag == "NetLiquidation" and s.currency == "USD"), "N/A")
    positions = ib.positions()
    print(f"  Account: {acct}")
    print(f"  Cash: USD {cash}")
    print(f"  Net Liquidation: USD {nlv}")
    print(f"  Positions: {len(positions)}")
    for p in positions:
        c = p.contract
        mv = p.avgCost * abs(p.position)
        sym = c.symbol
        right = getattr(c, "right", "")
        strike = getattr(c, "strike", 0)
        exp = getattr(c, "lastTradeDateOrContractMonth", "")
        print(f"    {sym} {right}{strike} {exp} qty={p.position} avgCost={p.avgCost:.2f}")
    ib.disconnect()
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")

# === 2. Moomoo via futu ===
print()
print("=== MOOMOO (OpenD 11111) ===")
try:
    from futu import Currency, OpenSecTradeContext, TrdEnv, TrdMarket
    ctx = OpenSecTradeContext(host="127.0.0.1", port=11111, filter_trdmarket=TrdMarket.US, security_firm=None)
    ret, data = ctx.accinfo_query(trd_env=TrdEnv.REAL, currency=Currency.USD)
    if ret == 0:
        print("  USD Account Info:")
        for col in ["total_assets", "cash", "market_val", "frozen_cash"]:
            if col in data.columns:
                val = data[col].iloc[0]
                print(f"    {col}: {val}")
    else:
        print(f"  USD query failed: {data}")
    ret2, data2 = ctx.accinfo_query(trd_env=TrdEnv.REAL, currency=Currency.CAD)
    if ret2 == 0:
        print("  CAD Account Info:")
        for col in ["total_assets", "cash", "market_val", "frozen_cash"]:
            if col in data2.columns:
                val = data2[col].iloc[0]
                print(f"    {col}: {val}")
    else:
        print(f"  CAD query failed: {data2}")
    # positions
    ret3, pos = ctx.position_list_query(trd_env=TrdEnv.REAL)
    if ret3 == 0 and not pos.empty:
        print(f"  Positions ({len(pos)}):")
        for _, r in pos.iterrows():
            code = r.get("code", "?")
            qty = r.get("qty", 0)
            cost = r.get("cost_price", 0)
            mval = r.get("market_val", 0)
            print(f"    {code} qty={qty} cost={cost:.2f} mkt={mval:.2f}")
    elif ret3 == 0:
        print("  No positions")
    ctx.close()
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")

# === 3. Polymarket via public RPC ===
print()
print("=== POLYMARKET (Polygon on-chain) ===")
try:
    from web3 import Web3
    rpcs = [
        "https://polygon.llamarpc.com",
        "https://rpc-mainnet.maticvigil.com",
        "https://polygon-rpc.com",
    ]
    w3 = None
    for rpc in rpcs:
        try:
            w3t = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 5}))
            if w3t.is_connected():
                w3 = w3t
                print(f"  Connected via {rpc}")
                break
        except Exception:
            continue
    if w3 and w3.is_connected():
        funder = env.get("POLYMARKET_FUNDER_ADDRESS", "")
        usdc_addr = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        abi = [{"constant": True, "inputs": [{"name": "account", "type": "address"}],
                "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "type": "function"}]
        usdc = w3.eth.contract(address=Web3.to_checksum_address(usdc_addr), abi=abi)
        raw = usdc.functions.balanceOf(Web3.to_checksum_address(funder)).call()
        print(f"  USDC balance: ${raw / 1e6:.2f}")
        matic = w3.eth.get_balance(Web3.to_checksum_address(funder))
        print(f"  POL/MATIC: {matic / 1e18:.6f}")
    else:
        print("  All RPCs failed - cannot check on-chain balance")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")

# === 4. NDAX ===
print()
print("=== NDAX ===")
try:
    import ccxt
    ndax = ccxt.ndax({
        "apiKey": env.get("NDAX_API_KEY", ""),
        "secret": env.get("NDAX_API_SECRET", ""),
        "uid": env.get("NDAX_USER_ID", "") or env.get("NDAX_ACCOUNT_ID", ""),
    })
    bal = ndax.fetch_balance()
    non_zero = {k: v for k, v in bal.get("total", {}).items() if float(v) > 0}
    print(f"  Non-zero: {non_zero}")
    if not non_zero:
        print("  ALL ZERO - fully liquidated")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")

# === 5. Summary ===
print()
print("=== NO API AVAILABLE ===")
print("  WealthSimple: No API -- ask user for current TFSA balance")
print("  EQ Bank: No API -- ask user for current balance")
