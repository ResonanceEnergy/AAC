"""Show all Polymarket open orders and positions."""
import json
import os
import sys

import requests

sys.path.insert(0, ".")
from dotenv import load_dotenv

load_dotenv()

PROXY = os.getenv("POLYMARKET_FUNDER_ADDRESS", "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8")
EOA = "0x6e9b70D1175ecA144111743503441300A9494297"

# ═══════════════════════════════════════════════════════════════
# 1. OPEN ORDERS via CLOB SDK
# ═══════════════════════════════════════════════════════════════
print("=" * 120)
print("  POLYMARKET OPEN ORDERS (CLOB API)")
print("=" * 120)

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OpenOrderParams

creds = ApiCreds(
    api_key=os.getenv("POLYMARKET_API_KEY", ""),
    api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
    api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", ""),
)
client = ClobClient(
    "https://clob.polymarket.com",
    key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
    chain_id=137,
    signature_type=1,
    funder=PROXY,
    creds=creds,
)

orders = client.get_orders(OpenOrderParams())
if isinstance(orders, list):
    live_orders = [o for o in orders if o.get("status", "") in ("live", "LIVE", "matched", "MATCHED", "open", "OPEN")]
    if not live_orders:
        live_orders = orders  # show all if no status filter
    print(f"  Total orders returned: {len(orders)}")
    print(f"  Live/Open orders: {len(live_orders)}")
    print()
    hdr = "  {:>3} {:>8} {:>4} {:>8} {:>10} {:>10} {:12} {}"
    print(hdr.format("#", "STATUS", "SIDE", "PRICE", "SIZE", "FILLED", "ORDER_ID", "MARKET"))
    print("  " + "-" * 115)
    total_committed = 0.0
    for i, o in enumerate(live_orders[:80], 1):
        oid = o.get("id", o.get("orderID", ""))[:12]
        status = o.get("status", "?")
        side = o.get("side", "?")
        price = float(o.get("price", 0))
        size = float(o.get("original_size", o.get("size", 0)))
        filled = float(o.get("size_matched", o.get("filled", 0)))
        cost = price * size
        total_committed += cost
        asset = o.get("asset_id", o.get("token_id", ""))[:20]
        market = o.get("market", "")[:55] or asset
        print(f"  {i:3} {status:>8} {side:>4} {price:>8.4f} {size:>10.1f} {filled:>10.1f} {oid:12} {market}")
    print("  " + "-" * 115)
    print(f"  TOTAL ORDERS: {len(live_orders)}  |  TOTAL COMMITTED: ${total_committed:,.2f}")
else:
    print(f"  Response: {str(orders)[:500]}")

# ═══════════════════════════════════════════════════════════════
# 2. POSITIONS via Data API
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 120)
print("  POLYMARKET POSITIONS (DATA API)")
print("=" * 120)

DATA_API = "https://data-api.polymarket.com"
for label, addr in [("Proxy", PROXY), ("EOA", EOA)]:
    try:
        r = requests.get(f"{DATA_API}/positions?user={addr}", timeout=15)
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            print(f"\n  [{label}] {addr} -- {len(data)} positions")
            print()
            phdr = "  {:>3} {:>6} {:>10} {:>10} {:>10} {}"
            print(phdr.format("#", "SIDE", "SIZE", "AVG_PRICE", "VALUE", "MARKET"))
            print("  " + "-" * 100)
            total_val = 0
            for j, p in enumerate(data[:60], 1):
                sz = float(p.get("size", p.get("amount", 0)))
                avg = float(p.get("avgPrice", p.get("avg_price", 0)))
                cur = float(p.get("curPrice", p.get("price", 0)))
                val = sz * cur if cur > 0 else sz * avg
                total_val += val
                side = p.get("side", p.get("outcome", "?"))
                title = p.get("title", p.get("market", ""))[:55] or p.get("asset", "")[:20]
                print(f"  {j:3} {side:>6} {sz:>10.1f} {avg:>10.4f} {val:>10.2f} {title}")
            print("  " + "-" * 100)
            dollar = "$"
            print(f"  TOTAL POSITION VALUE: {dollar}{total_val:,.2f}")
        else:
            print(f"\n  [{label}] {addr} -- 0 positions")
    except Exception as e:
        print(f"\n  [{label}] Error: {e}")

# ═══════════════════════════════════════════════════════════════
# 3. USDC BALANCE
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 120)
print("  WALLET BALANCE")
print("=" * 120)
try:
    # USDC.e on Polygon
    USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    RPC = "https://polygon-rpc.com"
    # balanceOf(address) selector = 0x70a08231
    for label, addr in [("Proxy", PROXY), ("EOA", EOA)]:
        padded = addr[2:].lower().zfill(64)
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{"to": USDC_E, "data": f"0x70a08231{padded}"}, "latest"],
            "id": 1,
        }
        r = requests.post(RPC, json=payload, timeout=10)
        result = r.json().get("result", "0x0")
        bal = int(result, 16) / 1e6
        dollar = "$"
        print(f"  [{label}] {addr}: {dollar}{bal:,.2f} USDC.e")
except Exception as e:
    print(f"  Balance check error: {e}")

print()
print("=" * 120)
