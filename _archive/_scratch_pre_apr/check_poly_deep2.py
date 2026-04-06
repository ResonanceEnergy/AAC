"""Deep dive #2: Authenticated CLOB check + Polygonscan tx history + CTF exchange."""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, ".")
from shared.config_loader import load_env_file

load_env_file()

EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
PROXY = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"
POLYGON_API = os.getenv("POLYGON_API_KEY", "KK12YHTrpB24R9mwdD3u25rC6ePnCVw8")

def api_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read())

# ============================================================
# PART 1: Authenticated CLOB client balance check
# ============================================================
print("=" * 60)
print("PART 1: Authenticated CLOB Client")
print("=" * 60)

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
funder = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
chain = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))

print(f"  Private Key: {pk[:10]}...{pk[-6:]}")
print(f"  Funder: {funder}")
print(f"  Chain: {chain}")
print(f"  Sig Type: {sig_type}")

client = ClobClient(
    "https://clob.polymarket.com",
    key=pk,
    chain_id=chain,
    signature_type=sig_type,
    funder=funder,
)
creds = client.create_or_derive_api_creds()
client.set_api_creds(creds)
print(f"  API Key: {creds.api_key}")

# Check all balance types
for asset_type in [AssetType.COLLATERAL, AssetType.CONDITIONAL]:
    for st in [0, 1, 2]:
        try:
            params = BalanceAllowanceParams(asset_type=asset_type, signature_type=st)
            bal = client.get_balance_allowance(params)
            b = bal.get("balance", "0")
            if b != "0":
                print(f"  *** FOUND BALANCE: {asset_type} sig={st}: {bal}")
            else:
                print(f"  {asset_type} sig={st}: balance=0")
        except Exception as e:
            print(f"  {asset_type} sig={st}: ERROR {e}")

# Check open orders (authenticated)
print("\nOpen Orders:")
try:
    orders = client.get_orders()
    if orders:
        print(f"  Count: {len(orders)}")
        for o in orders[:10]:
            print(f"  {o}")
    else:
        print("  No open orders")
except Exception as e:
    print(f"  Error: {e}")

# Check trade history (authenticated)
print("\nTrade History:")
try:
    trades = client.get_trades()
    if trades:
        print(f"  Count: {len(trades)}")
        for t in trades[:10]:
            print(f"  {t}")
    else:
        print("  No trades")
except Exception as e:
    print(f"  Error: {e}")

# ============================================================
# PART 2: Polygonscan transaction history
# ============================================================
print(f"\n{'=' * 60}")
print("PART 2: Polygonscan Transaction History")
print("=" * 60)

for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
    print(f"\n--- {name} ({addr[:12]}...) ---")

    # Normal txns
    try:
        url = f"https://api.polygonscan.com/api?module=account&action=txlist&address={addr}&startblock=0&endblock=99999999&page=1&offset=10&sort=desc&apikey={POLYGON_API}"
        data = api_get(url)
        txns = data.get("result", [])
        if isinstance(txns, list) and len(txns) > 0:
            print(f"  Normal TX count (last 10): {len(txns)}")
            for tx in txns[:5]:
                val_eth = int(tx.get("value", "0")) / 1e18
                print(f"    hash={tx['hash'][:16]}... from={tx['from'][:12]}... to={tx.get('to','')[:12]}... value={val_eth:.6f} MATIC")
        else:
            print(f"  No normal transactions found. API message: {data.get('message', '?')}")
    except Exception as e:
        print(f"  Normal TX error: {e}")
    time.sleep(0.3)

    # ERC-20 token transfers (includes USDC)
    try:
        url = f"https://api.polygonscan.com/api?module=account&action=tokentx&address={addr}&startblock=0&endblock=99999999&page=1&offset=20&sort=desc&apikey={POLYGON_API}"
        data = api_get(url)
        txns = data.get("result", [])
        if isinstance(txns, list) and len(txns) > 0:
            print(f"  ERC-20 transfers (last 20): {len(txns)}")
            for tx in txns[:10]:
                val = int(tx.get("value", "0"))
                decimals = int(tx.get("tokenDecimal", "18"))
                amount = val / (10 ** decimals)
                symbol = tx.get("tokenSymbol", "?")
                direction = "IN" if tx.get("to", "").lower() == addr.lower() else "OUT"
                print(f"    {direction} {amount:.4f} {symbol} from={tx['from'][:12]}... to={tx.get('to','')[:12]}...")
        else:
            print(f"  No ERC-20 transfers found. API message: {data.get('message', '?')}")
    except Exception as e:
        print(f"  ERC-20 error: {e}")
    time.sleep(0.3)

# ============================================================
# PART 3: Check Polymarket exchange/CTF contracts directly
# ============================================================
print(f"\n{'=' * 60}")
print("PART 3: Polymarket Contract Interactions")
print("=" * 60)

# Known Polymarket contracts on Polygon
POLYMARKET_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
POLYMARKET_NEG_RISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
POLYMARKET_NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"  # ConditionalTokens

# Check if PROXY has interacted with Polymarket contracts
for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
    print(f"\n--- Internal txns for {name} ---")
    try:
        url = f"https://api.polygonscan.com/api?module=account&action=txlistinternal&address={addr}&startblock=0&endblock=99999999&page=1&offset=10&sort=desc&apikey={POLYGON_API}"
        data = api_get(url)
        txns = data.get("result", [])
        if isinstance(txns, list) and len(txns) > 0:
            print(f"  Internal TX count: {len(txns)}")
            for tx in txns[:5]:
                val = int(tx.get("value", "0")) / 1e18
                print(f"    from={tx['from'][:12]}... to={tx.get('to','')[:12]}... value={val:.6f}")
        else:
            print(f"  No internal txns. Message: {data.get('message', '?')}")
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(0.3)

# ============================================================
# PART 4: Check if the funder address is actually a Polymarket proxy
# ============================================================
print(f"\n{'=' * 60}")
print("PART 4: Contract Code Check")
print("=" * 60)

for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
    try:
        url = f"https://api.polygonscan.com/api?module=proxy&action=eth_getCode&address={addr}&tag=latest&apikey={POLYGON_API}"
        data = api_get(url)
        code = data.get("result", "0x")
        is_contract = code != "0x" and len(code) > 2
        print(f"  {name} ({addr[:12]}...): {'CONTRACT' if is_contract else 'EOA (not a contract)'} (code len={len(code)})")
    except Exception as e:
        print(f"  {name}: Error {e}")
    time.sleep(0.3)

# ============================================================
# PART 5: Try Polymarket frontend API endpoints
# ============================================================
print(f"\n{'=' * 60}")
print("PART 5: Polymarket Frontend/Data APIs")
print("=" * 60)

endpoints = [
    ("Balance", "https://data-api.polymarket.com/balance?user={addr}"),
    ("Portfolio", "https://data-api.polymarket.com/portfolio?user={addr}"),
    ("Activity", "https://data-api.polymarket.com/activity?user={addr}&limit=10"),
    ("PnL", "https://data-api.polymarket.com/pnl?user={addr}"),
    ("Earnings", "https://data-api.polymarket.com/earnings?user={addr}"),
]

for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
    print(f"\n--- {name} ({addr[:12]}...) ---")
    for ep_name, ep_url in endpoints:
        url = ep_url.format(addr=addr)
        try:
            data = api_get(url)
            summary = json.dumps(data)
            if len(summary) > 200:
                summary = summary[:200] + "..."
            print(f"  {ep_name}: {summary}")
        except urllib.error.HTTPError as e:
            print(f"  {ep_name}: HTTP {e.code}")
        except Exception as e:
            print(f"  {ep_name}: {e}")
        time.sleep(0.3)

print("\n\nDONE.")
