"""
DEEP DIVE: Full Polymarket account analysis.
Key: 0x9a9f...c113 (imported into Polymarket at signup)
Proxy wallet: 0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8
"""
import json, os, sys, traceback
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
from urllib.request import Request, urlopen

HOST = "https://clob.polymarket.com"
KEY = os.environ["POLYMARKET_PRIVATE_KEY"]
CHAIN_ID = 137
FUNDER = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"
INFURA = "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213"

# Polymarket contract addresses on Polygon
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEGRISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
THIRD_ADDR = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"


def rpc(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = Request(INFURA, data=body, headers={"Content-Type": "application/json"})
    resp = json.loads(urlopen(req, timeout=10).read())
    if "error" in resp:
        raise Exception(resp["error"])
    return resp["result"]


def erc20_balance(token, addr):
    call_data = "0x70a08231" + addr[2:].lower().zfill(64)
    result = rpc("eth_call", [{"to": token, "data": call_data}, "latest"])
    return int(result, 16) / 1e6


def erc20_allowance(token, owner, spender):
    """allowance(address owner, address spender) -> uint256"""
    call_data = "0xdd62ed3e" + owner[2:].lower().zfill(64) + spender[2:].lower().zfill(64)
    result = rpc("eth_call", [{"to": token, "data": call_data}, "latest"])
    return int(result, 16) / 1e6


print("=" * 70)
print("POLYMARKET DEEP DIVE")
print("=" * 70)

# ─── 1. Derive EOA from private key ───
from eth_account import Account
eoa = Account.from_key(KEY)
print(f"\n[1] KEY DERIVATION")
print(f"  Private key: {KEY[:10]}...{KEY[-6:]}")
print(f"  Derived EOA: {eoa.address}")
print(f"  Proxy/Funder: {FUNDER}")
print()

# ─── 2. On-chain balances ───
print(f"[2] ON-CHAIN BALANCES (Polygon)")
# EOA
eoa_pol = int(rpc("eth_getBalance", [eoa.address, "latest"]), 16) / 1e18
eoa_usdc_e = erc20_balance(USDC_E, eoa.address)
eoa_nonce = int(rpc("eth_getTransactionCount", [eoa.address, "latest"]), 16)
print(f"  EOA ({eoa.address[:10]}...):")
print(f"    POL: {eoa_pol:.6f}")
print(f"    USDC.e: ${eoa_usdc_e:.2f}")
print(f"    Nonce: {eoa_nonce}")

# Proxy
proxy_pol = int(rpc("eth_getBalance", [FUNDER, "latest"]), 16) / 1e18
proxy_usdc_e = erc20_balance(USDC_E, FUNDER)
proxy_nonce = int(rpc("eth_getTransactionCount", [FUNDER, "latest"]), 16)
proxy_code = rpc("eth_getCode", [FUNDER, "latest"])
print(f"  Proxy ({FUNDER[:10]}...):")
print(f"    POL: {proxy_pol:.6f}")
print(f"    USDC.e: ${proxy_usdc_e:.2f}")
print(f"    Nonce: {proxy_nonce}")
print(f"    Is contract: {proxy_code != '0x'} ({(len(proxy_code)-2)//2} bytes)")
print()

# ─── 3. USDC.e Allowances from proxy to exchanges ───
print(f"[3] USDC.e ALLOWANCES (proxy -> exchanges)")
for name, addr in [("Exchange", EXCHANGE), ("NegRisk Exchange", NEGRISK_EXCHANGE), ("Third", THIRD_ADDR)]:
    allow = erc20_allowance(USDC_E, FUNDER, addr)
    print(f"  {name} ({addr[:10]}...): ${allow:,.2f}")
print()

# Also check EOA allowances
print(f"[3b] USDC.e ALLOWANCES (EOA -> exchanges)")
for name, addr in [("Exchange", EXCHANGE), ("NegRisk Exchange", NEGRISK_EXCHANGE), ("Third", THIRD_ADDR)]:
    allow = erc20_allowance(USDC_E, eoa.address, addr)
    print(f"  {name} ({addr[:10]}...): ${allow:,.2f}")
print()

# ─── 4. CLOB API - All signature types ───
print(f"[4] CLOB API - BALANCE CHECK (all sig types)")
for sig_type in [0, 1, 2]:
    label = {0: "EOA", 1: "POLY_PROXY", 2: "GNOSIS_SAFE"}[sig_type]
    try:
        client = ClobClient(HOST, key=KEY, chain_id=CHAIN_ID, signature_type=sig_type, funder=FUNDER)
        creds = client.derive_api_key()
        client.set_api_creds(creds)

        bal = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        balance = bal.get("balance", "?")
        
        # Also check conditional token balance
        # bal_cond = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL))
        
        print(f"  Sig {sig_type} ({label:12s}): API Key={creds.api_key[:16]}... Balance=${balance}")
        
        # Check allowances from CLOB perspective
        allowances = bal.get("allowances", {})
        for addr, val in allowances.items():
            if val != "0":
                print(f"    Allowance {addr[:10]}...: {val}")
        if all(v == "0" for v in allowances.values()):
            print(f"    All CLOB allowances: 0")
            
    except Exception as e:
        print(f"  Sig {sig_type} ({label:12s}): ERROR - {e}")
print()

# ─── 5. CLOB API - Orders, Trades, Positions ───
print(f"[5] CLOB API - ORDERS / TRADES / POSITIONS (sig_type=1, funder={FUNDER[:10]}...)")
try:
    client = ClobClient(HOST, key=KEY, chain_id=CHAIN_ID, signature_type=1, funder=FUNDER)
    creds = client.derive_api_key()
    client.set_api_creds(creds)
    
    # Open orders
    try:
        orders = client.get_orders()
        print(f"  Open orders: {len(orders) if isinstance(orders, list) else orders}")
    except Exception as e:
        print(f"  Open orders error: {e}")
    
    # Trades
    try:
        trades = client.get_trades()
        print(f"  Trades: {len(trades) if isinstance(trades, list) else trades}")
    except Exception as e:
        print(f"  Trades error: {e}")

except Exception as e:
    print(f"  Client init error: {e}")
print()

# ─── 6. Try without funder (sig_type=0, EOA mode) ───
print(f"[6] CLOB API - NO FUNDER (sig_type=0, pure EOA)")
try:
    client = ClobClient(HOST, key=KEY, chain_id=CHAIN_ID, signature_type=0)
    creds = client.derive_api_key()
    client.set_api_creds(creds)
    
    bal = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
    print(f"  Balance: ${bal.get('balance', '?')}")
    
    orders = client.get_orders()
    print(f"  Orders: {len(orders) if isinstance(orders, list) else orders}")
    
    trades = client.get_trades()
    print(f"  Trades: {len(trades) if isinstance(trades, list) else trades}")
except Exception as e:
    print(f"  ERROR: {e}")
print()

# ─── 7. Direct CLOB REST API calls ───
print(f"[7] DIRECT CLOB REST API")
# Check if there's a profile
for endpoint in [
    f"https://clob.polymarket.com/profile/{eoa.address}",
    f"https://clob.polymarket.com/profile/{FUNDER}",
]:
    try:
        resp = urlopen(Request(endpoint, headers={"Accept": "application/json"}), timeout=10)
        data = json.loads(resp.read())
        print(f"  Profile {endpoint.split('/')[-1][:10]}...: {json.dumps(data)[:200]}")
    except Exception as e:
        print(f"  Profile {endpoint.split('/')[-1][:10]}...: {e}")

# Check gamma markets API for positions
for addr in [eoa.address, FUNDER]:
    try:
        url = f"https://gamma-api.polymarket.com/positions?user={addr}"
        resp = urlopen(Request(url, headers={"Accept": "application/json"}), timeout=10)
        data = json.loads(resp.read())
        count = len(data) if isinstance(data, list) else "N/A"
        print(f"  Gamma positions ({addr[:10]}...): {count}")
        if isinstance(data, list) and data:
            for pos in data[:3]:
                print(f"    {json.dumps(pos)[:150]}")
    except Exception as e:
        print(f"  Gamma positions ({addr[:10]}...): {e}")

print()

# ─── 8. Polymarket data API ───
print(f"[8] POLYMARKET DATA API - Activity")
for addr in [eoa.address, FUNDER]:
    try:
        url = f"https://data-api.polymarket.com/activity?user={addr.lower()}"
        resp = urlopen(Request(url, headers={"Accept": "application/json"}), timeout=10)
        data = json.loads(resp.read())
        count = len(data) if isinstance(data, list) else "N/A"
        print(f"  Activity ({addr[:10]}...): {count} items")
        if isinstance(data, list) and data:
            for item in data[:3]:
                print(f"    {json.dumps(item)[:150]}")
    except Exception as e:
        print(f"  Activity ({addr[:10]}...): {e}")

print()

# ─── 9. Check proxy factory mapping ───
print(f"[9] PROXY FACTORY - CREATE2 VERIFICATION")
FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"
IMPL = "0x44e999d5c2f66ef0861317f9a4805ac2e90aeb4f"

try:
    from web3 import Web3
    w3 = Web3()
    
    # Standard EIP-1167 init code
    init_code_hex = "3d602d80600a3d3981f3363d3d373d3d3d363d73" + IMPL[2:].lower() + "5af43d82803e903d91602b57fd5bf3"
    init_code_hash = w3.keccak(bytes.fromhex(init_code_hex))
    
    # Try many salt formulations
    salts = {
        "keccak(EOA padded 32)": w3.keccak(bytes.fromhex(eoa.address[2:].lower().zfill(64))),
        "keccak(EOA 20 bytes)": w3.keccak(bytes.fromhex(eoa.address[2:].lower())),
        "raw EOA padded 32": bytes.fromhex(eoa.address[2:].lower().zfill(64)),
        "zero salt": bytes(32),
        "keccak(abi.encode(addr,0))": w3.keccak(bytes.fromhex(eoa.address[2:].lower().zfill(64) + "0" * 64)),
        "keccak(abi.encode(addr,1))": w3.keccak(bytes.fromhex(eoa.address[2:].lower().zfill(64) + "0" * 63 + "1")),
    }
    
    for name, salt in salts.items():
        if isinstance(salt, bytes) and len(salt) == 32:
            pre = b'\xff' + bytes.fromhex(FACTORY[2:]) + salt + init_code_hash
            computed = "0x" + w3.keccak(pre).hex()[-40:]
            match = computed.lower() == FUNDER.lower()
            if match:
                print(f"  *** MATCH *** Salt={name} -> {computed}")
            # Only print non-matches in verbose mode
    
    # Also try with EOA as uint256 directly (no keccak)
    for i in range(5):
        salt_raw = bytes.fromhex(eoa.address[2:].lower().zfill(64))
        # Vary the salt with a nonce
        salt_with_nonce = w3.keccak(salt_raw + i.to_bytes(32, 'big'))
        pre = b'\xff' + bytes.fromhex(FACTORY[2:]) + salt_with_nonce + init_code_hash
        computed = "0x" + w3.keccak(pre).hex()[-40:]
        if computed.lower() == FUNDER.lower():
            print(f"  *** MATCH with nonce {i} ***")
    
    # Direct: just the address left-padded
    salt_left = bytes(12) + bytes.fromhex(eoa.address[2:].lower())
    pre = b'\xff' + bytes.fromhex(FACTORY[2:]) + salt_left + init_code_hash
    computed = "0x" + w3.keccak(pre).hex()[-40:]
    if computed.lower() == FUNDER.lower():
        print(f"  *** MATCH with left-padded address ***")
    
    print(f"  (Tested {len(salts) + 6} salt formulations)")
    print(f"  Note: If no match found, the proxy may use a different CREATE2 pattern")
    
except ImportError:
    print("  web3 not available for CREATE2 check")
except Exception as e:
    print(f"  Error: {e}")

print()

# ─── 10. Check Polymarket subgraph ───
print(f"[10] POLYMARKET SUBGRAPH - User lookup")
subgraph_url = "https://api.thegraph.com/subgraphs/name/polymarket/polymarket-matic"
for addr in [eoa.address.lower(), FUNDER.lower()]:
    query = json.dumps({
        "query": f'{{ user(id: "{addr}") {{ id numTrades lastTradeTimestamp positions {{ id }} }} }}'
    }).encode()
    try:
        req = Request(subgraph_url, data=query, headers={"Content-Type": "application/json"})
        resp = json.loads(urlopen(req, timeout=10).read())
        user_data = resp.get("data", {}).get("user")
        if user_data:
            print(f"  User {addr[:10]}...: trades={user_data.get('numTrades')}, positions={len(user_data.get('positions', []))}")
        else:
            print(f"  User {addr[:10]}...: not found")
    except Exception as e:
        print(f"  User {addr[:10]}...: {e}")

print()
print("=" * 70)
print("DEEP DIVE COMPLETE")
print("=" * 70)
