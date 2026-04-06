"""Decode the minimal proxy bytecode and find what EOA created/owns it."""
import json
import sys
from urllib.request import Request, urlopen

INFURA = "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213"
PROXY = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"
EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"

# Polymarket contracts on Polygon
EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEGRISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# Polymarket proxy factory / wallet factory
PROXY_FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"  # known Polymarket proxy factory

def rpc(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = Request(INFURA, data=body, headers={"Content-Type": "application/json"})
    resp = json.loads(urlopen(req, timeout=10).read())
    if "error" in resp:
        return f"ERROR: {resp['error']}"
    return resp["result"]

# 1. Read bytecode
code = rpc("eth_getCode", [PROXY, "latest"])
print(f"Bytecode: {code}")
print(f"Length: {(len(code)-2)//2} bytes")
print()

# Check if EIP-1167 minimal proxy: 363d3d373d3d3d363d73<address>5af43d82803e903d91602b57fd5bf3
if "363d3d373d3d3d363d73" in code:
    # EIP-1167 format
    idx = code.index("363d3d373d3d3d363d73") + len("363d3d373d3d3d363d73")
    impl_addr = "0x" + code[idx:idx+40]
    print(f"EIP-1167 Minimal Proxy -> Implementation: {impl_addr}")
else:
    print("Not standard EIP-1167. Trying to decode...")
    # Check for other patterns
    # Polymarket uses a custom minimal proxy pattern
    # The bytecode might contain the implementation address somewhere
    hex_code = code[2:]  # strip 0x
    print(f"Raw hex: {hex_code}")
    # Look for any 20-byte address-like patterns
    for i in range(0, len(hex_code) - 40, 2):
        potential = hex_code[i:i+40]
        if potential.startswith("000000"):
            continue
        # Check if it could be an address
        if len(set(potential)) > 4:  # not all zeros or simple pattern
            print(f"  Potential addr at offset {i//2}: 0x{potential}")

print()

# 2. Try Polygonscan API to find creation tx
POLYGONSCAN_KEY = "U8HCAUW68ZXNXSMEZGNGKIMWEA4U4VX9YM"  # etherscan key may work for polygon too
# Actually use the Etherscan V2 API with chainid=137
url = f"https://api.etherscan.io/v2/api?chainid=137&module=account&action=txlist&address={PROXY}&startblock=0&endblock=99999999&page=1&offset=20&sort=asc&apikey={POLYGONSCAN_KEY}"
try:
    resp = json.loads(urlopen(Request(url), timeout=15).read())
    if resp.get("status") == "1":
        txs = resp["result"]
        print(f"=== {len(txs)} transactions for {PROXY} ===")
        for tx in txs:
            direction = "IN" if tx["to"].lower() == PROXY.lower() else "OUT"
            value_eth = int(tx["value"]) / 1e18
            print(f"  {direction} | from={tx['from'][:12]}... to={tx['to'][:12]}... | {value_eth:.4f} POL | func={tx['input'][:10] if tx['input'] != '0x' else 'transfer'}")
    else:
        print(f"Polygonscan: {resp.get('message', 'error')} - {resp.get('result', '')}")
except Exception as e:
    print(f"Polygonscan error: {e}")

print()

# 3. Check internal txs (contract creation)
url2 = f"https://api.etherscan.io/v2/api?chainid=137&module=account&action=txlistinternal&address={PROXY}&startblock=0&endblock=99999999&page=1&offset=20&sort=asc&apikey={POLYGONSCAN_KEY}"
try:
    resp = json.loads(urlopen(Request(url2), timeout=15).read())
    if resp.get("status") == "1":
        txs = resp["result"]
        print(f"=== {len(txs)} internal transactions ===")
        for tx in txs:
            print(f"  from={tx.get('from','?')[:12]}... to={tx.get('to','?')[:12]}... type={tx.get('type','?')} value={int(tx.get('value',0))/1e18:.4f}")
    else:
        print(f"Internal txs: {resp.get('message', 'error')} - {resp.get('result', '')}")
except Exception as e:
    print(f"Internal tx error: {e}")

print()

# 4. Check ERC20 transfers to/from proxy
url3 = f"https://api.etherscan.io/v2/api?chainid=137&module=account&action=tokentx&address={PROXY}&startblock=0&endblock=99999999&page=1&offset=20&sort=asc&apikey={POLYGONSCAN_KEY}"
try:
    resp = json.loads(urlopen(Request(url3), timeout=15).read())
    if resp.get("status") == "1":
        txs = resp["result"]
        print(f"=== {len(txs)} ERC20 transfers ===")
        for tx in txs:
            direction = "IN" if tx["to"].lower() == PROXY.lower() else "OUT"
            decimals = int(tx.get("tokenDecimal", 18))
            value = int(tx["value"]) / (10 ** decimals)
            print(f"  {direction} | {tx['tokenSymbol']:>8} | {value:>15.4f} | from={tx['from'][:12]}... to={tx['to'][:12]}...")
    else:
        print(f"ERC20: {resp.get('message', 'error')} - {resp.get('result', '')}")
except Exception as e:
    print(f"ERC20 error: {e}")

# 5. Try getPolyProxyWalletAddress with different function selectors
print()
print("=== Proxy wallet lookup ===")
# Try multiple known function selectors for proxy wallet lookup
selectors = {
    "getPolyProxyWalletAddress": "0xebd8ae50",
    "proxyWalletFor": "0x9bc4add4",
    "getProxyWallet": "0x63e3e7d2",
}
for name, sel in selectors.items():
    call_data = sel + EOA[2:].lower().zfill(64)
    try:
        result = rpc("eth_call", [{"to": EXCHANGE, "data": call_data}, "latest"])
        if isinstance(result, str) and result.startswith("0x") and len(result) >= 42:
            addr = "0x" + result[-40:]
            print(f"  Exchange.{name}(EOA) = {addr}")
        else:
            print(f"  Exchange.{name}(EOA) = {result}")
    except Exception as e:
        print(f"  Exchange.{name} failed: {e}")
