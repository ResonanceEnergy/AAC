"""Check wallet status using Etherscan V2 API + direct RPC."""
import json
import urllib.request
import time

EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
PROXY = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"
# USDC on Polygon
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
USDC_NATIVE = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"

def api_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read())

def rpc_call(rpc_url, method, params):
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(rpc_url, payload, {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    body = json.loads(resp.read())
    if "error" in body:
        raise Exception(body["error"])
    return body.get("result")

# Try multiple RPCs
RPCS = [
    "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213",  # free infura
    "https://polygon.drpc.org",
    "https://polygon.gateway.tenderly.co",
]

rpc = None
for r in RPCS:
    try:
        block = rpc_call(r, "eth_blockNumber", [])
        print(f"RPC {r[:40]}... connected. Block: {int(block, 16)}")
        rpc = r
        break
    except Exception as e:
        print(f"RPC {r[:40]}... failed: {e}")

if not rpc:
    # Fallback: try without auth
    rpc = "https://polygon-rpc.com"
    try:
        # polygon-rpc.com needs different approach
        block = rpc_call(rpc, "eth_blockNumber", [])
        print(f"RPC {rpc} connected. Block: {int(block, 16)}")
    except Exception as e:
        print(f"All RPCs failed. Last error: {e}")
        rpc = None

if rpc:
    print(f"\nUsing RPC: {rpc}")
    
    for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
        print(f"\n{'='*50}")
        print(f"{name}: {addr}")
        print("=" * 50)
        
        # Native MATIC balance
        bal = rpc_call(rpc, "eth_getBalance", [addr, "latest"])
        matic = int(bal, 16) / 1e18
        print(f"  MATIC: {matic:.8f}")
        
        # Code check
        code = rpc_call(rpc, "eth_getCode", [addr, "latest"])
        code_bytes = (len(code) - 2) // 2
        print(f"  Contract code: {code_bytes} bytes")
        print(f"  Raw code: {code}")
        
        if code.startswith("0x363d3d373d3d3d363d73"):
            impl = "0x" + code[22:62]
            print(f"  ** EIP-1167 MINIMAL PROXY -> implementation: {impl}")
        elif code == "0x":
            print(f"  ** Plain EOA")
        
        # Nonce
        nonce = rpc_call(rpc, "eth_getTransactionCount", [addr, "latest"])
        print(f"  Nonce: {int(nonce, 16)}")
        
        # USDC balances (balanceOf)
        for token_name, token_addr in [("USDC.e", USDC_E), ("USDC", USDC_NATIVE)]:
            padded = addr[2:].lower().zfill(64)
            calldata = "0x70a08231" + padded
            result = rpc_call(rpc, "eth_call", [{"to": token_addr, "data": calldata}, "latest"])
            bal_raw = int(result, 16)
            bal_usd = bal_raw / 1e6
            marker = " <<<" if bal_usd > 0 else ""
            print(f"  {token_name}: ${bal_usd:.2f}{marker}")
        
        time.sleep(0.3)
    
    # Also check the Polymarket "proxy factory" - derive our actual proxy from EOA
    # Polymarket proxy factory on Polygon
    PROXY_FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"
    print(f"\n{'='*50}")
    print("PROXY FACTORY CHECK")
    print("=" * 50)
    
    # getProxy(address) = 0x59659440 (function selector for getProxy)
    padded_eoa = EOA[2:].lower().zfill(64)
    calldata = "0x59659440" + padded_eoa
    try:
        result = rpc_call(rpc, "eth_call", [{"to": PROXY_FACTORY, "data": calldata}, "latest"])
        if result and result != "0x":
            derived_proxy = "0x" + result[-40:]
            print(f"  Proxy Factory says EOA's proxy is: {derived_proxy}")
            print(f"  Matches FUNDER_ADDRESS? {derived_proxy.lower() == PROXY.lower()}")
        else:
            print(f"  Raw result: {result}")
    except Exception as e:
        print(f"  Error: {e}")

    # Reverse: derive EOA for the proxy address
    # proxyOwner(address) or similar
    print(f"\n  Checking standard proxy patterns on PROXY contract...")
    
    # owner() = 0x8da5cb5b
    try:
        result = rpc_call(rpc, "eth_call", [{"to": PROXY, "data": "0x8da5cb5b"}, "latest"])
        if result and result != "0x" and len(result) > 2:
            owner = "0x" + result[-40:]
            print(f"  owner(): {owner}")
        else:
            print(f"  owner(): no result")
    except Exception as e:
        print(f"  owner() error: {e}")

else:
    print("\nNo RPC available, trying Etherscan V2...")
    # Etherscan V2 API
    POLYGON_KEY = "KK12YHTrpB24R9mwdD3u25rC6ePnCVw8"
    for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
        print(f"\n{name}:")
        url = f"https://api.etherscan.io/v2/api?chainid=137&module=account&action=balance&address={addr}&tag=latest&apikey={POLYGON_KEY}"
        try:
            data = api_get(url)
            print(f"  {json.dumps(data)[:300]}")
        except Exception as e:
            print(f"  Error: {e}")
