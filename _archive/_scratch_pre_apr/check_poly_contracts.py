"""Check contract code and proxy status of both addresses."""
import json
import urllib.request

POLYGON_API = "KK12YHTrpB24R9mwdD3u25rC6ePnCVw8"
EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
PROXY = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"

def api_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read())

# API key test
url = f"https://api.polygonscan.com/api?module=account&action=balance&address=0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045&tag=latest&apikey={POLYGON_API}"
data = api_get(url)
status = data.get("status")
msg = data.get("message")
print(f"API Key Test: status={status} message={msg}")

for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
    print(f"\n{name} ({addr}):")

    # TX count
    url = f"https://api.polygonscan.com/api?module=proxy&action=eth_getTransactionCount&address={addr}&tag=latest&apikey={POLYGON_API}"
    data = api_get(url)
    nonce = int(data.get("result", "0x0"), 16)
    print(f"  TX count (nonce): {nonce}")

    # Contract code
    url = f"https://api.polygonscan.com/api?module=proxy&action=eth_getCode&address={addr}&tag=latest&apikey={POLYGON_API}"
    data = api_get(url)
    code = data.get("result", "0x")
    code_bytes = (len(code) - 2) // 2
    print(f"  Code length: {code_bytes} bytes")
    print(f"  Code: {code}")

    # EIP-1167 minimal proxy check
    if code.startswith("0x363d3d373d3d3d363d73"):
        impl = "0x" + code[22:62]
        print(f"  ** EIP-1167 Minimal Proxy -> implementation: {impl}")
    elif code == "0x" or len(code) <= 2:
        print(f"  ** Regular EOA (no contract code)")
    else:
        print(f"  ** Contract (not EIP-1167)")

    # MATIC balance
    url = f"https://api.polygonscan.com/api?module=account&action=balance&address={addr}&tag=latest&apikey={POLYGON_API}"
    data = api_get(url)
    bal = data.get("result", "0")
    if data.get("status") == "1":
        matic = int(bal) / 1e18
        print(f"  MATIC: {matic}")
    else:
        print(f"  MATIC query: {data.get('message', '?')} result={bal}")

    # Transaction list (simpler query)
    url = f"https://api.polygonscan.com/api?module=account&action=txlist&address={addr}&startblock=0&endblock=99999999&sort=desc&apikey={POLYGON_API}"
    data = api_get(url)
    msg = data.get("message", "")
    result = data.get("result", [])
    if isinstance(result, list):
        print(f"  Transactions: {len(result)} (message={msg})")
    else:
        print(f"  Transactions: message={msg} result={str(result)[:200]}")
