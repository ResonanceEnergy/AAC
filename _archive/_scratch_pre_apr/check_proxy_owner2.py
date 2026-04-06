"""Check on-chain state of 0xF4Ba... proxy wallet on Polygon."""
import json, os, sys
from urllib.request import Request, urlopen

INFURA = "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213"
PROXY = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e on Polygon
USDC   = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"  # native USDC on Polygon

def rpc(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = Request(INFURA, data=body, headers={"Content-Type": "application/json"})
    return json.loads(urlopen(req, timeout=10).read())["result"]

# 1. Check if it's a contract
code = rpc("eth_getCode", [PROXY, "latest"])
print(f"Address: {PROXY}")
print(f"Code length: {len(code)} chars ({(len(code)-2)//2} bytes)")
print(f"Is contract: {code != '0x'}")
print()

# 2. Check POL balance
bal_hex = rpc("eth_getBalance", [PROXY, "latest"])
bal = int(bal_hex, 16) / 1e18
print(f"POL balance: {bal:.6f}")

# 3. Check USDC.e balance (balanceOf)
call_data = "0x70a08231" + PROXY[2:].lower().zfill(64)
usdc_e_bal = rpc("eth_call", [{"to": USDC_E, "data": call_data}, "latest"])
usdc_e_val = int(usdc_e_bal, 16) / 1e6
print(f"USDC.e balance: ${usdc_e_val:.2f}")

# 4. Check native USDC balance
usdc_bal = rpc("eth_call", [{"to": USDC, "data": call_data}, "latest"])
usdc_val = int(usdc_bal, 16) / 1e6
print(f"USDC balance: ${usdc_val:.2f}")

# 5. Check nonce
nonce = int(rpc("eth_getTransactionCount", [PROXY, "latest"]), 16)
print(f"Nonce: {nonce}")
print()

# 6. If it's a contract, try to read owner/admin slots
if code != '0x':
    print("=== Proxy storage slots ===")
    impl_slot = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
    impl = rpc("eth_getStorageAt", [PROXY, impl_slot, "latest"])
    print(f"EIP-1967 impl: {impl}")
    
    admin_slot = "0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103"
    admin = rpc("eth_getStorageAt", [PROXY, admin_slot, "latest"])
    print(f"EIP-1967 admin: {admin}")
    
    slot0 = rpc("eth_getStorageAt", [PROXY, "0x0", "latest"])
    print(f"Slot 0 (owner?): {slot0}")
    
    slot1 = rpc("eth_getStorageAt", [PROXY, "0x1", "latest"])
    print(f"Slot 1: {slot1}")

    try:
        owner = rpc("eth_call", [{"to": PROXY, "data": "0x8da5cb5b"}, "latest"])
        print(f"owner() call: {owner}")
    except:
        print("owner() call failed")

print()

# 7. Check what proxy the private key's EOA maps to via Exchange contract
EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
# getPolyProxyWalletAddress(address) = 0xebd8ae50
call_data2 = "0xebd8ae50" + EOA[2:].lower().zfill(64)
try:
    proxy_for_eoa = rpc("eth_call", [{"to": EXCHANGE, "data": call_data2}, "latest"])
    proxy_addr = "0x" + proxy_for_eoa[-40:]
    print(f"Exchange.getPolyProxyWalletAddress(EOA {EOA[:10]}...) = {proxy_addr}")
    print(f"Matches 0xF4Ba...? {proxy_addr.lower() == PROXY.lower()}")
except Exception as e:
    print(f"getPolyProxyWalletAddress failed: {e}")

# 8. Reverse lookup: find what EOA maps to 0xF4Ba... proxy
# Try a different approach - check if the proxy has getOwner or similar
if code != '0x':
    print()
    print("=== Additional proxy queries ===")
    # getProxyAdmin() = 0x3e47158c
    try:
        result = rpc("eth_call", [{"to": PROXY, "data": "0x3e47158c"}, "latest"])
        print(f"getProxyAdmin(): {result}")
    except:
        pass
    
    # implementation() = 0x5c60da1b
    try:
        result = rpc("eth_call", [{"to": PROXY, "data": "0x5c60da1b"}, "latest"])
        print(f"implementation(): {result}")
    except:
        pass
