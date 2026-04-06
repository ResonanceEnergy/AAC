"""Find the owner EOA of the Polymarket proxy wallet."""
import json
from urllib.request import Request, urlopen

INFURA = "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213"
PROXY = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"
FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"
IMPL = "0x44e999d5c2f66ef0861317f9a4805ac2e90aeb4f"
EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"

def rpc(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = Request(INFURA, data=body, headers={"Content-Type": "application/json"})
    resp = json.loads(urlopen(req, timeout=10).read())
    return resp.get("result", resp.get("error", "no result"))

# 1. Read storage slots 0-10 on the proxy
print("=== Proxy storage slots ===")
for i in range(11):
    slot = hex(i)
    val = rpc("eth_getStorageAt", [PROXY, slot, "latest"])
    if val and val != "0x0000000000000000000000000000000000000000000000000000000000000000":
        addr_part = "0x" + val[-40:] if isinstance(val, str) else ""
        print(f"  Slot {i}: {val}  (addr: {addr_part})")
    else:
        print(f"  Slot {i}: (zero)")

print()

# 2. Try various function calls on the proxy (delegates to implementation)
calls = {
    "owner()": "0x8da5cb5b",
    "getOwner()": "0x893d20e8",
    "admin()": "0xf851a440",
    "getAdmin()": "0x6e9960c3",
    "wallet()": "0x521eb273",  
    "signerAddress()": "0x5b7633d0",
    "signer()": "0x238ac933",
    "proxyOwner()": "0x025313a2",
    "masterCopy()": "0xa619486e",
    "getThreshold()": "0xe75235b8",
    "getOwners()": "0xa0e67e2b",
    "nonce()": "0xaffed0e0",
}
print("=== Function calls on proxy ===")
for name, sel in calls.items():
    try:
        result = rpc("eth_call", [{"to": PROXY, "data": sel}, "latest"])
        if isinstance(result, str) and result.startswith("0x") and len(result) > 2:
            if len(result) == 66:  # single address/uint256
                val = int(result, 16)
                addr = "0x" + result[-40:]
                if val > 0:
                    print(f"  {name:25s} = {result} (addr: {addr}, uint: {val})")
                else:
                    print(f"  {name:25s} = 0")
            else:
                print(f"  {name:25s} = {result[:80]}...")
        elif isinstance(result, dict):
            print(f"  {name:25s} REVERTED: {result}")
        else:
            print(f"  {name:25s} = {result}")
    except Exception as e:
        print(f"  {name:25s} ERROR: {e}")

print()

# 3. Try the factory contract
print("=== Factory queries ===")
factory_calls = {
    "getProxyWallet(addr)": "0x",  # We need the right selector
}

# Check if factory itself has info
factory_code = rpc("eth_getCode", [FACTORY, "latest"])
print(f"Factory code length: {(len(factory_code)-2)//2 if isinstance(factory_code, str) else 'error'} bytes")

# Try calling factory with our EOA to see what proxy it maps to
# polyProxy(address) 
factory_selectors = {
    "polyProxy": "0x37c3cc0c",
    "getProxy": "0xec618c04", 
    "proxyFor": "0x52d1902d",
    "wallets": "0x7bb98a68",
}
for name, sel in factory_selectors.items():
    call_data = sel + EOA[2:].lower().zfill(64)
    try:
        result = rpc("eth_call", [{"to": FACTORY, "data": call_data}, "latest"])
        if isinstance(result, str) and result.startswith("0x") and len(result) >= 42:
            addr = "0x" + result[-40:]
            if int(result, 16) > 0:
                print(f"  Factory.{name}(EOA) = {addr}")
        elif isinstance(result, dict):
            print(f"  Factory.{name}(EOA) REVERTED: {result.get('message','')}")
        else:
            print(f"  Factory.{name}(EOA) = {result}")
    except Exception as e:
        print(f"  Factory.{name} error: {e}")

print()

# 4. Read implementation contract to understand storage layout
print("=== Implementation contract ===")
impl_code = rpc("eth_getCode", [IMPL, "latest"])
print(f"Implementation code length: {(len(impl_code)-2)//2 if isinstance(impl_code, str) else 'error'} bytes")
# Read its storage too (though proxies delegate storage to proxy address)
for i in range(5):
    slot = hex(i)
    val = rpc("eth_getStorageAt", [IMPL, slot, "latest"])
    if val and val != "0x0000000000000000000000000000000000000000000000000000000000000000":
        print(f"  Impl Slot {i}: {val}")

print()

# 5. Key insight: with EIP-1167 proxy + CREATE2, the salt often encodes the owner
# Let's check if our EOA would produce the same proxy address via CREATE2
import hashlib
# CREATE2: keccak256(0xff ++ deployer ++ salt ++ keccak256(initCode))
# For Polymarket, salt is typically keccak256(abi.encode(ownerAddress, bytes32(0)))

# The init code for EIP-1167 clone:
init_code_prefix = "3d602d80600a3d3981f3"  # standard EIP-1167 creation code prefix
impl_stripped = IMPL[2:].lower()
runtime = f"363d3d373d3d3d363d73{impl_stripped}5af43d82803e903d91602b57fd5bf3"
init_code = init_code_prefix + runtime
print(f"Expected init code: {init_code}")

# Try with EOA as salt (raw address padded)
from hashlib import sha3_256
try:
    from eth_abi import encode as eth_encode
    # salt = keccak256(abi.encode(["address", "uint256"], [eoa, 0]))
    from web3 import Web3
    w3 = Web3()
    init_code_hash = w3.keccak(bytes.fromhex(init_code))
    
    # Try salt = keccak256(addr)
    salt1 = w3.keccak(bytes.fromhex(EOA[2:].lower().zfill(64)))
    
    # Try salt = address padded to 32 bytes  
    salt2 = bytes.fromhex(EOA[2:].lower().zfill(64))
    
    # CREATE2 = keccak256(0xff ++ factory ++ salt ++ keccak256(init_code))
    for salt_name, salt in [("keccak(EOA)", salt1), ("padded EOA", salt2)]:
        pre = b'\xff' + bytes.fromhex(FACTORY[2:]) + salt + init_code_hash
        addr = "0x" + w3.keccak(pre).hex()[-40:]
        match = addr.lower() == PROXY.lower()
        print(f"  CREATE2 with {salt_name}: {addr} {'*** MATCH! ***' if match else '(no match)'}")
        
except ImportError as e:
    print(f"  (web3/eth_abi not available for CREATE2 check: {e})")
    # Fallback: try with hashlib
    import hashlib
    # keccak256 via hashlib if available
    try:
        import _pysha3
    except:
        print("  (no keccak256 available)")
