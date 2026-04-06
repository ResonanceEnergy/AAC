"""
Compute deterministic Polymarket proxy address for our EOA.
Polymarket uses CREATE2 via a ProxyFactory to deploy minimal proxies.
The address is: keccak256(0xff ++ factory ++ salt ++ keccak256(init_code))[12:]
where salt = keccak256(abi.encode(eoa_address, impl_address))
"""
import os

import dotenv
import requests
from eth_abi import encode
from web3 import Web3

dotenv.load_dotenv()

RPC = "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213"
w3 = Web3(Web3.HTTPProvider(RPC))

EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
PROXY = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"
IMPL = "0x44e999d5c2f66ef0861317f9a4805ac2e90aeb4f"

# Known Polymarket proxy factory contracts on Polygon
# Source: Polymarket github + on-chain analysis
FACTORIES = [
    "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052",  # Known factory 1
    "0x6A499875cFC55E6a1F524bc052aD4eE3e8d85C2f",  # Known factory 2
    "0x72A206aaE3B9C1c2b13f76CF6e9c74d34abBa74a",  # Known factory 3
]

print(f"EOA:   {EOA}")
print(f"PROXY: {PROXY}")
print(f"IMPL:  {IMPL}")

# Approach 1: Try each factory's getProxy() call
print("\n=== Factory getProxy() calls ===")
get_proxy_sig = Web3.keccak(text="getProxy(address)")[:4]

for factory in FACTORIES:
    factory_cs = Web3.to_checksum_address(factory)
    calldata = get_proxy_sig + encode(["address"], [EOA])
    try:
        result = w3.eth.call({"to": factory_cs, "data": "0x" + calldata.hex()})
        addr = "0x" + result.hex()[-40:]
        is_match = addr.lower() == PROXY.lower()
        if addr != "0x" + "0" * 40:
            print(f"  Factory {factory}: getProxy(EOA) = {addr} {'MATCH!' if is_match else 'NO MATCH'}")
        else:
            print(f"  Factory {factory}: getProxy(EOA) = 0x0 (not registered)")
    except Exception as e:
        print(f"  Factory {factory}: ERROR - {e}")

# Approach 2: Reverse lookup - find which EOA maps to our proxy
# Try calling getProxy on a bunch of possible signers
# Actually - let's check if the proxy has a Polymarket-specific function to get its owner
print("\n=== Check proxy owner via implementation ===")

# Try calling owner() on the proxy
owner_sig = Web3.keccak(text="owner()")[:4]
try:
    result = w3.eth.call({"to": Web3.to_checksum_address(PROXY), "data": "0x" + owner_sig.hex()})
    owner = "0x" + result.hex()[-40:]
    print(f"  owner(): {owner}")
    if owner.lower() == EOA.lower():
        print("  --> MATCHES our EOA!")
    elif owner != "0x" + "0" * 40:
        print(f"  --> DIFFERENT from our EOA ({EOA})")
except Exception as e:
    print(f"  owner() reverted: {e}")

# Approach 3: Compute CREATE2 address ourselves
print("\n=== CREATE2 computation ===")

# The EIP-1167 minimal proxy bytecode for impl
# 363d3d373d3d3d363d73{impl}5af43d82803e903d91602b57fd5bf3
proxy_bytecode = bytes.fromhex(
    "363d3d373d3d3d363d73"
    + IMPL[2:].lower()
    + "5af43d82803e903d91602b57fd5bf3"
)
print(f"  Minimal proxy runtime bytecode: {proxy_bytecode.hex()}")

# The creation code wraps the runtime code
# Common pattern: 3d602d80600a3d3981f3{runtime}
creation_code = bytes.fromhex(
    "3d602d80600a3d3981f3"
    + proxy_bytecode.hex()
)
init_code_hash = Web3.keccak(creation_code)
print(f"  Init code hash: {init_code_hash.hex()}")

# Try different salt derivation methods
for factory in FACTORIES:
    factory_cs = Web3.to_checksum_address(factory)

    # Method 1: salt = keccak256(eoa_address)
    salt1 = Web3.keccak(bytes.fromhex(EOA[2:].lower().zfill(64)))
    addr1 = Web3.keccak(
        b"\xff"
        + bytes.fromhex(factory[2:])
        + salt1
        + init_code_hash
    )
    computed1 = Web3.to_checksum_address("0x" + addr1.hex()[-40:])

    # Method 2: salt = abi.encode(address) padded to 32 bytes
    salt2 = Web3.keccak(encode(["address"], [EOA]))
    addr2 = Web3.keccak(
        b"\xff"
        + bytes.fromhex(factory[2:])
        + salt2
        + init_code_hash
    )
    computed2 = Web3.to_checksum_address("0x" + addr2.hex()[-40:])

    # Method 3: salt = raw address bytes padded to 32
    salt3 = bytes(12) + bytes.fromhex(EOA[2:])  # left-pad to 32 bytes
    addr3 = Web3.keccak(
        b"\xff"
        + bytes.fromhex(factory[2:])
        + salt3
        + init_code_hash
    )
    computed3 = Web3.to_checksum_address("0x" + addr3.hex()[-40:])

    print(f"\n  Factory {factory[:10]}...:")
    print(f"    salt=keccak(eoa):       {computed1} {'MATCH!' if computed1.lower() == PROXY.lower() else ''}")
    print(f"    salt=keccak(encode):    {computed2} {'MATCH!' if computed2.lower() == PROXY.lower() else ''}")
    print(f"    salt=padded_eoa:        {computed3} {'MATCH!' if computed3.lower() == PROXY.lower() else ''}")

# Approach 4: Check via Polymarket's own API
print("\n=== Polymarket API proxy lookup ===")
try:
    # The CLOB /auth/derive-api-key might tell us something
    # Actually let's try the profile endpoint
    resp = requests.get(
        f"https://gamma-api.polymarket.com/users/{EOA.lower()}",
        timeout=10,
    )
    print(f"  Gamma API users/{EOA[:10]}...: status={resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"    Data: {data}")
except Exception as e:
    print(f"  Gamma API error: {e}")

try:
    resp = requests.get(
        f"https://gamma-api.polymarket.com/users/{PROXY.lower()}",
        timeout=10,
    )
    print(f"  Gamma API users/{PROXY[:10]}...: status={resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"    Data: {data}")
except Exception as e:
    print(f"  Gamma API error: {e}")

# Approach 5: Check proxy deployment tx via Polygonscan events
# Since V1 API is deprecated, try direct event log query
print("\n=== Proxy creation event logs ===")
# Search for Transfer events TO the proxy (first interaction)
USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
transfer_topic = Web3.keccak(text="Transfer(address,address,uint256)")
proxy_topic = "0x" + "0" * 24 + PROXY[2:].lower()

try:
    # Get earliest Transfer to the proxy (limited block range)
    # Check recent 1M blocks (~23 days)
    latest = w3.eth.block_number
    logs = w3.eth.get_logs({
        "address": USDC_E,
        "topics": [transfer_topic.hex(), None, proxy_topic],
        "fromBlock": max(0, latest - 1000000),
        "toBlock": latest,
    })
    print(f"  Found {len(logs)} USDC.e transfers TO proxy in last ~23 days")
    if logs:
        first = logs[0]
        sender = "0x" + first["topics"][1].hex()[-40:]
        amount = int(first["data"].hex(), 16) / 1e6
        print(f"    First transfer: from {sender}, amount ${amount:.2f}, block {first['blockNumber']}")
except Exception as e:
    print(f"  Event log error: {e}")
