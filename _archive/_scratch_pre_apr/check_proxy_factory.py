"""
Find the actual Polymarket proxy factory by:
1. Checking exchange contract for factory references
2. Looking at proxy creation transaction
3. Trying the Polymarket CTF Exchange proxy wallet registry
"""
import os

import dotenv
from eth_abi import decode, encode
from web3 import Web3

dotenv.load_dotenv()

RPC = "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213"
w3 = Web3(Web3.HTTPProvider(RPC))

EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
PROXY = Web3.to_checksum_address("0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8")
EXCHANGE = Web3.to_checksum_address("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E")
NEG_EXCHANGE = Web3.to_checksum_address("0xC5d563A36AE78145C45a50134d48A1215220f80a")

# 1. Check exchange contract for factory/proxyFactory references
print("=== Exchange contract function calls ===")
# Common factory-related selectors
calls = {
    "proxyFactory()": Web3.keccak(text="proxyFactory()")[:4],
    "getProxyFactory()": Web3.keccak(text="getProxyFactory()")[:4],
    "proxyWalletFactory()": Web3.keccak(text="proxyWalletFactory()")[:4],
    "safeFactory()": Web3.keccak(text="safeFactory()")[:4],
    "registry()": Web3.keccak(text="registry()")[:4],
    "getPolyProxyFactoryAddress()": Web3.keccak(text="getPolyProxyFactoryAddress()")[:4],
    "getProxyWalletAddress(address)": Web3.keccak(text="getProxyWalletAddress(address)")[:4],
    "getPolyProxyWalletAddress(address)": Web3.keccak(text="getPolyProxyWalletAddress(address)")[:4],
}

for name, sel in calls.items():
    for label, addr in [("Exchange", EXCHANGE), ("NegRisk", NEG_EXCHANGE)]:
        try:
            if "address" in name:
                data = sel + encode(["address"], [EOA])
            else:
                data = sel
            result = w3.eth.call({"to": addr, "data": "0x" + data.hex()})
            if result != b"\x00" * 32 and len(result) >= 32:
                decoded = "0x" + result.hex()[-40:]
                if decoded != "0x" + "0" * 40:
                    print(f"  {label}.{name} = {decoded}")
        except Exception:
            pass  # function doesn't exist

# 2. Check if the CLOB server has a /proxy endpoint
print("\n=== CLOB API proxy endpoints ===")
import requests

for path in [
    f"/proxy/{EOA}",
    f"/auth/proxy/{EOA}",
    f"/users/{EOA.lower()}",
    f"/profile/{EOA.lower()}",
]:
    try:
        resp = requests.get(f"https://clob.polymarket.com{path}", timeout=10)
        print(f"  GET {path}: {resp.status_code} - {resp.text[:200]}")
    except Exception as e:
        print(f"  GET {path}: ERROR - {e}")

# 3. Search for the CORRECT Polymarket proxy factory
# Polymarket uses a PolyGnosisSafeFactory at known addresses
# Actually, let's try the standard Gnosis Safe factory
print("\n=== Gnosis Safe Proxy Factory lookups ===")
GNOSIS_FACTORIES = [
    "0xa6B71E26C5e0845f74c812102Ca7114b6a896AB2",  # Gnosis Safe Proxy Factory v1.3
    "0xC22834581EbC8527d974F8a1c97E1bEA4EF910BC",  # Gnosis Safe Proxy Factory v1.3 (alt)
    "0x4e1DCf7AD4e460CfD30791CCC4F9c8a4f820ec67",  # Gnosis Safe Proxy Factory v1.4
]

for factory in GNOSIS_FACTORIES:
    factory_cs = Web3.to_checksum_address(factory)
    code = w3.eth.get_code(factory_cs)
    if len(code) > 0:
        print(f"  Factory {factory[:10]}...: {len(code)} bytes code")
    else:
        print(f"  Factory {factory[:10]}...: no code")

# 4. More importantly - let's find which address CREATED the proxy
# Check proxy transaction count
nonce = w3.eth.get_transaction_count(PROXY)
print(f"\n=== Proxy details ===")
print(f"  Nonce: {nonce}")
print(f"  Code length: {len(w3.eth.get_code(PROXY))} bytes")

# 5. Try to find proxy creation via broader event search
# ProxyCreation is the Gnosis Safe standard event
print("\n=== Searching for ProxyCreation events (Gnosis Safe pattern) ===")
proxy_creation_topic = Web3.keccak(text="ProxyCreation(address,address)")
print(f"  Topic: {proxy_creation_topic.hex()}")

# Search for events where our proxy was the first arg
proxy_padded = "0x" + "0" * 24 + PROXY.hex()[2:].lower() if isinstance(PROXY, bytes) else "0x" + "0" * 24 + PROXY[2:].lower()
try:
    latest = w3.eth.block_number
    # Try broader search - last 5M blocks (~115 days)
    logs = w3.eth.get_logs({
        "topics": [proxy_creation_topic.hex(), proxy_padded],
        "fromBlock": max(0, latest - 5000000),
        "toBlock": latest,
    })
    print(f"  Found {len(logs)} ProxyCreation events for our proxy")
    for log in logs:
        print(f"    Block: {log['blockNumber']}, Contract: {log['address']}")
        if len(log['topics']) > 1:
            print(f"    Topic1: {log['topics'][1].hex()}")
except Exception as e:
    print(f"  Search error: {e}")

# 6. Try the Polymarket-specific PolyProxyFactory
# The correct factory for Polymarket Proxy Wallets on Polygon:
# Source: Polymarket CTF exchange deployed contracts
print("\n=== Polymarket-specific proxy factory search ===")
# These are the official addresses from Polymarket's documentation/github
POLY_FACTORIES = [
    "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052",
    "0x6A499875cFC55E6a1F524bc052aD4eE3e8d85C2f",
    "0x72A206aaE3B9C1c2b13f76CF6e9c74d34abBa74a",
    "0xd08cda4b7F20E77bb6d4f8da2943FA67e8194382",  # Another known address
    "0x87d20B8e9EF72b36C4e6Cf4E4D50d5A56eBb79dA",  # Yet another
]

for factory in POLY_FACTORIES:
    factory_cs = Web3.to_checksum_address(factory)
    code = w3.eth.get_code(factory_cs)
    if len(code) > 0:
        # Try getProxy
        get_proxy_sig = Web3.keccak(text="getProxy(address)")[:4]
        calldata = get_proxy_sig + encode(["address"], [EOA])
        try:
            result = w3.eth.call({"to": factory_cs, "data": "0x" + calldata.hex()})
            addr = "0x" + result.hex()[-40:]
            if addr.lower() != "0x" + "0" * 40:
                is_match = addr.lower() == PROXY.lower()
                print(f"  {factory[:10]}...: getProxy(EOA) = {addr} {'MATCH!!!' if is_match else 'no match'}")
            else:
                print(f"  {factory[:10]}...: getProxy(EOA) = 0x0")
        except Exception:
            # Try other function names
            for fn in ["getProxyFor(address)", "computeAddress(address)", "proxies(address)"]:
                sig = Web3.keccak(text=fn)[:4]
                calldata = sig + encode(["address"], [EOA])
                try:
                    result = w3.eth.call({"to": factory_cs, "data": "0x" + calldata.hex()})
                    addr = "0x" + result.hex()[-40:]
                    if addr.lower() != "0x" + "0" * 40:
                        print(f"  {factory[:10]}...: {fn.split('(')[0]}(EOA) = {addr}")
                except Exception:
                    pass
            print(f"  {factory[:10]}...: no getProxy (has {len(code)} bytes)")
    else:
        print(f"  {factory[:10]}...: no code deployed")

# 7. Try to trace back from the proxy itself
# Check if the proxy's initialize() was called with an owner
# Storage slot 0 is often the owner in proxied contracts
print("\n=== Storage slot analysis of PROXY ===")
for i in range(5):
    raw = w3.eth.get_storage_at(PROXY, i)
    if raw != b"\x00" * 32:
        print(f"  Slot {i}: 0x{raw.hex()}")

# Check the keccak of common Polymarket storage labels
for label in ["polymarket.proxy.owner", "proxy.owner", "org.polymarket.proxy.owner"]:
    slot = Web3.keccak(text=label)
    raw = w3.eth.get_storage_at(PROXY, int(slot.hex(), 16))
    if raw != b"\x00" * 32:
        print(f"  keccak(\"{label}\"): 0x{raw.hex()}")
