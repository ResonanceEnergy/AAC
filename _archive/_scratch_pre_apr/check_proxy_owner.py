"""Check who actually owns/controls the PROXY wallet."""
import json
import os
import sys
from web3 import Web3
from eth_account import Account

sys.path.insert(0, ".")
from shared.config_loader import load_env_file
load_env_file()

RPC = "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213"
w3 = Web3(Web3.HTTPProvider(RPC))

PK = os.getenv("POLYMARKET_PRIVATE_KEY", "")
acct = Account.from_key(PK)
EOA = acct.address
PROXY = Web3.to_checksum_address("0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8")

print(f"EOA (from private key): {EOA}")
print(f"PROXY (from .env): {PROXY}")

# The proxy is EIP-1167 -> impl at 0x44e999d5c2f66ef0861317f9a4805ac2e90aeb4f
IMPL = Web3.to_checksum_address("0x44e999d5c2f66ef0861317f9a4805ac2e90aeb4f")

# Known Polymarket Proxy Factory addresses
PROXY_FACTORIES = [
    "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052",  # Main factory
    "0x3Ba7A5bA43Facf7c19E8bA7bA92ceE3286Aaeeb1",  # Alt factory
    "0x76d04B1E67a57a5a5d53FBe5aDC5B1AE1D0B6D84",  # Another
]

# Check proxy factory getProxy(address)
print("\n--- Proxy Factory Lookups ---")
getProxy_selector = "0x" + Web3.keccak(text="getProxy(address)")[:4].hex()
print(f"getProxy selector: {getProxy_selector}")

for factory_addr in PROXY_FACTORIES:
    factory = Web3.to_checksum_address(factory_addr)
    padded_eoa = EOA[2:].lower().zfill(64)
    calldata = getProxy_selector + padded_eoa
    try:
        result = w3.eth.call({"to": factory, "data": calldata})
        if result and len(result) >= 20:
            derived = Web3.to_checksum_address("0x" + result[-20:].hex())
            match = derived == PROXY
            print(f"  Factory {factory_addr[:12]}... getProxy(EOA) = {derived} {'MATCH!' if match else 'NO MATCH'}")
        else:
            print(f"  Factory {factory_addr[:12]}...: empty result")
    except Exception as e:
        print(f"  Factory {factory_addr[:12]}...: {str(e)[:80]}")

# Reverse: Try to find who deployed/owns the PROXY
# Check creation tx of the PROXY contract
print("\n--- PROXY Creation Check ---")

# The proxy at 0xF4Ba has nonce=1, meaning it has sent 1 tx or was created by a factory
# Let's check the storage slot 0 which typically holds the owner for Polymarket proxies
for slot in range(5):
    raw = w3.eth.get_storage_at(PROXY, slot)
    if raw != b'\x00' * 32:
        hex_val = raw.hex()
        # Check if it looks like an address (20 bytes, usually right-aligned)
        if hex_val[:24] == "0" * 24:  # right-aligned address
            addr = "0x" + hex_val[24:]
            print(f"  Slot {slot}: {addr} {'== EOA!' if addr.lower() == EOA.lower() else ''}")
        else:
            print(f"  Slot {slot}: 0x{hex_val}")
    else:
        print(f"  Slot {slot}: empty")

# Also check the implementation's storage layout
print("\n--- Implementation Storage Probe ---")
for slot in range(5):
    raw = w3.eth.get_storage_at(IMPL, slot)
    if raw != b'\x00' * 32:
        hex_val = raw.hex()
        print(f"  IMPL Slot {slot}: 0x{hex_val}")

# Try Polymarket's known proxy resolution API
print("\n--- Polymarket Proxy Resolution ---")
import urllib.request

# The Polymarket CLOB server has a mapping. Let's check via proxy resolution
# Try registering the funder with the CLOB
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

client = ClobClient(
    "https://clob.polymarket.com",
    key=PK,
    chain_id=137,
    signature_type=1,
    funder=PROXY.lower(),
)
creds = client.create_or_derive_api_creds()
client.set_api_creds(creds)

# What does the CLOB think our address is?
print(f"  CLOB client address: {client.get_address()}")
print(f"  CLOB builder funder: {client.builder.funder}")
print(f"  CLOB builder sig_type: {client.builder.sig_type}")

# Check if there's a register_proxy or similar 
import inspect
all_attrs = [m for m in dir(client) if not m.startswith('_')]
proxy_methods = [m for m in all_attrs if any(kw in m.lower() for kw in ['register', 'proxy', 'fund', 'map'])]
print(f"  Proxy-related methods: {proxy_methods}")

# Try to see what the CLOB server knows about our addresses
print("\n--- CLOB Data Endpoints ---")
for endpoint in ["/auth/api-keys", "/auth/api-key"]:
    try:
        full_url = f"https://clob.polymarket.com{endpoint}"
        from py_clob_client.headers.headers import create_level_2_headers
        from py_clob_client.signing.model import RequestArgs
        req_args = RequestArgs(method="GET", request_path=endpoint)
        headers = create_level_2_headers(client.signer, client.creds, req_args)
        req = urllib.request.Request(full_url, headers={**headers, "User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        print(f"  {endpoint}: {json.dumps(data)[:300]}")
    except Exception as e:
        print(f"  {endpoint}: {e}")

print("\nDONE.")
