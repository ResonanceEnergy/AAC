"""
ROOT CAUSE CONFIRMED: Address mismatch.

CTF Exchange derives proxy 0xB3A0... for our EOA, but funds are at 0xF4Ba...
These are DIFFERENT addresses. Let's investigate both.
"""
import json
import os

import requests
from dotenv import load_dotenv
from eth_abi import encode
from eth_account import Account
from web3 import Web3

load_dotenv()

PK = os.getenv("POLYMARKET_PRIVATE_KEY")
INFURA = "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213"
w3 = Web3(Web3.HTTPProvider(INFURA))
acct = Account.from_key(PK)
EOA = acct.address

KNOWN_PROXY = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"
DERIVED_PROXY = "0xB3A0E41056a09FA388Db2FAcd6044646a4F2f057"

print(f"EOA:            {EOA}")
print(f"Known Proxy:    {KNOWN_PROXY} (has $556.39 USDC.e)")
print(f"Derived Proxy:  {DERIVED_PROXY} (what CLOB expects)")
print()

# ============================================================
# 1. Check derived proxy address state
# ============================================================
print("=== DERIVED PROXY (0xB3A0...) ===")
code = w3.eth.get_code(w3.to_checksum_address(DERIVED_PROXY))
print(f"  Code length: {len(code)} bytes (0 = not deployed)")
bal = w3.eth.get_balance(w3.to_checksum_address(DERIVED_PROXY))
print(f"  MATIC balance: {bal / 1e18}")

USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
bal_sel = Web3.keccak(text="balanceOf(address)")[:4]
data = bal_sel + encode(["address"], [w3.to_checksum_address(DERIVED_PROXY)])
result = w3.eth.call({"to": w3.to_checksum_address(USDC_E), "data": data.hex()})
usdc_bal = int(result.hex(), 16)
print(f"  USDC.e balance: {usdc_bal / 1e6}")
print()

# ============================================================
# 2. Check known proxy — who is the actual owner?
# ============================================================
print("=== KNOWN PROXY (0xF4Ba... — has $556 USDC.e) ===")
# The proxy is an EIP-1167 minimal proxy — it delegates to implementation
# The implementation stores the owner in its storage
# Let's read storage slots
for slot in range(5):
    val = w3.eth.get_storage_at(w3.to_checksum_address(KNOWN_PROXY), slot)
    if val != b'\x00' * 32:
        print(f"  Slot {slot}: {val.hex()}")
    else:
        print(f"  Slot {slot}: empty")

# Try calling functions on the proxy
for fn in ["owner()", "getOwner()", "getOwners()", "isOwner(address)"]:
    sel = Web3.keccak(text=fn)[:4]
    if "address" in fn:
        calldata = sel + encode(["address"], [w3.to_checksum_address(EOA)])
    else:
        calldata = sel
    try:
        result = w3.eth.call({"to": w3.to_checksum_address(KNOWN_PROXY), "data": calldata.hex()})
        if result and result != b'\x00' * 32:
            print(f"  {fn}: 0x{result.hex()}")
    except Exception as e:
        print(f"  {fn}: reverted — {str(e)[:80]}")
print()

# ============================================================
# 3. Find WHICH EOA maps to 0xF4Ba... (reverse lookup)
# ============================================================
print("=== REVERSE LOOKUP: Who owns 0xF4Ba...? ===")
print("  Checking recent Etherscan V2 API for creation TX...")

ETHERSCAN_KEY = "U8HCAUW68ZXNXSMEZGNGKIMWEA4U4VX9YM"

# Get internal transactions that created the proxy
try:
    url = (
        f"https://api.etherscan.io/v2/api?chainid=137"
        f"&module=account&action=txlistinternal"
        f"&address={KNOWN_PROXY}"
        f"&startblock=0&endblock=99999999"
        f"&sort=asc&apikey={ETHERSCAN_KEY}"
    )
    r = requests.get(url, timeout=15)
    data = r.json()
    if data.get("result"):
        for tx in data["result"][:5]:
            print(f"  TX: {tx.get('hash')}")
            print(f"    From: {tx.get('from')}")
            print(f"    To: {tx.get('to')}")
            print(f"    Type: {tx.get('type')}")
            print(f"    Block: {tx.get('blockNumber')}")
    else:
        print(f"  No internal txs found: {data.get('message')}")
except Exception as e:
    print(f"  Error: {e}")
print()

# Get normal transactions for the proxy
try:
    url = (
        f"https://api.etherscan.io/v2/api?chainid=137"
        f"&module=account&action=txlist"
        f"&address={KNOWN_PROXY}"
        f"&startblock=0&endblock=99999999"
        f"&sort=asc&page=1&offset=10&apikey={ETHERSCAN_KEY}"
    )
    r = requests.get(url, timeout=15)
    data = r.json()
    if data.get("result"):
        print(f"  Normal txs for proxy ({len(data['result'])} found):")
        for tx in data["result"][:5]:
            print(f"    TX: {tx.get('hash')}")
            print(f"      From: {tx.get('from')}")
            print(f"      To: {tx.get('to')}")
            print(f"      Method: {tx.get('methodId')}")
    else:
        print(f"  No normal txs: {data.get('message')}")
except Exception as e:
    print(f"  Error: {e}")
print()

# ============================================================
# 4. Critical question: deploy the DERIVED proxy?
# ============================================================
print("=== KEY QUESTION ===")
print(f"""
ROOT CAUSE: The CLOB derives proxy address 0xB3A0... for our EOA.
But the $556.39 USDC.e sits at a DIFFERENT proxy 0xF4Ba...

This means EITHER:
  (a) The private key 0x9a9f... doesn't own 0xF4Ba...
      → A different private key owns it
  (b) 0xF4Ba... was created by a different mechanism
      that doesn't match the standard derivation

The derived proxy 0xB3A0... is our EOA's CORRECT proxy.
It's currently empty ({usdc_bal / 1e6} USDC.e).

OPTIONS:
  1. Move funds FROM 0xF4Ba... to 0xB3A0... (need the real owner key)
  2. Find the REAL private key that owns 0xF4Ba...
  3. Deposit fresh funds into 0xB3A0... via Polymarket bridge
""")
