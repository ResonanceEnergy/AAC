"""
Verify the CLOB's proxy address derivation from our EOA.
The CLOB uses CREATE2 to compute proxy address from:
  factory + salt(EOA) + initCodeHash

If the derived address != 0xF4Ba..., that explains $0 balance.
"""
import json
import os

import requests
from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3

load_dotenv()

PK = os.getenv("POLYMARKET_PRIVATE_KEY")
FUNDER = os.getenv("POLYMARKET_FUNDER_ADDRESS")
INFURA = "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213"

w3 = Web3(Web3.HTTPProvider(INFURA))
acct = Account.from_key(PK)
EOA = acct.address

print(f"EOA:           {EOA}")
print(f"Known Proxy:   {FUNDER}")
print()

# ============================================================
# 1. Try getPolyProxyWalletAddress on the CTF Exchange
# ============================================================
print("=== CTF Exchange: getPolyProxyWalletAddress ===")
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
# Function selector for getPolyProxyWalletAddress(address)
# keccak256("getPolyProxyWalletAddress(address)") = first 4 bytes
import hashlib

from eth_abi import encode

func_sig = Web3.keccak(text="getPolyProxyWalletAddress(address)")[:4]
encoded_addr = encode(["address"], [EOA])
calldata = func_sig + encoded_addr

try:
    result = w3.eth.call({"to": w3.to_checksum_address(CTF_EXCHANGE), "data": calldata.hex()})
    derived = "0x" + result.hex()[-40:]
    print(f"  CTF Exchange derived proxy: {Web3.to_checksum_address(derived)}")
    print(f"  Matches known proxy? {derived.lower() == FUNDER.lower()}")
except Exception as e:
    print(f"  Error: {e}")
print()

# ============================================================
# 2. Try on Neg Risk CTF Exchange
# ============================================================
print("=== Neg Risk CTF Exchange: getPolyProxyWalletAddress ===")
NEG_RISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
try:
    result = w3.eth.call({"to": w3.to_checksum_address(NEG_RISK_EXCHANGE), "data": calldata.hex()})
    derived = "0x" + result.hex()[-40:]
    print(f"  NR Exchange derived proxy: {Web3.to_checksum_address(derived)}")
    print(f"  Matches known proxy? {derived.lower() == FUNDER.lower()}")
except Exception as e:
    print(f"  Error: {e}")
print()

# ============================================================
# 3. Compute CREATE2 manually with Polymarket Proxy Factory
# ============================================================
print("=== Manual CREATE2 Computation ===")
PROXY_FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"

# Get the proxy code from the known proxy (to derive init code hash)
proxy_code = w3.eth.get_code(w3.to_checksum_address(FUNDER))
print(f"  Proxy runtime code ({len(proxy_code)} bytes): {proxy_code.hex()}")

# EIP-1167 minimal proxy pattern: 363d3d373d3d3d363d73<addr>5af43d82803e903d91602b57fd5bf3
# Extract implementation address from the proxy code
if len(proxy_code) == 45:
    impl_addr = "0x" + proxy_code[10:30].hex()
    print(f"  Implementation address: {Web3.to_checksum_address(impl_addr)}")
print()

# Try different salt derivations
print("=== Trying different CREATE2 salt derivations ===")
eoa_bytes = bytes.fromhex(EOA[2:].lower())

# Salt option 1: keccak256(EOA)
salt1 = Web3.keccak(eoa_bytes)
print(f"  Salt1 (keccak(EOA)): {salt1.hex()}")

# Salt option 2: EOA padded to 32 bytes
salt2 = b'\x00' * 12 + eoa_bytes
print(f"  Salt2 (padded EOA):  {salt2.hex()}")

# Salt option 3: Just EOA as bytes32
salt3 = eoa_bytes.ljust(32, b'\x00')
print(f"  Salt3 (EOA left-padded): {salt3.hex()}")

# For CREATE2: address = keccak256(0xff ++ factory ++ salt ++ keccak256(initCode))
# But we need the INIT code (not deployed code) — it's the creation code
# Let's try to get the init code from the factory
print()
print("=== Polymarket Proxy Factory code ===")
factory_code = w3.eth.get_code(w3.to_checksum_address(PROXY_FACTORY))
print(f"  Factory code length: {len(factory_code)} bytes")
print(f"  First 100 bytes: {factory_code[:100].hex()}")
print()

# ============================================================
# 4. Try getProxyWalletAddress on the Factory itself
# ============================================================
print("=== Proxy Factory: getProxyWalletAddress ===")

# Common function selectors to try
func_names = [
    "getProxyWalletAddress(address)",
    "getProxy(address)",
    "computeProxyAddress(address)",
    "getAddress(address)",
    "walletOf(address)",
    "proxyOf(address)",
    "getWalletAddress(address)",
]

for fn in func_names:
    sel = Web3.keccak(text=fn)[:4]
    data = sel + encoded_addr
    try:
        result = w3.eth.call({"to": w3.to_checksum_address(PROXY_FACTORY), "data": data.hex()})
        if len(result) >= 20 and result != b'\x00' * 32:
            addr = "0x" + result.hex()[-40:]
            print(f"  {fn}: {Web3.to_checksum_address(addr)}")
            print(f"    Matches? {addr.lower() == FUNDER.lower()}")
    except:
        pass
print()

# ============================================================
# 5. Check the proxy factory's creation tx for init code
# ============================================================
print("=== Proxy Owner (getOwner/owner) on the proxy itself ===")
for fn in ["owner()", "getOwner()", "masterCopy()", "implementation()"]:
    sel = Web3.keccak(text=fn)[:4]
    try:
        result = w3.eth.call({"to": w3.to_checksum_address(FUNDER), "data": sel.hex()})
        if result and result != b'\x00' * 32:
            val = "0x" + result.hex()[-40:]
            print(f"  {fn}: {val}")
    except:
        pass
print()

# ============================================================
# 6. Try Gnosis Safe derivation with our EOA
# ============================================================
print("=== Gnosis Safe Factory: check derivation ===")
SAFE_FACTORY = "0xaacfeea03eb1561c4e67d661e40682bd20e3541b"
safe_factory_code = w3.eth.get_code(w3.to_checksum_address(SAFE_FACTORY))
print(f"  Safe factory code length: {len(safe_factory_code)} bytes")

# Try common proxy factory methods
for fn in ["calculateCreateProxyWithNonceAddress(address,bytes,uint256)", "proxyCreationCode()"]:
    sel = Web3.keccak(text=fn)[:4]
    print(f"  {fn}: selector={sel.hex()}")
print()

# ============================================================
# 7. Check USDC.e balance directly for BOTH EOA and proxy
# ============================================================
print("=== Direct on-chain USDC.e balances ===")
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
bal_sel = Web3.keccak(text="balanceOf(address)")[:4]

for label, addr in [("EOA", EOA), ("Known Proxy", FUNDER)]:
    data = bal_sel + encode(["address"], [Web3.to_checksum_address(addr)])
    result = w3.eth.call({"to": w3.to_checksum_address(USDC_E), "data": data.hex()})
    balance = int(result.hex(), 16)
    print(f"  {label} ({addr}): {balance / 1e6} USDC.e")

print()

# ============================================================
# 8. Scan proxy factory events to find our proxy's salt
# ============================================================
print("=== Proxy Creation Event from factory ===")
# ProxyCreation(address proxy) or similar event
# Let's look for Transfer events near where the proxy was created
# From trace, the proxy was created at some block. Let's check latest logs from factory

# Try to find the creation tx via the proxy's nonce
proxy_nonce = w3.eth.get_transaction_count(w3.to_checksum_address(FUNDER))
print(f"  Proxy nonce: {proxy_nonce}")

# Check if proxy is a contract
proxy_code_check = w3.eth.get_code(w3.to_checksum_address(FUNDER))
print(f"  Proxy is contract: {len(proxy_code_check) > 0}")
print(f"  Proxy code length: {len(proxy_code_check)} bytes")

print("\n=== SUMMARY ===")
print(f"  EOA: {EOA}")
print(f"  Known Proxy: {FUNDER}")
print("  The CLOB identifies user by POLY_ADDRESS header (=EOA),")
print("  then uses signature_type to DERIVE the proxy address.")
print("  If derivation != known proxy, balance shows $0.")
print("  Need to find what address the CLOB derives for this EOA.")
