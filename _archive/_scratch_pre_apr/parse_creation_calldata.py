"""
Parse the full creation TX calldata to find the real owner EOA.
The owner's address is encoded as the CREATE2 salt inside the factory call.
"""
import json
import os

from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

INFURA = "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213"
w3 = Web3(Web3.HTTPProvider(INFURA))

CREATION_TX = "0x1d1648ab72deda8e81ac9c721955983d97df83c460954637d110e86128243007"
KNOWN_PROXY = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"

tx = w3.eth.get_transaction(CREATION_TX)
input_hex = tx["input"].hex()

print(f"Full input data length: {len(input_hex)} hex chars = {len(input_hex)//2} bytes")
print()

# Parse the outer call: method(params)
method_id = input_hex[:8]
print(f"Outer method: 0x{method_id}")

# Skip method selector, parse 32-byte words
data = input_hex[8:]
words = [data[i:i+64] for i in range(0, len(data), 64)]

print(f"\nAll {len(words)} words (32-byte each):")
for i, w in enumerate(words):
    offset = i * 32
    # Try to interpret as address, uint, etc
    stripped = w.lstrip('0')
    if len(stripped) <= 40 and len(stripped) > 0:
        try:
            addr = Web3.to_checksum_address("0x" + w[-40:])
            print(f"  [{i:3d}] offset={offset:4d} 0x{offset:04x}  {w}  → addr: {addr}")
        except:
            val = int(w, 16) if w != "0" * 64 else 0
            print(f"  [{i:3d}] offset={offset:4d} 0x{offset:04x}  {w}  → uint: {val}")
    elif w == "f" * 64:
        print(f"  [{i:3d}] offset={offset:4d} 0x{offset:04x}  {w}  → MAX_UINT256")
    else:
        val = int(w, 16) if stripped else 0
        print(f"  [{i:3d}] offset={offset:4d} 0x{offset:04x}  {w}  → uint: {val}")

print()

# ============================================================
# Look for embedded addresses and check each against the factory
# ============================================================
print("=== SCANNING ALL EMBEDDED ADDRESSES ===")
from eth_abi import encode

CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
func_sig = Web3.keccak(text="getPolyProxyWalletAddress(address)")[:4]

# Extract all 20-byte values that look like addresses
found_addrs = set()
raw = bytes.fromhex(input_hex[8:])  # skip method selector

# Scan every 32-byte offset (ABI encoded)
for offset in range(0, len(raw) - 31, 32):
    chunk = raw[offset:offset+32]
    # Check if first 12 bytes are zero (address in ABI encoding)
    if chunk[:12] == b'\x00' * 12 and chunk[12:] != b'\x00' * 20:
        addr = "0x" + chunk[12:].hex()
        try:
            cs = Web3.to_checksum_address(addr)
            found_addrs.add(cs)
        except:
            pass

# Also scan unaligned (function calls inside bytes may not be 32-aligned)
for offset in range(0, len(raw) - 19):
    # Look for patterns that could be ABI-encoded addresses
    pass  # skip unaligned scan for now

print(f"Found {len(found_addrs)} unique addresses in calldata:")
for addr in sorted(found_addrs):
    print(f"  {addr}")

print(f"\nChecking which derives to {KNOWN_PROXY}:")
for addr in sorted(found_addrs):
    encoded = encode(["address"], [addr])
    calldata = func_sig + encoded
    try:
        result = w3.eth.call({"to": w3.to_checksum_address(CTF_EXCHANGE), "data": calldata.hex()})
        derived = Web3.to_checksum_address("0x" + result.hex()[-40:])
        match = " *** MATCH ***" if derived.lower() == KNOWN_PROXY.lower() else ""
        print(f"  {addr} → {derived}{match}")
    except Exception as e:
        print(f"  {addr} → error: {e}")

# ============================================================
# Also: try to find the owner by scanning for CREATE2 salt
# The proxy factory likely uses salt = keccak256(ownerAddress)
# or salt = ownerAddress padded to 32 bytes
# We can reverse-check: read the factory's createProxy function
# ============================================================
print()
print("=== FACTORY FUNCTION SIGNATURES ===")
factory_code = w3.eth.get_code(w3.to_checksum_address("0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"))

# Look for 4-byte function selectors in the calldata that match known factory methods
# From the inner call data:
# The outer contract (0xD216153c...) calls the factory
# Let me find the actual call to the factory inside the bytes param

# The dynamic bytes data starts at offset 0x120 = 288 from the start of params
# Word [2] is offset 0x120 → so it's in data[288*2:] = data[576:]
# But first, word at offset 288 has the LENGTH of the bytes
bytes_offset = 0x120  # 288
bytes_len_word = words[bytes_offset // 32]  # word at relative offset 288/32 = 9
print(f"  Word at bytes offset ({bytes_offset}): {words[bytes_offset // 32]}")
print(f"  → bytes length: {int(words[bytes_offset // 32], 16)}")

bytes_len = int(words[bytes_offset // 32], 16)
bytes_start_word = bytes_offset // 32 + 1  # word after the length

# Extract the bytes data
raw_bytes_hex = "".join(words[bytes_start_word:bytes_start_word + (bytes_len + 31) // 32])
raw_bytes = bytes.fromhex(raw_bytes_hex[:bytes_len * 2])
print(f"  Embedded bytes data ({len(raw_bytes)} bytes): {raw_bytes.hex()[:200]}...")

# This embedded bytes is what gets passed to the factory
# Parse it for function calls
if len(raw_bytes) >= 4:
    inner_method = raw_bytes[:4].hex()
    print(f"  Inner method selector: 0x{inner_method}")

    # Parse the inner bytes for addresses
    print(f"\n  Inner call data words:")
    inner_data = raw_bytes[4:]
    inner_words = [inner_data[i:i+32] for i in range(0, len(inner_data), 32)]
    for i, w_bytes in enumerate(inner_words[:20]):
        w = w_bytes.hex()
        if w_bytes[:12] == b'\x00' * 12 and w_bytes[12:] != b'\x00' * 20:
            addr = "0x" + w_bytes[12:].hex()
            try:
                addr = Web3.to_checksum_address(addr)
                print(f"    [{i}] {w}  → addr: {addr}")
            except:
                print(f"    [{i}] {w}")
        else:
            val = int.from_bytes(w_bytes, "big")
            print(f"    [{i}] {w}  → {val}")
