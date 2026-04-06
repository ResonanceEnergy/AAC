"""
Decode the creation TX of 0xF4Ba... to find the REAL owner EOA.
TX: 0x1d1648ab72deda8e81ac9c721955983d97df83c460954637d110e86128243007
"""
import json
import os

import requests
from dotenv import load_dotenv
from eth_abi import decode
from web3 import Web3

load_dotenv()

INFURA = "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213"
w3 = Web3(Web3.HTTPProvider(INFURA))
ETHERSCAN_KEY = "U8HCAUW68ZXNXSMEZGNGKIMWEA4U4VX9YM"

CREATION_TX = "0x1d1648ab72deda8e81ac9c721955983d97df83c460954637d110e86128243007"
KNOWN_PROXY = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"
PROXY_FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"

# ============================================================
# 1. Get the full creation transaction
# ============================================================
print("=== CREATION TRANSACTION ===")
tx = w3.eth.get_transaction(CREATION_TX)
print(f"  From: {tx['from']}")
print(f"  To:   {tx.get('to')}")
print(f"  Block: {tx['blockNumber']}")
print(f"  Input data length: {len(tx['input'])} bytes")
print(f"  Input (first 200 chars): {tx['input'].hex()[:200]}")
print()

# Decode the input data
input_data = tx["input"].hex()
print(f"  Full input data:")
print(f"    Method selector: {input_data[:8]}")

# Common method selectors on the proxy factory:
# multicall(bytes[]) = ac9650d8
# createProxy(address,bytes) = various
# Try to decode
method_id = input_data[:8]
print(f"  Method ID: 0x{method_id}")

# Try to decode the rest as ABI-encoded data
remaining = bytes.fromhex(input_data[8:])
print(f"  Remaining data ({len(remaining)} bytes): {remaining[:200].hex()}")
print()

# ============================================================
# 2. Get the transaction receipt with logs
# ============================================================
print("=== TRANSACTION RECEIPT & LOGS ===")
receipt = w3.eth.get_transaction_receipt(CREATION_TX)
print(f"  Status: {receipt['status']} ({'success' if receipt['status'] == 1 else 'failed'})")
print(f"  Logs: {len(receipt['logs'])} events")

for i, log in enumerate(receipt["logs"]):
    print(f"\n  --- Log {i} ---")
    print(f"    Address: {log['address']}")
    print(f"    Topics: {[t.hex() for t in log['topics']]}")
    if log["data"]:
        data_hex = log["data"].hex() if isinstance(log["data"], bytes) else log["data"]
        print(f"    Data: {data_hex[:200]}")
        # If data contains an address, extract it
        if len(data_hex) >= 64:
            potential_addr = "0x" + data_hex[24:64]
            if potential_addr != "0x" + "0" * 40:
                try:
                    print(f"    → Potential address in data: {Web3.to_checksum_address(potential_addr)}")
                except:
                    pass
print()

# ============================================================
# 3. Check the relay TX that called the factory (parent TX)
# ============================================================
print("=== RELAY TRANSACTION ANALYSIS ===")
from_addr = tx["from"]
print(f"  TX sender (relay): {from_addr}")

# Check if the sender is a contract (relay)
relay_code = w3.eth.get_code(w3.to_checksum_address(from_addr))
print(f"  Relay is contract: {len(relay_code) > 0}")
print()

# ============================================================
# 4. Look at Etherscan for decoded tx
# ============================================================
print("=== ETHERSCAN TX DECODE ===")
try:
    url = (
        f"https://api.etherscan.io/v2/api?chainid=137"
        f"&module=proxy&action=eth_getTransactionByHash"
        f"&txhash={CREATION_TX}"
        f"&apikey={ETHERSCAN_KEY}"
    )
    r = requests.get(url, timeout=15)
    data = r.json()
    if data.get("result"):
        tx_data = data["result"]
        print(f"  From: {tx_data.get('from')}")
        print(f"  To: {tx_data.get('to')}")
        print(f"  Input: {tx_data.get('input', '')[:200]}...")
    else:
        print(f"  No result: {data}")
except Exception as e:
    print(f"  Error: {e}")
print()

# ============================================================
# 5. Try to find the ACTUAL owner by brute-force checking
#    proxy factory's derivation with known addresses from logs
# ============================================================
print("=== FIND REAL OWNER ===")
# Extract all addresses from tx logs
addrs_to_check = set()
for log in receipt["logs"]:
    for topic in log["topics"]:
        if len(topic) == 32:
            addr = "0x" + topic.hex()[-40:]
            if addr != "0x" + "0" * 40:
                try:
                    addrs_to_check.add(Web3.to_checksum_address(addr))
                except:
                    pass
    if log["data"]:
        data_hex = log["data"].hex() if isinstance(log["data"], bytes) else log["data"]
        # Scan for 20-byte aligned addresses in data
        for offset in range(0, len(data_hex) - 40, 64):
            chunk = data_hex[offset:offset + 64]
            if chunk[:24] == "0" * 24:
                addr = "0x" + chunk[24:]
                if addr != "0x" + "0" * 40:
                    try:
                        addrs_to_check.add(Web3.to_checksum_address(addr))
                    except:
                        pass

print(f"  Addresses found in logs: {len(addrs_to_check)}")
for addr in addrs_to_check:
    print(f"    {addr}")

# Now check each address: does getPolyProxyWalletAddress(addr) == 0xF4Ba...?
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
func_sig = Web3.keccak(text="getPolyProxyWalletAddress(address)")[:4]
from eth_abi import encode as abi_encode

print(f"\n  Checking which address derives to {KNOWN_PROXY}:")
for addr in addrs_to_check:
    encoded = abi_encode(["address"], [addr])
    calldata = func_sig + encoded
    try:
        result = w3.eth.call({"to": w3.to_checksum_address(CTF_EXCHANGE), "data": calldata.hex()})
        derived = "0x" + result.hex()[-40:]
        derived_cs = Web3.to_checksum_address(derived)
        if derived_cs.lower() == KNOWN_PROXY.lower():
            print(f"    *** MATCH: {addr} → {derived_cs} ***")
        else:
            print(f"    {addr} → {derived_cs}")
    except Exception as e:
        print(f"    {addr} → error: {e}")
print()

# ============================================================
# 6. Also decode the method call on the factory
# ============================================================
print("=== DECODE FACTORY METHOD CALL ===")
# The TX goes TO the proxy factory
# Method ID tells us which function was called
# Let's try common function signatures
known_sigs = {
    "ec9b5578": "createProxy(address)",
    "1688f0b9": "createProxyWithNonce(address,bytes,uint256)",
    "61b69abd": "createProxy(address,bytes)",
    "85a5affe": "multiSendCalldata(bytes)",
    "ac9650d8": "multicall(bytes[])",
    "8d80ff0a": "multiSend(bytes)",
    "ee22610b": "createProxyWithCallback(address,bytes,uint256,address)",
}

print(f"  Method ID: 0x{method_id}")
if method_id in known_sigs:
    print(f"  → {known_sigs[method_id]}")
else:
    print(f"  → Unknown method. Checking ABI...")
    # Try to decode as createProxy(address,bytes)
    try:
        decoded = decode(["address", "bytes"], remaining)
        print(f"  Decoded as (address, bytes): {decoded[0]}, data_len={len(decoded[1])}")
    except:
        pass
    try:
        decoded = decode(["address"], remaining[:32])
        print(f"  First param as address: {decoded[0]}")
    except:
        pass
