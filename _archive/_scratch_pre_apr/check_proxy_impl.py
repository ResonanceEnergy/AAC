"""Investigate the Polymarket proxy wallet implementation contract.
Find the execute/call method to send approve transactions through the proxy."""
import json
import os
import sys
import urllib.request

from web3 import Web3

sys.path.insert(0, ".")
from shared.config_loader import load_env_file
load_env_file()

RPC = "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213"
w3 = Web3(Web3.HTTPProvider(RPC))
print(f"Connected: {w3.is_connected()}, Block: {w3.eth.block_number}")

PROXY = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"
IMPL = "0x44e999d5c2f66ef0861317f9a4805ac2e90aeb4f"  # from EIP-1167

# Get the proxy implementation's code to understand its interface
code = w3.eth.get_code(Web3.to_checksum_address(IMPL))
print(f"\nImplementation contract code: {len(code)} bytes")

# Try common proxy wallet function selectors
# exec(address to, uint256 value, bytes data) = various selectors
selectors = {
    "0x1cff79cd": "exec(address,bytes)",  # Common proxy exec
    "0xb61d27f6": "execute(address,uint256,bytes)",  # Gnosis Safe style
    "0x6a761202": "execTransaction(...)",  # Gnosis Safe
    "0xd460f0a2": "execute_0(address,uint256,bytes)",
    "0x0000189a": "getOwner()",
    "0x8da5cb5b": "owner()",
    "0xee97f7f3": "executor()",
    "0xa7f43779": "masterCopy()",
    "0x3659cfe6": "upgradeTo(address)",
    "0x5c60da1b": "implementation()",
}

print(f"\nProbing implementation contract functions:")
for sel, name in selectors.items():
    try:
        result = w3.eth.call({"to": Web3.to_checksum_address(IMPL), "data": sel})
        print(f"  {name} ({sel}): {result.hex()}")
    except Exception as e:
        err_str = str(e)
        if "revert" in err_str.lower():
            print(f"  {name} ({sel}): reverted (function exists but failed)")
        elif "out of gas" in err_str.lower():
            print(f"  {name} ({sel}): out of gas (function likely exists)")
        else:
            print(f"  {name} ({sel}): {err_str[:80]}")

# Also probe the PROXY directly (delegates to impl)
print(f"\nProbing PROXY contract functions:")
for sel, name in selectors.items():
    try:
        result = w3.eth.call({"to": Web3.to_checksum_address(PROXY), "data": sel})
        print(f"  {name} ({sel}): {result.hex()}")
    except Exception as e:
        err_str = str(e)
        if "revert" in err_str.lower():
            print(f"  {name} ({sel}): reverted")
        else:
            print(f"  {name} ({sel}): {err_str[:80]}")

# Try to get the Polymarket proxy wallet ABI from Polygonscan
print("\n\nChecking known Polymarket proxy wallet signatures...")
# The Polymarket proxy wallet typically has:
# - exec(address,bytes) for arbitrary calls
# - execMeta(address,bytes,bytes) for meta-transactions
# Let's try the most common: exec(address to, bytes calldata)

# Build an exec call to check USDC.e balance through the proxy
# This is just to verify exec works, not to approve
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
# balanceOf(proxy) selector + arg
balance_calldata = "0x70a08231" + PROXY[2:].lower().zfill(64)

# exec(USDC_E, balanceOf_calldata) 
# Selector: 0x1cff79cd = exec(address,bytes)
exec_data = (
    "0x1cff79cd"
    + USDC_E[2:].lower().zfill(64)  # address to
    + "0000000000000000000000000000000000000000000000000000000000000040"  # bytes offset
    + "0000000000000000000000000000000000000000000000000000000000000024"  # bytes length (36)
    + balance_calldata[2:]  # the actual calldata without 0x
    + "00000000000000000000000000000000000000000000000000000000"  # padding
)
try:
    result = w3.eth.call({"to": Web3.to_checksum_address(PROXY), "data": exec_data})
    print(f"exec(USDC_E, balanceOf) result: {result.hex()}")
except Exception as e:
    print(f"exec(USDC_E, balanceOf) error: {e}")

# Try Etherscan V2 for the implementation ABI
print("\nTrying Etherscan V2 for ABI...")
api_key = "KK12YHTrpB24R9mwdD3u25rC6ePnCVw8"
for label, addr in [("IMPL", IMPL), ("PROXY", PROXY)]:
    url = f"https://api.etherscan.io/v2/api?chainid=137&module=contract&action=getabi&address={addr}&apikey={api_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        status = data.get("status")
        result = data.get("result", "")
        if status == "1":
            abi = json.loads(result)
            print(f"\n{label} ABI ({len(abi)} functions):")
            for item in abi:
                if item.get("type") == "function":
                    name = item.get("name", "?")
                    inputs = ",".join(i.get("type", "?") for i in item.get("inputs", []))
                    print(f"  {name}({inputs})")
        else:
            print(f"\n{label}: {data.get('message', '?')} - {result[:200]}")
    except Exception as e:
        print(f"\n{label}: Error {e}")
