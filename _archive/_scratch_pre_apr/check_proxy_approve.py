"""Proxy wallet: call functions with from=EOA and approve USDC.e for exchange."""
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
print(f"Connected: {w3.is_connected()}")

PK = os.getenv("POLYMARKET_PRIVATE_KEY", "")
acct = Account.from_key(PK)
EOA = acct.address
PROXY = Web3.to_checksum_address("0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8")
USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")

EXCHANGE = Web3.to_checksum_address("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E")
NEG_RISK_EXCHANGE = Web3.to_checksum_address("0xC5d563A36AE78145C45a50134d48A1215220f80a")
NEG_RISK_ADAPTER = Web3.to_checksum_address("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296")

print(f"EOA: {EOA}")
print(f"PROXY: {PROXY}")

# Check proxy owner() with from=EOA
print("\n--- Calling proxy functions with from=EOA ---")
selectors = {
    "owner()": "0x8da5cb5b",
    "implementation()": "0x5c60da1b",
}

for name, sel in selectors.items():
    try:
        result = w3.eth.call({"from": EOA, "to": PROXY, "data": sel})
        if len(result) >= 20:
            addr = "0x" + result[-20:].hex()
            print(f"  {name} = {addr}")
        else:
            print(f"  {name} = {result.hex()}")
    except Exception as e:
        print(f"  {name}: {e}")

# Check USDC.e allowance from proxy to exchange contracts
print("\n--- Current USDC.e Allowances from PROXY ---")
usdc_abi = [
    {"constant": True, "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}], "name": "allowance", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
    {"constant": True, "inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
    {"constant": False, "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "approve", "outputs": [{"name": "", "type": "bool"}], "type": "function"},
]
usdc = w3.eth.contract(address=USDC_E, abi=usdc_abi)

balance = usdc.functions.balanceOf(PROXY).call()
print(f"  USDC.e Balance: ${balance / 1e6:.2f}")

for name, spender in [("Exchange", EXCHANGE), ("NegRisk Exchange", NEG_RISK_EXCHANGE), ("NegRisk Adapter", NEG_RISK_ADAPTER)]:
    allowance = usdc.functions.allowance(PROXY, spender).call()
    print(f"  Allowance to {name}: ${allowance / 1e6:.2f}")

# Now try to build the approve transaction through the proxy
# The proxy has exec(address to, bytes data) function
# We need to call: PROXY.exec(USDC_E, encode(approve(EXCHANGE, MAX_UINT)))

print("\n--- Building Approve Transactions ---")

# Polymarket proxy ABI (exec function)
proxy_abi = [
    {"inputs": [{"name": "to", "type": "address"}, {"name": "data", "type": "bytes"}], "name": "exec", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [], "name": "owner", "outputs": [{"name": "", "type": "address"}], "stateMutability": "view", "type": "function"},
]
proxy = w3.eth.contract(address=PROXY, abi=proxy_abi)

# Call owner() to verify
try:
    owner = proxy.functions.owner().call({"from": EOA})
    print(f"  Proxy owner: {owner}")
    print(f"  Our EOA:     {EOA}")
    print(f"  Match: {owner.lower() == EOA.lower()}")
except Exception as e:
    print(f"  owner() call failed: {e}")

# Encode the approve call: approve(spender, type(uint256).max)
MAX_UINT = 2**256 - 1
approve_data = usdc.encodeABI(fn_name="approve", args=[EXCHANGE, MAX_UINT])
print(f"\n  Approve calldata: {approve_data[:40]}...")

# Simulate the exec call (dry run via eth_call)
print("\n--- DRY RUN: Simulating exec(USDC_E, approve(EXCHANGE, MAX)) ---")
try:
    exec_data = proxy.encodeABI(fn_name="exec", args=[USDC_E, bytes.fromhex(approve_data[2:])])
    result = w3.eth.call({"from": EOA, "to": PROXY, "data": exec_data})
    print(f"  Simulation SUCCESS! Result: {result.hex()}")
    print("  The approve transaction would work!")
except Exception as e:
    print(f"  Simulation FAILED: {e}")

# Check gas estimate
print("\n--- Gas Estimates ---")
for name, spender in [("Exchange", EXCHANGE), ("NegRisk Exchange", NEG_RISK_EXCHANGE), ("NegRisk Adapter", NEG_RISK_ADAPTER)]:
    try:
        approve_data_inner = usdc.encodeABI(fn_name="approve", args=[spender, MAX_UINT])
        exec_data = proxy.encodeABI(fn_name="exec", args=[USDC_E, bytes.fromhex(approve_data_inner[2:])])
        gas = w3.eth.estimate_gas({"from": EOA, "to": PROXY, "data": exec_data})
        print(f"  Approve {name}: {gas} gas")
    except Exception as e:
        print(f"  Approve {name}: {e}")

# Check EOA balance for gas
eoa_balance = w3.eth.get_balance(EOA)
print(f"\n  EOA MATIC balance (for gas): {eoa_balance / 1e18:.8f} MATIC")
gas_price = w3.eth.gas_price
print(f"  Current gas price: {gas_price / 1e9:.2f} Gwei")
if eoa_balance > 0:
    est_cost_3_approvals = (3 * 60000 * gas_price) / 1e18
    print(f"  Estimated cost for 3 approvals: ~{est_cost_3_approvals:.6f} MATIC")
    print(f"  Affordable: {eoa_balance > 3 * 60000 * gas_price}")
else:
    print("  *** EOA has 0 MATIC - cannot pay gas for approval transactions! ***")
    print("  *** You need to send some MATIC to the EOA for gas ***")
