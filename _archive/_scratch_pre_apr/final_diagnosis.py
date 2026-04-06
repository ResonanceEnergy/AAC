"""
FINAL ANSWER: The real owner of the $556 proxy.

getPolyProxyWalletAddress(0x6e9b70D1...) = 0xF4Ba... (our proxy with $556.39)
getPolyProxyWalletAddress(0x4BFC40EA...) = 0xB3A0... (empty, undeployed)

The private key in .env (0x9a9f...) → EOA 0x4BFC40EA → proxy 0xB3A0... ($0)
The REAL owner of 0xF4Ba... ($556) → EOA 0x6e9b70D1 → DIFFERENT private key
"""
import os

import requests
from dotenv import load_dotenv
from eth_abi import encode
from web3 import Web3

load_dotenv()

INFURA = "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213"
w3 = Web3(Web3.HTTPProvider(INFURA))
ETHERSCAN_KEY = "U8HCAUW68ZXNXSMEZGNGKIMWEA4U4VX9YM"

REAL_OWNER = "0x6e9b70D1175ecA144111743503441300A9494297"
OUR_EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"

print(f"OUR EOA:     {OUR_EOA}")
print(f"REAL OWNER:  {REAL_OWNER}")
print()

# 1. Is the real owner an EOA or contract?
code = w3.eth.get_code(w3.to_checksum_address(REAL_OWNER))
print(f"Real owner code length: {len(code)} bytes")
if len(code) == 0:
    print("  → This is an EOA (externally owned account)")
    print("  → Need to find the private key for this address")
else:
    print("  → This is a CONTRACT")
    print(f"  → Code: {code.hex()[:100]}...")

# 2. Check balances
bal = w3.eth.get_balance(w3.to_checksum_address(REAL_OWNER))
print(f"Real owner MATIC: {bal / 1e18}")

USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
bal_sel = Web3.keccak(text="balanceOf(address)")[:4]
data = bal_sel + encode(["address"], [w3.to_checksum_address(REAL_OWNER)])
result = w3.eth.call({"to": w3.to_checksum_address(USDC_E), "data": data.hex()})
usdc = int(result.hex(), 16)
print(f"Real owner USDC.e: {usdc / 1e6}")

# 3. Check nonce
nonce = w3.eth.get_transaction_count(w3.to_checksum_address(REAL_OWNER))
print(f"Real owner nonce: {nonce}")
print()

# 4. Check Etherscan tx history
print("=== REAL OWNER TX HISTORY ===")
try:
    url = (
        f"https://api.etherscan.io/v2/api?chainid=137"
        f"&module=account&action=txlist"
        f"&address={REAL_OWNER}"
        f"&startblock=0&endblock=99999999"
        f"&sort=desc&page=1&offset=5&apikey={ETHERSCAN_KEY}"
    )
    r = requests.get(url, timeout=15)
    data = r.json()
    if data.get("result"):
        for tx in data["result"]:
            print(f"  TX: {tx.get('hash', '')[:20]}...")
            print(f"    From: {tx.get('from')}")
            print(f"    To: {tx.get('to')}")
            print(f"    Block: {tx.get('blockNumber')}")
    else:
        print(f"  No txs: {data.get('message')}")
except Exception as e:
    print(f"  Error: {e}")
print()

# 5. Also check on Ethereum mainnet
print("=== REAL OWNER ON ETHEREUM MAINNET ===")
ETH_INFURA = "https://mainnet.infura.io/v3/84842078b09946638c03157f83405213"
w3_eth = Web3(Web3.HTTPProvider(ETH_INFURA))
eth_nonce = w3_eth.eth.get_transaction_count(w3_eth.to_checksum_address(REAL_OWNER))
eth_bal = w3_eth.eth.get_balance(w3_eth.to_checksum_address(REAL_OWNER))
eth_code = w3_eth.eth.get_code(w3_eth.to_checksum_address(REAL_OWNER))
print(f"  Code: {len(eth_code)} bytes")
print(f"  ETH balance: {eth_bal / 1e18}")
print(f"  Nonce: {eth_nonce}")
print()

print("=" * 60)
print("CONCLUSION")
print("=" * 60)
print(f"""
The private key in .env (0x9a9f...c113) controls EOA:
  {OUR_EOA}
  
Its Polymarket proxy would be at:
  0xB3A0E41056a09FA388Db2FAcd6044646a4F2f057
  (not yet deployed, $0 balance)

The proxy with $556.39 USDC.e at:
  0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8
  
Is owned by a DIFFERENT EOA:
  {REAL_OWNER}
  
This is a DIFFERENT private key. The key in .env cannot
access these funds through Polymarket.

ACTION NEEDED:
  1. Check your Polymarket account settings/profile for the 
     correct wallet address
  2. If you signed up with Magic Link (email/Google), the 
     real private key may be in your Magic Link export
  3. The address {REAL_OWNER}
     may be your Magic Link wallet from signup
  4. OR deposit new funds into your actual proxy at  
     0xB3A0E41056a09FA388Db2FAcd6044646a4F2f057
""")
