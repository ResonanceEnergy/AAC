"""Verify: Exchange.getPolyProxyWalletAddress(EOA) vs PROXY_IN_ENV."""
import os
import dotenv
from web3 import Web3
from eth_abi import encode

dotenv.load_dotenv()

RPC = "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213"
w3 = Web3(Web3.HTTPProvider(RPC))

EXCHANGE = Web3.to_checksum_address("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E")
EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
PROXY_IN_ENV = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"

# Get the deterministic proxy for our EOA per the Exchange contract
sig = Web3.keccak(text="getPolyProxyWalletAddress(address)")[:4]
calldata = sig + encode(["address"], [EOA])
result = w3.eth.call({"to": EXCHANGE, "data": "0x" + calldata.hex()})
computed_proxy = Web3.to_checksum_address("0x" + result.hex()[-40:])

print(f"Our EOA:          {EOA}")
print(f"Proxy in .env:    {PROXY_IN_ENV}")
print(f"Exchange says:    {computed_proxy}")
print(f"Match: {computed_proxy.lower() == PROXY_IN_ENV.lower()}")
print()

# Check balances on both
USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
bal_sig = Web3.keccak(text="balanceOf(address)")[:4]

for label, addr in [("Computed proxy", computed_proxy), ("Proxy in .env", PROXY_IN_ENV)]:
    addr_cs = Web3.to_checksum_address(addr)
    calldata = bal_sig + encode(["address"], [addr_cs])
    result = w3.eth.call({"to": USDC_E, "data": "0x" + calldata.hex()})
    balance = int(result.hex(), 16) / 1e6
    code = w3.eth.get_code(addr_cs)
    nonce = w3.eth.get_transaction_count(addr_cs)
    print(f"{label} ({addr}):")
    print(f"  USDC.e: ${balance:.2f}")
    print(f"  Code:   {len(code)} bytes")
    print(f"  Nonce:  {nonce}")
    print()

# Now: what if we USE the computed proxy as funder instead?
# First check if it's deployed (has code)
computed_code = w3.eth.get_code(Web3.to_checksum_address(computed_proxy))
if len(computed_code) == 0:
    print("*** Computed proxy has NO code - it needs to be DEPLOYED first! ***")
    print("The Polymarket docs say: 'These can be deterministically derived or")
    print("you can deploy them on behalf of the user.'")
    print("This proxy was never deployed because this EOA never logged into Polymarket.")
    print()
    print("OPTIONS:")
    print("1. Update POLYMARKET_FUNDER_ADDRESS to the computed proxy address")
    print("2. Deploy the proxy via the factory contract")
    print("3. Deposit USDC.e to the computed proxy address")
else:
    print(f"Computed proxy IS deployed ({len(computed_code)} bytes)")

# Also check the safe factory 
SAFE_FACTORY = Web3.to_checksum_address("0xaacfeea03eb1561c4e67d661e40682bd20e3541b")
safe_code = w3.eth.get_code(SAFE_FACTORY)
print(f"\nSafe factory: {len(safe_code)} bytes code")
