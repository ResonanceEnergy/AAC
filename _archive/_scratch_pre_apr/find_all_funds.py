"""Find all funds across all chains for our EOA."""
from web3 import Web3
from eth_abi import encode, decode
import json
import urllib.request

EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"

RPCS = {
    "Ethereum": "https://mainnet.infura.io/v3/84842078b09946638c03157f83405213",
    "Polygon": "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213",
    "Arbitrum": "https://arb1.arbitrum.io/rpc",
    "Optimism": "https://mainnet.optimism.io",
    "Base": "https://mainnet.base.org",
}

# USDC/USDT addresses per chain
STABLES = {
    "Ethereum": {
        "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    },
    "Polygon": {
        "USDC.e": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        "USDC": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
        "USDT": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
    },
    "Arbitrum": {
        "USDC": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        "USDC.e": "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
    },
    "Optimism": {
        "USDC": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
    },
    "Base": {
        "USDC": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    },
}

def erc20_balance(w3, token, owner):
    sel = w3.keccak(text="balanceOf(address)")[:4]
    data = sel + encode(["address"], [Web3.to_checksum_address(owner)])
    try:
        raw = w3.eth.call({"to": Web3.to_checksum_address(token), "data": data})
        return decode(["uint256"], raw)[0] / 1e6  # USDC/USDT have 6 decimals
    except Exception:
        return 0

print(f"=== Fund Search for {EOA} ===\n")

total_value = 0

for chain, rpc in RPCS.items():
    try:
        w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
        if not w3.is_connected():
            print(f"{chain}: OFFLINE")
            continue

        # Native balance
        native = w3.eth.get_balance(Web3.to_checksum_address(EOA))
        native_eth = float(Web3.from_wei(native, "ether"))
        nonce = w3.eth.get_transaction_count(Web3.to_checksum_address(EOA))

        native_name = "ETH" if chain != "Polygon" else "POL"
        # Rough ETH price for value estimate
        native_usd = native_eth * 2000 if chain != "Polygon" else native_eth * 0.40

        line = f"{chain}: {native_eth:.6f} {native_name} (~${native_usd:.2f}), nonce={nonce}"
        if native_usd > 0.01:
            total_value += native_usd

        # Stablecoins
        stable_parts = []
        if chain in STABLES:
            for name, addr in STABLES[chain].items():
                bal = erc20_balance(w3, addr, EOA)
                if bal > 0.01:
                    stable_parts.append(f"{name}: ${bal:.2f}")
                    total_value += bal

        if stable_parts:
            line += " | " + ", ".join(stable_parts)

        print(line)

    except Exception as e:
        print(f"{chain}: ERROR - {e}")

print(f"\n=== TOTAL ESTIMATED VALUE: ${total_value:.2f} ===")

if total_value < 1:
    print("\nWallet is EMPTY across all major chains.")
    print("You need to deposit funds. Options:")
    print("  1. polymarket.com -> Deposit (card/crypto)")
    print("  2. Send USDC to your EOA on Polygon")
    print("  3. Bridge from another chain")
    print(f"\n  Your deposit address: {EOA}")
else:
    print(f"\nFunds found! Can bridge to Polygon for Polymarket.")
