"""Check Polymarket on-chain USDC balance via multiple public RPCs."""
from web3 import Web3

rpcs = [
    "https://polygon.llamarpc.com",
    "https://rpc.ankr.com/polygon",
    "https://polygon-mainnet.public.blastapi.io",
    "https://1rpc.io/matic",
]

funder = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"
usdc_addr = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
abi = [{"constant": True, "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "type": "function"}]

for rpc in rpcs:
    try:
        w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 8}))
        if w3.is_connected():
            print(f"Connected: {rpc}")
            usdc = w3.eth.contract(address=Web3.to_checksum_address(usdc_addr), abi=abi)
            raw = usdc.functions.balanceOf(Web3.to_checksum_address(funder)).call()
            matic = w3.eth.get_balance(Web3.to_checksum_address(funder))
            usd = raw / 1e6
            pol = matic / 1e18
            print(f"USDC: {usd:.2f}")
            print(f"POL:  {pol:.6f}")
            break
        else:
            print(f"Not connected: {rpc}")
    except Exception as e:
        print(f"Failed {rpc}: {type(e).__name__}: {str(e)[:120]}")
else:
    print("ALL RPCs FAILED")
