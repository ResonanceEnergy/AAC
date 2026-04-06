"""Check the destination address and the Ethereum mainnet TX."""
from eth_abi import decode, encode
from web3 import Web3

EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
DEST = "0x89404369C1D90145462e38BA479970a3e1e6736E"
TX_HASH = "0x18ec3182a1f9713b5bfa1f062d0cc2bf477829af032d287359b6a4499c07655e"

# Try multiple RPCs
for rpc in ["https://mainnet.infura.io/v3/84842078b09946638c03157f83405213", "https://rpc.ankr.com/eth"]:
    try:
        w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 15}))
        if w3.is_connected():
            print(f"Connected to Ethereum via {rpc}")
            break
    except:
        continue
else:
    print("Cannot connect to Ethereum")
    exit(1)

# Check destination address
dest_cs = Web3.to_checksum_address(DEST)
code = w3.eth.get_code(dest_cs)
bal = w3.eth.get_balance(dest_cs)
nonce = w3.eth.get_transaction_count(dest_cs)

print(f"\nDestination: {DEST}")
print(f"  Code:    {len(code)} bytes ({'CONTRACT' if len(code) > 0 else 'EOA'})")
print(f"  ETH:     {Web3.from_wei(bal, 'ether')}")
print(f"  Nonce:   {nonce}")

# Look up the TX
print(f"\nTransaction: {TX_HASH}")
try:
    tx = w3.eth.get_transaction(TX_HASH)
    receipt = w3.eth.get_transaction_receipt(TX_HASH)
    sender = tx["from"]
    to = tx["to"]
    value = Web3.from_wei(tx["value"], "ether")
    gas_used = receipt["gasUsed"]
    status = receipt["status"]
    block = tx["blockNumber"]

    print(f"  From:    {sender}")
    print(f"  To:      {to}")
    print(f"  Value:   {value} ETH")
    print(f"  Block:   {block}")
    print(f"  Status:  {'SUCCESS' if status == 1 else 'FAILED'}")
    print(f"  Gas:     {gas_used}")

    # Check if there are any logs (token transfers, etc.)
    if receipt["logs"]:
        print(f"  Logs:    {len(receipt['logs'])} events")
        for i, log in enumerate(receipt["logs"]):
            print(f"    Log {i}: contract={log['address']}, topics={len(log['topics'])}")
            if len(log["topics"]) > 0:
                topic0 = log["topics"][0].hex()
                # Transfer event signature
                if topic0 == "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef":
                    print(f"           ^ ERC20 Transfer event")
    else:
        print(f"  Logs:    None (simple ETH transfer)")

    # Check if from matches our EOA
    if sender.lower() == EOA.lower():
        print(f"\n  ** CONFIRMED: This TX was sent FROM your wallet **")
    else:
        print(f"\n  ** WARNING: Sender {sender} is NOT your EOA {EOA} **")

except Exception as e:
    print(f"  Error: {e}")

# Also check what the destination has on Polygon
print("\n--- Checking destination on Polygon ---")
for rpc in ["https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213", "https://polygon-rpc.com"]:
    try:
        w3p = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
        if w3p.is_connected():
            code_p = w3p.eth.get_code(dest_cs)
            bal_p = w3p.eth.get_balance(dest_cs)
            print(f"Polygon: {len(code_p)} bytes code, {Web3.from_wei(bal_p, 'ether')} POL")

            # Check USDC.e balance
            USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
            sel = w3p.keccak(text="balanceOf(address)")[:4]
            data = sel + encode(["address"], [dest_cs])
            raw = w3p.eth.call({"to": Web3.to_checksum_address(USDC_E), "data": data})
            usdc = decode(["uint256"], raw)[0] / 1e6
            print(f"Polygon USDC.e: ${usdc:.2f}")
            break
    except Exception as e:
        print(f"Polygon check failed: {e}")

# Check our EOA's full TX history on Ethereum (nonce = 5 means 5 TXs)
print(f"\n--- Your EOA on Ethereum: nonce={w3.eth.get_transaction_count(Web3.to_checksum_address(EOA))} ---")
print("(5 transactions sent from this address on Ethereum mainnet)")
