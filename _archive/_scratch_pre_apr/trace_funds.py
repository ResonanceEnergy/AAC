"""Trace all ETH transactions from our EOA to understand fund flow."""
import json
import time
import urllib.request

from web3 import Web3

EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
DEST = "0x89404369C1D90145462e38BA479970a3e1e6736E"

# Etherscan free API (no key needed for basic queries, 5/sec limit)
ETHERSCAN = "https://api.etherscan.io/api"

def etherscan_get(params):
    params["apikey"] = "YourApiKeyToken"  # free tier works without key
    url = ETHERSCAN + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

# Get all TXs from our EOA (nonce=5, so 5 outgoing TXs)
print(f"=== All transactions FROM {EOA} ===")
try:
    data = etherscan_get({
        "module": "account",
        "action": "txlist",
        "address": EOA,
        "startblock": "0",
        "endblock": "99999999",
        "sort": "asc",
    })

    if data["status"] == "1":
        txs = data["result"]
        print(f"Found {len(txs)} transactions\n")

        total_in = 0
        total_out = 0
        total_gas = 0

        for i, tx in enumerate(txs):
            direction = "OUT" if tx["from"].lower() == EOA.lower() else "IN"
            value_eth = Web3.from_wei(int(tx["value"]), "ether")
            gas_cost = Web3.from_wei(int(tx["gasUsed"]) * int(tx["gasPrice"]), "ether")

            if direction == "OUT":
                total_out += float(value_eth)
                total_gas += float(gas_cost)
            else:
                total_in += float(value_eth)

            # Decode function if it has input data
            func = ""
            inp = tx.get("input", "0x")
            if inp and inp != "0x" and len(inp) >= 10:
                func = f" [fn: {inp[:10]}]"

            status = "OK" if tx.get("isError", "0") == "0" else "FAILED"
            to_addr = tx.get("to", "CONTRACT_CREATE")

            # Flag if to matches our destination
            flag = " <<< YOUR DEST" if to_addr and to_addr.lower() == DEST.lower() else ""

            print(f"TX {i+1}: {direction} | {value_eth:.6f} ETH | gas: {gas_cost:.6f} ETH | {status}")
            print(f"       To: {to_addr}{func}{flag}")
            print(f"       Block: {tx['blockNumber']} | Hash: {tx['hash'][:20]}...")
            print()

        print(f"--- Summary ---")
        print(f"Total received: {total_in:.6f} ETH")
        print(f"Total sent:     {total_out:.6f} ETH")
        print(f"Total gas:      {total_gas:.6f} ETH")
        print(f"Net:            {total_in - total_out - total_gas:.6f} ETH")
    else:
        print(f"Etherscan returned: {data['message']}")

except Exception as e:
    print(f"Etherscan error: {e}")

time.sleep(0.3)

# Also check internal TXs (contract calls moving ETH)
print(f"\n=== Internal transactions for {EOA} ===")
try:
    data = etherscan_get({
        "module": "account",
        "action": "txlistinternal",
        "address": EOA,
        "startblock": "0",
        "endblock": "99999999",
        "sort": "asc",
    })
    if data["status"] == "1" and data["result"]:
        for tx in data["result"]:
            value_eth = Web3.from_wei(int(tx["value"]), "ether")
            direction = "OUT" if tx["from"].lower() == EOA.lower() else "IN"
            print(f"  {direction}: {value_eth:.6f} ETH | to: {tx.get('to', '?')} | from: {tx.get('from', '?')}")
    else:
        print("  No internal transactions")
except Exception as e:
    print(f"  Error: {e}")

time.sleep(0.3)

# Check ERC20 token transfers
print(f"\n=== ERC20 token transfers for {EOA} ===")
try:
    data = etherscan_get({
        "module": "account",
        "action": "tokentx",
        "address": EOA,
        "startblock": "0",
        "endblock": "99999999",
        "sort": "asc",
    })
    if data["status"] == "1" and data["result"]:
        for tx in data["result"]:
            dec = int(tx.get("tokenDecimal", "18"))
            val = int(tx["value"]) / (10 ** dec)
            direction = "OUT" if tx["from"].lower() == EOA.lower() else "IN"
            sym = tx.get("tokenSymbol", "???")
            print(f"  {direction}: {val:.4f} {sym} | to: {tx['to'][:20]}... | contract: {tx['contractAddress'][:20]}...")
    else:
        print("  No ERC20 transfers")
except Exception as e:
    print(f"  Error: {e}")

time.sleep(0.3)

# Now check the DEST address transactions
print(f"\n=== All transactions for DEST {DEST} ===")
try:
    data = etherscan_get({
        "module": "account",
        "action": "txlist",
        "address": DEST,
        "startblock": "0",
        "endblock": "99999999",
        "sort": "asc",
    })
    if data["status"] == "1":
        for tx in data["result"][:10]:
            direction = "OUT" if tx["from"].lower() == DEST.lower() else "IN"
            value_eth = Web3.from_wei(int(tx["value"]), "ether")
            from_addr = tx["from"][:20]
            to_addr = (tx.get("to") or "CREATE")[:20]
            inp = tx.get("input", "0x")
            func = f" [fn: {inp[:10]}]" if inp and inp != "0x" and len(inp) >= 10 else ""
            print(f"  {direction}: {value_eth:.6f} ETH | from: {from_addr}... to: {to_addr}...{func}")
    else:
        print(f"  {data['message']}")
except Exception as e:
    print(f"  Error: {e}")
