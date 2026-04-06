"""Trace all ETH transactions from our EOA to understand fund flow."""
import json, urllib.request, time, os, sys
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
DEST = "0x89404369C1D90145462e38BA479970a3e1e6736E"
ETHERSCAN_KEY = os.getenv("ETHERSCAN_API_KEY", "")
ETHERSCAN = "https://api.etherscan.io/api"

def etherscan_get(params):
    params["apikey"] = ETHERSCAN_KEY
    url = ETHERSCAN + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

# Get all TXs from our EOA
print(f"=== All transactions for {EOA} ===")
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
            
            func = ""
            inp = tx.get("input", "0x")
            if inp and inp != "0x" and len(inp) >= 10:
                func = f" [fn: {inp[:10]}]"
            
            status = "OK" if tx.get("isError", "0") == "0" else "FAILED"
            to_addr = tx.get("to", "") or "CONTRACT_CREATE"
            flag = " <<< YOUR DEST" if to_addr.lower() == DEST.lower() else ""
            
            print(f"TX {i+1}: {direction} | {float(value_eth):.6f} ETH | gas: {float(gas_cost):.6f} ETH | {status}")
            print(f"       From: {tx['from']}")
            print(f"       To:   {to_addr}{func}{flag}")
            print(f"       Hash: {tx['hash']}")
            print()
        
        print(f"--- Summary ---")
        print(f"Total received:  {total_in:.6f} ETH")
        print(f"Total sent:      {total_out:.6f} ETH")
        print(f"Total gas spent: {total_gas:.6f} ETH")
        print(f"Net balance:     {total_in - total_out - total_gas:.6f} ETH")
    else:
        print(f"Error: {data.get('message', 'unknown')} | {data.get('result', '')}")
        
except Exception as e:
    print(f"Error: {e}")

time.sleep(0.25)

# ERC20 token transfers
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
            val = int(tx["value"]) / (10 ** dec) if dec > 0 else int(tx["value"])
            direction = "OUT" if tx["from"].lower() == EOA.lower() else "IN"
            sym = tx.get("tokenSymbol", "???")
            print(f"  {direction}: {val} {sym}")
            print(f"       From: {tx['from']}")
            print(f"       To:   {tx['to']}")
            print(f"       Hash: {tx['hash']}")
    else:
        print("  No ERC20 transfers")
except Exception as e:
    print(f"  Error: {e}")

time.sleep(0.25)

# Check the DEST address separately
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
        for tx in data["result"][:15]:
            direction = "OUT" if tx["from"].lower() == DEST.lower() else "IN"
            value_eth = Web3.from_wei(int(tx["value"]), "ether")
            func = ""
            inp = tx.get("input", "0x")
            if inp and inp != "0x" and len(inp) >= 10:
                func = f" [fn: {inp[:10]}]"
            status = "OK" if tx.get("isError", "0") == "0" else "FAILED"
            print(f"  {direction}: {float(value_eth):.6f} ETH | {status}{func}")
            print(f"       From: {tx['from']}")
            print(f"       To:   {tx.get('to', 'CREATE')}")
            print(f"       Hash: {tx['hash']}")
    else:
        print(f"  {data.get('message', '')} | {data.get('result', '')}")
except Exception as e:
    print(f"  Error: {e}")

# Also check internal TXs on DEST
time.sleep(0.25)
print(f"\n=== Internal transactions for DEST {DEST} ===")
try:
    data = etherscan_get({
        "module": "account",
        "action": "txlistinternal",
        "address": DEST,
        "startblock": "0",
        "endblock": "99999999",
        "sort": "asc",
    })
    if data["status"] == "1" and data["result"]:
        for tx in data["result"][:15]:
            value_eth = Web3.from_wei(int(tx["value"]), "ether")
            direction = "OUT" if tx["from"].lower() == DEST.lower() else "IN"
            print(f"  {direction}: {float(value_eth):.6f} ETH | from: {tx['from'][:20]}... to: {tx.get('to','?')[:20]}...")
    else:
        print("  No internal transactions")
except Exception as e:
    print(f"  Error: {e}")

# Check ERC20 on DEST
time.sleep(0.25)
print(f"\n=== ERC20 token transfers for DEST {DEST} ===")
try:
    data = etherscan_get({
        "module": "account",
        "action": "tokentx",
        "address": DEST,
        "startblock": "0",
        "endblock": "99999999",
        "sort": "asc",
    })
    if data["status"] == "1" and data["result"]:
        for tx in data["result"]:
            dec = int(tx.get("tokenDecimal", "18"))
            val = int(tx["value"]) / (10 ** dec) if dec > 0 else int(tx["value"])
            direction = "OUT" if tx["from"].lower() == DEST.lower() else "IN"
            sym = tx.get("tokenSymbol", "???")
            print(f"  {direction}: {val} {sym} | from: {tx['from'][:20]}... to: {tx['to'][:20]}...")
    else:
        print("  No ERC20 transfers")
except Exception as e:
    print(f"  Error: {e}")
