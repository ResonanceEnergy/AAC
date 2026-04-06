"""Trace the creation of proxy 0xF4Ba... to find the actual owner EOA."""
import json
from urllib.request import Request, urlopen

ETHERSCAN_KEY = "U8HCAUW68ZXNXSMEZGNGKIMWEA4U4VX9YM"
PROXY = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"
FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"
EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"

def etherscan_v2(chainid, **params):
    params["apikey"] = ETHERSCAN_KEY
    url = f"https://api.etherscan.io/v2/api?chainid={chainid}&" + "&".join(f"{k}={v}" for k, v in params.items())
    resp = json.loads(urlopen(Request(url), timeout=15).read())
    return resp

# 1. Get ALL transactions for the proxy on Polygon (including creation)
print("=== ALL txs for proxy wallet on Polygon ===")
resp = etherscan_v2(137, module="account", action="txlist", address=PROXY, startblock=0, endblock=99999999, page=1, offset=50, sort="asc")
if resp.get("status") == "1":
    for tx in resp["result"]:
        print(f"  TX: {tx['hash'][:18]}...")
        print(f"    from={tx['from']}")
        print(f"    to={tx['to']}")
        print(f"    value={int(tx['value'])/1e18:.6f} POL")
        print(f"    input={tx['input'][:66]}...")
        print(f"    func={tx.get('functionName','')}")
        print(f"    block={tx['blockNumber']}")
        print()
else:
    print(f"  {resp.get('message')} - {resp.get('result')}")

# 2. Get internal txs for the proxy (creation)
print("\n=== Internal txs for proxy ===")
resp2 = etherscan_v2(137, module="account", action="txlistinternal", address=PROXY, startblock=0, endblock=99999999, page=1, offset=50, sort="asc")
if resp2.get("status") == "1":
    for tx in resp2["result"]:
        print(f"  TX: {tx.get('hash','?')[:18]}...")
        print(f"    from={tx.get('from','?')}")
        print(f"    to={tx.get('to','?')}")
        print(f"    type={tx.get('type','?')}")
        print(f"    value={int(tx.get('value',0))/1e18:.6f}")
        # The hash is the parent tx — let's get its input
        parent_hash = tx.get("hash")
        if parent_hash:
            print(f"    Parent TX hash: {parent_hash}")
        print()
else:
    print(f"  {resp2.get('message')} - {resp2.get('result')}")

# 3. Get the parent creation transaction details
print("\n=== Parent creation tx details ===")
if resp2.get("status") == "1" and resp2["result"]:
    parent_hash = resp2["result"][0].get("hash")
    if parent_hash:
        resp3 = etherscan_v2(137, module="proxy", action="eth_getTransactionByHash", txhash=parent_hash)
        if resp3.get("result"):
            tx = resp3["result"]
            print(f"  Hash: {tx.get('hash')}")
            print(f"  From: {tx.get('from')}")
            print(f"  To:   {tx.get('to')}")
            input_data = tx.get("input", "")
            print(f"  Input length: {len(input_data)} chars")
            print(f"  Input (first 200): {input_data[:200]}")

            # The factory function likely takes the owner address as parameter
            # Parse the input data
            if len(input_data) > 10:
                func_sig = input_data[:10]
                print(f"\n  Function signature: {func_sig}")
                # Decode parameters (each 32 bytes = 64 hex chars)
                params = input_data[10:]
                param_idx = 0
                while param_idx * 64 < len(params):
                    chunk = params[param_idx*64:(param_idx+1)*64]
                    if len(chunk) == 64:
                        # Check if it looks like an address (first 24 chars are zeros)
                        if chunk[:24] == "0" * 24:
                            addr = "0x" + chunk[24:]
                            is_our_eoa = addr.lower() == EOA.lower()
                            print(f"  Param {param_idx}: address {addr} {'<<< OUR EOA!' if is_our_eoa else ''}")
                        else:
                            val = int(chunk, 16)
                            print(f"  Param {param_idx}: {chunk} (uint256: {val})")
                    param_idx += 1
        else:
            print(f"  Could not fetch tx: {resp3}")

# 4. Also check factory txs around that time to find proxy creation calls for our EOA
print("\n=== Factory txs involving our EOA ===")
resp4 = etherscan_v2(137, module="account", action="txlist", address=FACTORY, startblock=0, endblock=99999999, page=1, offset=100, sort="desc")
if resp4.get("status") == "1":
    found = False
    for tx in resp4["result"]:
        if tx["from"].lower() == EOA.lower() or EOA[2:].lower() in tx["input"].lower():
            found = True
            print(f"  TX: {tx['hash'][:18]}...")
            print(f"    from={tx['from']}")
            print(f"    input={tx['input'][:200]}")
            print(f"    block={tx['blockNumber']}")
            print()
    if not found:
        print(f"  No txs from/involving our EOA in last {len(resp4['result'])} factory txs")
        # Show first 3 factory txs for context
        for tx in resp4["result"][:3]:
            print(f"  Sample TX from={tx['from'][:20]}... input={tx['input'][:80]}...")
else:
    print(f"  {resp4.get('message')} - {resp4.get('result')}")

# 5. Check if there are ERC20 approval txs from proxy
print("\n=== ERC20 token txs for proxy ===")
resp5 = etherscan_v2(137, module="account", action="tokentx", address=PROXY, startblock=0, endblock=99999999, page=1, offset=50, sort="asc")
if resp5.get("status") == "1":
    for tx in resp5["result"]:
        decimals = int(tx.get("tokenDecimal", 18))
        value = int(tx["value"]) / (10 ** decimals)
        direction = "IN" if tx["to"].lower() == PROXY.lower() else "OUT"
        print(f"  {direction} | {tx['tokenSymbol']:>8} | ${value:>15.4f} | from={tx['from'][:16]}... to={tx['to'][:16]}... | block={tx['blockNumber']}")
        # Check tx hash to find who initiated the transfer
        print(f"       TX: {tx['hash'][:18]}...")
else:
    print(f"  {resp5.get('message')} - {resp5.get('result')}")
