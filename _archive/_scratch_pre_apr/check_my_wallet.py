"""Check all proxy/safe options for our EOA on Polygon."""
from web3 import Web3
from eth_abi import decode

RPCS = [
    "https://polygon-rpc.com",
    "https://rpc.ankr.com/polygon",
    "https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213",
]
w3 = None
for rpc in RPCS:
    try:
        _w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
        if _w3.is_connected():
            w3 = _w3
            print(f"Connected via {rpc}")
            break
    except Exception:
        continue
if not w3:
    print("ERROR: Cannot connect to any Polygon RPC")
    exit(1)

EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

def call(contract, sig, types_in, args, types_out):
    sel = w3.keccak(text=sig)[:4]
    from eth_abi import encode
    data = sel + encode(types_in, args) if args else sel
    raw = w3.eth.call({"to": contract, "data": data})
    return decode(types_out, raw)

def usdc_balance(addr):
    sel = w3.keccak(text="balanceOf(address)")[:4]
    from eth_abi import encode
    data = sel + encode(["address"], [addr])
    raw = w3.eth.call({"to": USDC_E, "data": data})
    bal = decode(["uint256"], raw)[0]
    return bal / 1e6

print(f"=== Wallet Check for {EOA} ===\n")

# 1. EOA balances on Polygon
matic = w3.eth.get_balance(EOA)
usdc_eoa = usdc_balance(EOA)
nonce = w3.eth.get_transaction_count(EOA)
print(f"EOA on Polygon:")
print(f"  POL/MATIC: {Web3.from_wei(matic, 'ether')}")
print(f"  USDC.e:    ${usdc_eoa:.2f}")
print(f"  Nonce:     {nonce}")
print()

# 2. Exchange proxy
poly_proxy = Web3.to_checksum_address(call(EXCHANGE, "getPolyProxyWalletAddress(address)", ["address"], [EOA], ["address"])[0])
print(f"Exchange Poly Proxy: {poly_proxy}")
code = w3.eth.get_code(poly_proxy)
print(f"  Deployed: {len(code) > 0} ({len(code)} bytes)")
print(f"  USDC.e:   ${usdc_balance(poly_proxy):.2f}")
print()

# 3. Exchange safe
try:
    safe = Web3.to_checksum_address(call(EXCHANGE, "getSafeAddress(address)", ["address"], [EOA], ["address"])[0])
    print(f"Exchange Gnosis Safe: {safe}")
    code = w3.eth.get_code(safe)
    print(f"  Deployed: {len(code) > 0} ({len(code)} bytes)")
    print(f"  USDC.e:   ${usdc_balance(safe):.2f}")
except Exception as e:
    print(f"Exchange getSafeAddress: {e}")
print()

# 4. NegRisk Exchange proxy
try:
    neg_proxy = Web3.to_checksum_address(call(NEG_EXCHANGE, "getPolyProxyWalletAddress(address)", ["address"], [EOA], ["address"])[0])
    print(f"NegRisk Poly Proxy: {neg_proxy}")
    code = w3.eth.get_code(neg_proxy)
    print(f"  Deployed: {len(code) > 0} ({len(code)} bytes)")
    print(f"  USDC.e:   ${usdc_balance(neg_proxy):.2f}")
except Exception as e:
    print(f"NegRisk getPolyProxyWalletAddress: {e}")

try:
    neg_safe = Web3.to_checksum_address(call(NEG_EXCHANGE, "getSafeAddress(address)", ["address"], [EOA], ["address"])[0])
    print(f"NegRisk Gnosis Safe: {neg_safe}")
    code = w3.eth.get_code(neg_safe)
    print(f"  Deployed: {len(code) > 0} ({len(code)} bytes)")
    print(f"  USDC.e:   ${usdc_balance(neg_safe):.2f}")
except Exception as e:
    print(f"NegRisk getSafeAddress: {e}")
print()

# 5. Check the mystery funder
FUNDER = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"
print(f"Mystery funder in .env: {FUNDER}")
print(f"  USDC.e:   ${usdc_balance(FUNDER):.2f}")
print()

# 6. Summary
print("=== VERDICT ===")
if usdc_eoa > 0:
    print(f"Your EOA has ${usdc_eoa:.2f} USDC.e -- can use signature_type=0 (EOA)")
elif usdc_balance(poly_proxy) > 0:
    print(f"Your Poly Proxy has funds -- use signature_type=1")
else:
    print(f"No funds on your EOA or computed proxy.")
    print(f"The $556.39 in {FUNDER} does NOT belong to your key.")
    print()
    print("OPTIONS:")
    print("  A) Deposit USDC.e to your EOA on Polygon, use signature_type=0")
    print(f"     Send USDC.e to: {EOA}")
    print("     Also need a tiny bit of POL for gas")
    print()
    print("  B) Deploy your proxy and fund it, use signature_type=1")
    print(f"     Proxy addr: {poly_proxy}")
    print()
    print("  C) If you have a Polymarket.com account, check what address it shows")
    print("     Go to polymarket.com -> Profile -> the address shown is your proxy")
    print("     If it's NOT 0xF4Ba... then the funder in .env was wrong all along")
