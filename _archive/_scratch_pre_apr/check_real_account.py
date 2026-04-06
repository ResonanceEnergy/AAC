"""Check CLOB balance using 0x8940 as funder — the actual Polymarket smart account."""
import os
from dotenv import load_dotenv
load_dotenv()

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams, ApiCreds, AssetType

PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY")
EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
POLYMARKET_ACCOUNT = "0x89404369C1D90145462e38BA479970a3e1e6736E"
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

SIG_TYPES = {
    0: "EOA",
    1: "POLY_PROXY",
    2: "POLY_GNOSIS_SAFE",
}

# Test 1: Using 0x8940 as funder
print("=" * 60)
print(f"TEST 1: funder = {POLYMARKET_ACCOUNT}")
print("=" * 60)

for sig_type, desc in SIG_TYPES.items():
    print(f"\n  Sig type {sig_type} ({desc}):")
    try:
        client = ClobClient(
            HOST,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID,
            signature_type=sig_type,
            funder=POLYMARKET_ACCOUNT,
        )
        
        creds = client.derive_api_key()
        creds_obj = ApiCreds(
            api_key=creds.api_key,
            api_secret=creds.api_secret,
            api_passphrase=creds.api_passphrase,
        )
        print(f"    API Key: {creds.api_key[:25]}...")
        
        client2 = ClobClient(
            HOST,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID,
            signature_type=sig_type,
            funder=POLYMARKET_ACCOUNT,
            creds=creds_obj,
        )
        
        bal = client2.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        balance_val = bal.get("balance", "0") if isinstance(bal, dict) else "?"
        print(f"    Balance: {bal}")
        
        if balance_val and balance_val != "0":
            try:
                human = int(balance_val) / 1e6
                print(f"\n    *** FOUND ${human:.2f} USDC! sig_type={sig_type}, funder={POLYMARKET_ACCOUNT} ***")
            except:
                print(f"\n    *** FOUND: {balance_val} ***")
    except Exception as e:
        print(f"    Error: {str(e)[:150]}")

# Test 2: No funder
print("\n" + "=" * 60)
print("TEST 2: no funder (None)")
print("=" * 60)

for sig_type, desc in SIG_TYPES.items():
    print(f"\n  Sig type {sig_type} ({desc}):")
    try:
        client = ClobClient(
            HOST,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID,
            signature_type=sig_type,
        )
        creds = client.derive_api_key()
        creds_obj = ApiCreds(
            api_key=creds.api_key,
            api_secret=creds.api_secret,
            api_passphrase=creds.api_passphrase,
        )
        
        client2 = ClobClient(
            HOST,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID,
            signature_type=sig_type,
            creds=creds_obj,
        )
        
        bal = client2.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        balance_val = bal.get("balance", "0") if isinstance(bal, dict) else "?"
        print(f"    Balance: {bal}")
        
        if balance_val and balance_val != "0":
            try:
                human = int(balance_val) / 1e6
                print(f"\n    *** FOUND ${human:.2f} USDC! sig_type={sig_type}, NO funder ***")
            except:
                pass
    except Exception as e:
        print(f"    Error: {str(e)[:150]}")

# Test 3: Direct REST API check for the Polymarket smart account
print("\n" + "=" * 60)
print(f"TEST 3: Direct on-chain USDC.e balance of {POLYMARKET_ACCOUNT} on Polygon")
print("=" * 60)
from web3 import Web3
from eth_abi import encode, decode

for rpc in ["https://polygon-mainnet.infura.io/v3/84842078b09946638c03157f83405213", "https://polygon-rpc.com"]:
    try:
        w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
        if w3.is_connected():
            acct = Web3.to_checksum_address(POLYMARKET_ACCOUNT)
            # Native POL
            pol = w3.eth.get_balance(acct)
            print(f"  POL: {Web3.from_wei(pol, 'ether')}")
            
            # Code check
            code = w3.eth.get_code(acct)
            print(f"  Code: {len(code)} bytes")
            if len(code) > 0:
                print(f"  Bytecode: {code.hex()}")
            
            # USDC.e
            USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
            sel = w3.keccak(text="balanceOf(address)")[:4]
            data = sel + encode(["address"], [acct])
            raw = w3.eth.call({"to": USDC_E, "data": data})
            usdc = decode(["uint256"], raw)[0] / 1e6
            print(f"  USDC.e: ${usdc:.2f}")
            
            # USDC (native) 
            USDC_NATIVE = Web3.to_checksum_address("0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359")
            raw2 = w3.eth.call({"to": USDC_NATIVE, "data": data})
            usdc2 = decode(["uint256"], raw2)[0] / 1e6
            print(f"  USDC (native): ${usdc2:.2f}")
            
            # Nonce
            nonce = w3.eth.get_transaction_count(acct)
            print(f"  Nonce: {nonce}")
            break
    except Exception as e:
        print(f"  RPC error: {e}")
