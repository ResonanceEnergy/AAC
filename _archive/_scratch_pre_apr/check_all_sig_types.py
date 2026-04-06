"""Check CLOB balance for ALL signature types to find where the funds are."""
import os, sys
from dotenv import load_dotenv
load_dotenv()

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams

PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY")
EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

# USDC.e on Polygon (Polymarket collateral)
COLLATERAL = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

SIG_TYPES = {
    0: "EOA (MetaMask direct)",
    1: "POLY_PROXY (Magic Link)",
    2: "POLY_GNOSIS_SAFE (most common for UI deposits)",
}

for sig_type, desc in SIG_TYPES.items():
    print(f"\n{'='*60}")
    print(f"Signature Type {sig_type}: {desc}")
    print(f"{'='*60}")
    
    try:
        client = ClobClient(
            HOST,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID,
            signature_type=sig_type,
            funder=EOA,
        )
        
        # Derive API creds
        creds = client.derive_api_key()
        api_key = creds.get("apiKey", "NONE")
        print(f"  API Key: {api_key[:20]}...")
        
        # Create a new client with derived creds
        client2 = ClobClient(
            HOST,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID,
            signature_type=sig_type,
            funder=EOA,
            creds={
                "key": creds["apiKey"],
                "secret": creds["secret"],
                "passphrase": creds["passphrase"],
            }
        )
        
        # Check collateral balance
        params = BalanceAllowanceParams(asset_type=0)  # 0 = COLLATERAL
        bal = client2.get_balance_allowance(params)
        print(f"  COLLATERAL balance: {bal}")
        
        balance_val = bal.get("balance", "0")
        if balance_val and balance_val != "0":
            print(f"\n  *** FOUND FUNDS! Balance: {balance_val} ***")
            print(f"  *** Use signature_type={sig_type} in .env ***")
        
        # Also check conditional token balances (types 1 and 2)
        for asset_type, asset_name in [(1, "CONDITIONAL_YES"), (2, "CONDITIONAL_NO")]:
            try:
                params2 = BalanceAllowanceParams(asset_type=asset_type)
                bal2 = client2.get_balance_allowance(params2)
                b = bal2.get("balance", "0")
                if b and b != "0":
                    print(f"  {asset_name} balance: {b}")
            except:
                pass
                
    except Exception as e:
        err = str(e)
        if "not registered" in err.lower() or "no api key" in err.lower():
            print(f"  Not registered with this sig type")
        else:
            print(f"  Error: {err[:100]}")

print("\n\nDone. The signature type with a non-zero balance is the correct one for .env")
