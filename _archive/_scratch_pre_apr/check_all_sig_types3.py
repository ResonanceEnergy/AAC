"""Check CLOB balance for ALL signature types."""
import os
from dotenv import load_dotenv
load_dotenv()

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams, ApiCreds

PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY")
EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

SIG_TYPES = {
    0: "EOA (MetaMask direct)",
    1: "POLY_PROXY (Magic Link)",
    2: "POLY_GNOSIS_SAFE (UI deposits)",
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
        
        # Derive creds
        creds = client.derive_api_key()
        api_key = creds.api_key
        api_secret = creds.api_secret
        api_passphrase = creds.api_passphrase
        
        print(f"  API Key: {api_key[:25]}...")
        
        # Create client with creds - use ApiCreds object
        creds_obj = ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )
        
        client2 = ClobClient(
            HOST,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID,
            signature_type=sig_type,
            funder=EOA,
            creds=creds_obj,
        )
        
        # Check collateral balance
        params = BalanceAllowanceParams(asset_type=0)
        bal = client2.get_balance_allowance(params)
        print(f"  COLLATERAL: {bal}")
        
        balance_val = bal.get("balance", "0") if isinstance(bal, dict) else getattr(bal, "balance", "0")
        
        if balance_val and str(balance_val) != "0":
            # Convert from smallest unit (USDC has 6 decimals)
            try:
                human = int(balance_val) / 1e6
                print(f"\n  *** FOUND ${human:.2f} USDC! ***")
            except:
                print(f"\n  *** FOUND FUNDS: {balance_val} ***")
            print(f"  *** Use POLYMARKET_SIGNATURE_TYPE={sig_type} in .env ***")
                
    except Exception as e:
        err = str(e)
        if len(err) > 300:
            err = err[:300]
        print(f"  Error: {err}")

print("\nDone.")
