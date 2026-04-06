"""Check CLOB balance for ALL signature types to find where the funds are."""
import os
from dotenv import load_dotenv
load_dotenv()

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams

PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY")
EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

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
        
        # Derive API creds - returns ApiCreds object
        creds = client.derive_api_key()
        print(f"  Creds type: {type(creds).__name__}")
        
        # Try both dict and object access
        try:
            api_key = creds.apiKey if hasattr(creds, 'apiKey') else creds["apiKey"]
            secret = creds.secret if hasattr(creds, 'secret') else creds["secret"]
            passphrase = creds.passphrase if hasattr(creds, 'passphrase') else creds["passphrase"]
        except:
            # Inspect all attrs
            print(f"  Creds attrs: {[a for a in dir(creds) if not a.startswith('_')]}")
            api_key = getattr(creds, 'key', None) or getattr(creds, 'api_key', None)
            secret = getattr(creds, 'secret', None)
            passphrase = getattr(creds, 'passphrase', None)
            if not api_key:
                # Try dict-like
                if hasattr(creds, '__dict__'):
                    print(f"  Creds dict: {creds.__dict__}")
                raise Exception("Cannot extract API key")
        
        print(f"  API Key: {str(api_key)[:25]}...")
        
        # Create client with creds
        client2 = ClobClient(
            HOST,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID,
            signature_type=sig_type,
            funder=EOA,
            creds={
                "key": api_key,
                "secret": secret,
                "passphrase": passphrase,
            }
        )
        
        # Check collateral balance
        params = BalanceAllowanceParams(asset_type=0)
        bal = client2.get_balance_allowance(params)
        print(f"  COLLATERAL: {bal}")
        
        # Parse balance
        if isinstance(bal, dict):
            balance_val = bal.get("balance", "0")
        else:
            balance_val = getattr(bal, "balance", "0")
        
        if balance_val and str(balance_val) != "0":
            print(f"\n  *** FOUND FUNDS! Balance: {balance_val} ***")
            print(f"  *** Use POLYMARKET_SIGNATURE_TYPE={sig_type} in .env ***")
                
    except Exception as e:
        err = str(e)
        if len(err) > 200:
            err = err[:200]
        print(f"  Error: {err}")

print("\n\nDone.")
