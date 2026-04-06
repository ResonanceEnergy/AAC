"""Check CLOB balance with corrected EOA signature type."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams

HOST = "https://clob.polymarket.com"
KEY = os.environ["POLYMARKET_PRIVATE_KEY"]
CHAIN_ID = int(os.environ.get("POLYMARKET_CHAIN_ID", "137"))
FUNDER = os.environ["POLYMARKET_FUNDER_ADDRESS"]
SIG_TYPE = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "0"))

print(f"Key:       {KEY[:10]}...{KEY[-6:]}")
print(f"Funder:    {FUNDER}")
print(f"Sig Type:  {SIG_TYPE}")
print(f"Chain ID:  {CHAIN_ID}")
print()

client = ClobClient(HOST, key=KEY, chain_id=CHAIN_ID, signature_type=SIG_TYPE, funder=FUNDER)

print("Deriving API credentials...")
try:
    creds = client.derive_api_key()
    print(f"  API Key: {creds.api_key[:20]}...")
    client.set_api_creds(creds)
except Exception as e:
    print(f"  derive_api_key failed: {e}")
    print("  Trying create_api_key...")
    try:
        creds = client.create_api_key()
        print(f"  API Key: {creds.api_key[:20]}...")
        client.set_api_creds(creds)
    except Exception as e2:
        print(f"  create_api_key also failed: {e2}")
        sys.exit(1)

print()

for sig in [0, 1, 2]:
    try:
        params = BalanceAllowanceParams(asset_type=sig)
        ba = client.get_balance_allowance(params)
        print(f"Sig Type {sig} balance: {ba}")
    except Exception as e:
        print(f"Sig Type {sig}: {e}")
