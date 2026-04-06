"""Check CLOB balance with Polymarket wallet 0xF4Ba... as funder, all sig types."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams

HOST = "https://clob.polymarket.com"
KEY = os.environ["POLYMARKET_PRIVATE_KEY"]
CHAIN_ID = 137
FUNDER = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"

print(f"Key:    {KEY[:10]}...{KEY[-6:]}")
print(f"Funder: {FUNDER}")
print()

for sig_type in [0, 1, 2]:
    label = {0: "EOA", 1: "POLY_PROXY", 2: "GNOSIS_SAFE"}[sig_type]
    print(f"--- Sig Type {sig_type} ({label}) ---")
    try:
        client = ClobClient(HOST, key=KEY, chain_id=CHAIN_ID, signature_type=sig_type, funder=FUNDER)
        creds = client.derive_api_key()
        client.set_api_creds(creds)
        print(f"  API Key: {creds.api_key[:20]}...")

        from py_clob_client.clob_types import AssetType
        bal = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        print(f"  Balance: {bal}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
