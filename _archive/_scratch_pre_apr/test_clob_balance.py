"""Test CLOB balance using exact Polymarket SDK pattern."""
import os

import dotenv
import requests

dotenv.load_dotenv()

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams

key = os.getenv("POLYMARKET_PRIVATE_KEY")
funder = os.getenv("POLYMARKET_FUNDER_ADDRESS")

# Init exactly per docs
client = ClobClient(
    host="https://clob.polymarket.com",
    chain_id=137,
    key=key,
    signature_type=1,  # POLY_PROXY
    funder=funder,
)

# Derive L2 creds
creds = client.create_or_derive_api_creds()
client.set_api_creds(creds)
print(f"API Key: {creds.api_key}")

# Get a real token_id from active market
resp = requests.get(
    "https://gamma-api.polymarket.com/markets",
    params={"closed": "false", "limit": "1", "order": "volume24hr", "ascending": "false"},
)
market = resp.json()[0]
question = market.get("question", "?")
print(f"Market: {question[:60]}")

import json

tokens_raw = market.get("clobTokenIds", "")
if isinstance(tokens_raw, str):
    # Could be JSON array string like '["123","456"]'
    try:
        tokens_list = json.loads(tokens_raw)
        token_id = tokens_list[0]
    except (json.JSONDecodeError, IndexError):
        token_id = tokens_raw.split(",")[0].strip().strip('"[]')
else:
    token_id = tokens_raw[0]
print(f"Token ID: {token_id[:40]}...")

# Check what BalanceAllowanceParams expects
import inspect

print(f"\nBalanceAllowanceParams fields: {inspect.signature(BalanceAllowanceParams)}")

# Check COLLATERAL balance (= USDC.e available for trading)
# asset_type: "COLLATERAL" or "CONDITIONAL" (strings, not ints)
for asset_type_val in ["COLLATERAL", "CONDITIONAL"]:
    for sig in [1, 0]:
        try:
            params = BalanceAllowanceParams(
                asset_type=asset_type_val,
                token_id=token_id,
                signature_type=sig,
            )
            bal = client.get_balance_allowance(params)
            print(f"  {asset_type_val} (sig={sig}): {bal}")
        except Exception as e:
            print(f"  {asset_type_val} (sig={sig}): ERROR - {e}")
