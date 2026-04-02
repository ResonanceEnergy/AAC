#!/usr/bin/env python3
"""Debug signature issue."""
import os

from dotenv import load_dotenv

load_dotenv()

pk = os.getenv('POLYMARKET_PRIVATE_KEY')
funder = os.getenv('POLYMARKET_FUNDER_ADDRESS')
sig_type = int(os.getenv('POLYMARKET_SIGNATURE_TYPE', '1'))
chain_id = int(os.getenv('POLYMARKET_CHAIN_ID', '137'))

print(f"Private key: {pk[:10]}...{pk[-6:]}")
print(f"Funder: {funder}")
print(f"Sig type: {sig_type}")
print(f"Chain ID: {chain_id}")

from eth_account import Account

acct = Account.from_key(pk)
print(f"Derived EOA: {acct.address}")
if funder:
    print(f"Funder matches EOA: {acct.address.lower() == funder.lower()}")

# Try creating a simple order with different sig types
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType, PartialCreateOrderOptions
from py_clob_client.order_builder.constants import BUY

api_key = os.getenv('POLYMARKET_API_KEY')
api_secret = os.getenv('POLYMARKET_API_SECRET')
api_passphrase = os.getenv('POLYMARKET_API_PASSPHRASE')

creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_passphrase)

# Test with sig_type 0 (EOA) — maybe the key IS the direct signer
for st in [0, 1]:
    print(f"\n--- Testing signature_type={st} ---")
    try:
        client = ClobClient(
            "https://clob.polymarket.com",
            key=pk,
            chain_id=chain_id,
            signature_type=st,
            funder=funder if st == 1 else None,
            creds=creds,
        )
        # Use a known valid token_id — pick one that the tick-size endpoint accepted
        token_id = "6772099324721707224715070437205709623671146843087680984461675764599992554029"
        order = OrderArgs(token_id=token_id, price=0.01, size=1.0, side=BUY)
        options = PartialCreateOrderOptions(tick_size="0.01", neg_risk=False)
        signed = client.create_order(order, options)
        result = client.post_order(signed, OrderType.GTC)
        print(f"  SUCCESS: {result}")
        break
    except Exception as e:
        print(f"  FAILED: {e}")
