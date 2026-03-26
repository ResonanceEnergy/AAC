#!/usr/bin/env python3
"""Re-derive L2 API credentials with signature_type=0 (EOA)."""
import os
from dotenv import load_dotenv
load_dotenv()

from py_clob_client.client import ClobClient

pk = os.getenv('POLYMARKET_PRIVATE_KEY')
chain_id = int(os.getenv('POLYMARKET_CHAIN_ID', '137'))

client = ClobClient(
    "https://clob.polymarket.com",
    key=pk,
    chain_id=chain_id,
    signature_type=0,  # EOA
)

print("Deriving L2 API creds with signature_type=0 (EOA)...")
creds = client.create_or_derive_api_creds()
print(f"API_KEY:        {creds.api_key}")
print(f"API_SECRET:     {creds.api_secret}")
print(f"API_PASSPHRASE: {creds.api_passphrase}")

# Test auth
client.set_api_creds(creds)
orders = client.get_orders()
print(f"\nAuth test - get_orders(): {len(orders)} orders")
print("SUCCESS — L2 creds work with sig_type=0")
