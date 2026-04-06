"""
Polymarket Diagnostic v2 — Fixed creds handling.
"""
import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

PK = os.getenv("POLYMARKET_PRIVATE_KEY")
FUNDER = os.getenv("POLYMARKET_FUNDER_ADDRESS")
CHAIN_ID = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
HOST = "https://clob.polymarket.com"

from eth_account import Account
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

acct = Account.from_key(PK)
EOA = acct.address
print(f"EOA:    {EOA}")
print(f"Funder: {FUNDER}")
print()

# Get creds
print("=== DERIVE CREDS ===")
temp = ClobClient(HOST, CHAIN_ID, key=PK)
creds_raw = temp.create_or_derive_api_creds()
print(f"  Type: {type(creds_raw)}")
print(f"  Raw:  {creds_raw}")

# Handle both dict and object
if isinstance(creds_raw, dict):
    api_key = creds_raw.get("apiKey") or creds_raw.get("api_key")
    api_secret = creds_raw.get("secret") or creds_raw.get("api_secret")
    api_passphrase = creds_raw.get("passphrase") or creds_raw.get("api_passphrase")
else:
    api_key = creds_raw.api_key
    api_secret = creds_raw.api_secret
    api_passphrase = creds_raw.api_passphrase

print(f"  Key:  {api_key}")
print(f"  Sec:  {api_secret[:20]}...")
print(f"  Pass: {api_passphrase[:20]}...")
print()

creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_passphrase)

# ============================================================
# BALANCE CHECK — ALL 3 SIGNATURE TYPES
# ============================================================
print("=== BALANCE CHECK — ALL SIGNATURE TYPES ===")
sig_names = {0: "EOA", 1: "POLY_PROXY", 2: "GNOSIS_SAFE"}

for sig_type in [0, 1, 2]:
    funder_addr = EOA if sig_type == 0 else FUNDER
    print(f"\n  Type {sig_type} ({sig_names[sig_type]}), Funder: {funder_addr}")
    try:
        c = ClobClient(HOST, CHAIN_ID, key=PK, creds=creds, signature_type=sig_type, funder=funder_addr)
        bal = c.get_balance_allowance()
        print(f"    Balance: {bal}")
    except Exception as e:
        print(f"    Error: {e}")

print()

# ============================================================
# GET ALL API KEYS
# ============================================================
print("=== API KEYS ===")
try:
    c = ClobClient(HOST, CHAIN_ID, key=PK, creds=creds, signature_type=1, funder=FUNDER)
    keys = c.get_api_keys()
    print(f"  Keys: {json.dumps(keys, indent=2)}")
except Exception as e:
    print(f"  Error: {e}")
print()

# ============================================================
# OPEN ORDERS & TRADES
# ============================================================
print("=== OPEN ORDERS & TRADES ===")
for sig_type in [0, 1, 2]:
    funder_addr = EOA if sig_type == 0 else FUNDER
    try:
        c = ClobClient(HOST, CHAIN_ID, key=PK, creds=creds, signature_type=sig_type, funder=funder_addr)
        orders = c.get_orders()
        trades = c.get_trades()
        print(f"  {sig_names[sig_type]}: orders={orders}, trades={trades}")
    except Exception as e:
        print(f"  {sig_names[sig_type]}: {e}")
print()

# ============================================================
# NOTIFICATIONS
# ============================================================
print("=== NOTIFICATIONS ===")
try:
    c = ClobClient(HOST, CHAIN_ID, key=PK, creds=creds, signature_type=1, funder=FUNDER)
    notifs = c.get_notifications()
    print(f"  Notifications: {notifs}")
except Exception as e:
    print(f"  Error: {e}")
print()

# ============================================================
# RAW HTTP — balance-allowance endpoint
# ============================================================
print("=== RAW HTTP — balance-allowance ===")
try:
    # Build HMAC headers manually
    import base64
    import hashlib
    import hmac
    import time

    timestamp = str(int(time.time()))
    method = "GET"
    path = "/balance-allowance"
    body = ""

    for sig_str in ["EOA", "POLY_PROXY", "GNOSIS_SAFE"]:
        query = f"asset_type=COLLATERAL&signature_type={sig_str}"
        full_path = f"{path}?{query}"

        # HMAC signature: timestamp + method + path + body
        message = timestamp + method + full_path + body
        sig = base64.urlsafe_b64encode(
            hmac.new(
                base64.urlsafe_b64decode(api_secret),
                message.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        headers = {
            "POLY_ADDRESS": EOA,
            "POLY_SIGNATURE": sig,
            "POLY_TIMESTAMP": timestamp,
            "POLY_API_KEY": api_key,
            "POLY_PASSPHRASE": api_passphrase,
        }

        url = f"{HOST}{full_path}"
        r = requests.get(url, headers=headers, timeout=10)
        print(f"  {sig_str}: status={r.status_code}, body={r.text[:200]}")
except Exception as e:
    print(f"  Error: {e}")
print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("KEY FINDINGS:")
print("  - NOT geoblocked (CA/BC)")
print("  - Data API: 0 positions, 0 value for BOTH addresses")
print("  - createApiKey FAILED (400: Could not create)")
print("  - deriveApiKey works → key exists in CLOB")
print("  - Check balance results above for all 3 sig types")
print("=" * 60)
