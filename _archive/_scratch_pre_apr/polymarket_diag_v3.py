"""
Polymarket Diagnostic v3 — Correct BalanceAllowanceParams usage.
"""
import os, json, requests
from dotenv import load_dotenv
load_dotenv()

PK = os.getenv("POLYMARKET_PRIVATE_KEY")
FUNDER = os.getenv("POLYMARKET_FUNDER_ADDRESS")
CHAIN_ID = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
HOST = "https://clob.polymarket.com"

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType
from eth_account import Account

acct = Account.from_key(PK)
EOA = acct.address
print(f"EOA:    {EOA}")
print(f"Funder: {FUNDER}\n")

# Derive creds
temp = ClobClient(HOST, CHAIN_ID, key=PK)
creds = temp.create_or_derive_api_creds()
print(f"API Key: {creds.api_key}\n")

# ============================================================
# BALANCE CHECK — ALL 3 SIGNATURE TYPES
# ============================================================
print("=" * 60)
print("BALANCE CHECK — ALL SIGNATURE TYPES (COLLATERAL)")
print("=" * 60)

sig_names = {0: "EOA", 1: "POLY_PROXY", 2: "GNOSIS_SAFE"}

for sig_type in [0, 1, 2]:
    funder_addr = EOA if sig_type == 0 else FUNDER
    print(f"\n  Type {sig_type} ({sig_names[sig_type]}), Funder: {funder_addr}")
    try:
        c = ClobClient(
            HOST, CHAIN_ID, key=PK, creds=creds,
            signature_type=sig_type, funder=funder_addr,
        )
        # Pass proper BalanceAllowanceParams
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        bal = c.get_balance_allowance(params)
        print(f"    COLLATERAL: {bal}")
    except Exception as e:
        print(f"    Error: {e}")

print()

# ============================================================
# UPDATE BALANCE (resync from on-chain)
# ============================================================
print("=" * 60)
print("UPDATE BALANCE ALLOWANCE (resync from chain)")
print("=" * 60)

for sig_type in [1, 2]:
    funder_addr = FUNDER
    print(f"\n  Type {sig_type} ({sig_names[sig_type]}), Funder: {funder_addr}")
    try:
        c = ClobClient(
            HOST, CHAIN_ID, key=PK, creds=creds,
            signature_type=sig_type, funder=funder_addr,
        )
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        result = c.update_balance_allowance(params)
        print(f"    Update result: {result}")
        
        # Now re-check balance
        params2 = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        bal = c.get_balance_allowance(params2)
        print(f"    Balance after update: {bal}")
    except Exception as e:
        print(f"    Error: {e}")

print()

# ============================================================
# ALSO CHECK CONDITIONAL BALANCE (ERC1155 positions)
# ============================================================
print("=" * 60)
print("CONDITIONAL BALANCE CHECK (type 1)")
print("=" * 60)
try:
    c = ClobClient(
        HOST, CHAIN_ID, key=PK, creds=creds,
        signature_type=1, funder=FUNDER,
    )
    params = BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL)
    bal = c.get_balance_allowance(params)
    print(f"  CONDITIONAL (no token_id): {bal}")
except Exception as e:
    print(f"  Error: {e}")

print()
print("=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
If all balances show 0 with correct sig_type (1/POLY_PROXY):
  → The CLOB server's internal balance cache is empty.
  → update_balance_allowance should resync from on-chain.
  → If still 0 after update, the CLOB doesn't map this EOA→proxy.
  
If type 2 (GNOSIS_SAFE) works while type 1 doesn't:
  → The proxy might actually be a Gnosis Safe despite factory address.
""")
