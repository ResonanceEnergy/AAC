"""
Polymarket Full Diagnostic — based on complete docs deep dive.
Tests all signature types, Data API, geoblock, and proxy registration.
"""
import os, sys, json, time, requests
from dotenv import load_dotenv
load_dotenv()

PK = os.getenv("POLYMARKET_PRIVATE_KEY")
FUNDER = os.getenv("POLYMARKET_FUNDER_ADDRESS")  # 0xF4Ba...
CHAIN_ID = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
HOST = "https://clob.polymarket.com"

from py_clob_client.client import ClobClient
from eth_account import Account

acct = Account.from_key(PK)
EOA = acct.address
print(f"EOA:    {EOA}")
print(f"Funder: {FUNDER}")
print(f"Chain:  {CHAIN_ID}")
print()

# ============================================================
# 1. GEOBLOCK CHECK
# ============================================================
print("=" * 60)
print("1. GEOBLOCK CHECK")
print("=" * 60)
try:
    r = requests.get("https://polymarket.com/api/geoblock", timeout=10)
    geo = r.json()
    print(f"  Status: {r.status_code}")
    print(f"  Blocked: {geo.get('blocked')}")
    print(f"  IP:      {geo.get('ip')}")
    print(f"  Country: {geo.get('country')}")
    print(f"  Region:  {geo.get('region')}")
    if geo.get("blocked"):
        print("  *** WARNING: YOU ARE GEOBLOCKED — orders will be rejected ***")
except Exception as e:
    print(f"  Error: {e}")
print()

# ============================================================
# 2. CLOB HEALTH CHECK
# ============================================================
print("=" * 60)
print("2. CLOB HEALTH CHECK")
print("=" * 60)
try:
    r = requests.get(f"{HOST}/", timeout=5)
    print(f"  CLOB Status: {r.status_code} — {r.text[:100]}")
except Exception as e:
    print(f"  Error: {e}")

try:
    r = requests.get(f"{HOST}/time", timeout=5)
    print(f"  Server Time: {r.text[:100]}")
except Exception as e:
    print(f"  Error: {e}")
print()

# ============================================================
# 3. API KEY — CREATE OR DERIVE
# ============================================================
print("=" * 60)
print("3. API KEY — CREATE OR DERIVE")
print("=" * 60)
try:
    temp_client = ClobClient(HOST, CHAIN_ID, key=PK)
    creds = temp_client.create_or_derive_api_creds()
    print(f"  API Key:    {creds.api_key}")
    print(f"  Secret:     {creds.api_secret[:20]}...")
    print(f"  Passphrase: {creds.api_passphrase[:20]}...")
except Exception as e:
    print(f"  Error: {e}")
    creds = None
print()

# ============================================================
# 4. GET ALL API KEYS
# ============================================================
print("=" * 60)
print("4. GET ALL API KEYS")
print("=" * 60)
if creds:
    try:
        client_basic = ClobClient(HOST, CHAIN_ID, key=PK, creds={
            "apiKey": creds.api_key,
            "secret": creds.api_secret,
            "passphrase": creds.api_passphrase,
        })
        keys = client_basic.get_api_keys()
        print(f"  API Keys: {json.dumps(keys, indent=2)}")
    except Exception as e:
        print(f"  Error: {e}")
print()

# ============================================================
# 5. BALANCE CHECK — ALL 3 SIGNATURE TYPES
# ============================================================
print("=" * 60)
print("5. BALANCE CHECK — ALL SIGNATURE TYPES")
print("=" * 60)

sig_type_names = {0: "EOA", 1: "POLY_PROXY", 2: "GNOSIS_SAFE"}

for sig_type in [0, 1, 2]:
    sig_name = sig_type_names[sig_type]
    funder_addr = EOA if sig_type == 0 else FUNDER
    print(f"\n  --- Signature Type {sig_type} ({sig_name}), Funder: {funder_addr} ---")
    try:
        client = ClobClient(
            HOST, CHAIN_ID, key=PK,
            creds={"apiKey": creds.api_key, "secret": creds.api_secret, "passphrase": creds.api_passphrase},
            signature_type=sig_type,
            funder=funder_addr,
        )
        bal = client.get_balance_allowance()
        print(f"  COLLATERAL balance: {bal}")
    except Exception as e:
        print(f"  Error: {e}")

# Also try type 2 (GNOSIS_SAFE) with different factory address concept
# The Gnosis Safe Factory is 0xaacfeea03eb1561c4e67d661e40682bd20e3541b
# Our proxy was created by Polymarket Proxy Factory 0xaB45c5A4...
print()

# ============================================================
# 6. DATA API — POSITIONS & ACTIVITY (Public, no auth needed)
# ============================================================
print("=" * 60)
print("6. DATA API — POSITIONS & ACTIVITY")
print("=" * 60)

DATA_API = "https://data-api.polymarket.com"

for label, addr in [("EOA", EOA), ("Proxy/Funder", FUNDER)]:
    print(f"\n  --- {label}: {addr} ---")
    
    # Positions
    try:
        r = requests.get(f"{DATA_API}/positions?user={addr}", timeout=10)
        print(f"  Positions: status={r.status_code}")
        data = r.json()
        if isinstance(data, list):
            print(f"    Count: {len(data)}")
            for p in data[:3]:
                print(f"    - {p}")
        else:
            print(f"    Response: {json.dumps(data)[:200]}")
    except Exception as e:
        print(f"  Positions error: {e}")
    
    # Value
    try:
        r = requests.get(f"{DATA_API}/value?user={addr}", timeout=10)
        print(f"  Value: status={r.status_code}, data={r.text[:200]}")
    except Exception as e:
        print(f"  Value error: {e}")
    
    # Activity
    try:
        r = requests.get(f"{DATA_API}/activity?user={addr}", timeout=10)
        print(f"  Activity: status={r.status_code}")
        data = r.json()
        if isinstance(data, list):
            print(f"    Count: {len(data)}")
            for a in data[:3]:
                print(f"    - {a}")
        else:
            print(f"    Response: {json.dumps(data)[:200]}")
    except Exception as e:
        print(f"  Activity error: {e}")

print()

# ============================================================
# 7. GAMMA API — PROFILE LOOKUP
# ============================================================
print("=" * 60)
print("7. GAMMA API — PROFILE LOOKUP")
print("=" * 60)

GAMMA_API = "https://gamma-api.polymarket.com"

for label, addr in [("EOA", EOA), ("Proxy/Funder", FUNDER)]:
    print(f"\n  --- {label}: {addr} ---")
    try:
        r = requests.get(f"{GAMMA_API}/profiles/{addr}", timeout=10)
        print(f"  Profile: status={r.status_code}")
        if r.status_code == 200:
            print(f"    Data: {r.text[:300]}")
        else:
            print(f"    Response: {r.text[:200]}")
    except Exception as e:
        print(f"  Profile error: {e}")

print()

# ============================================================
# 8. TRY createApiKey (NEW key, not derive)
# ============================================================
print("=" * 60)
print("8. TRY createApiKey (new, not derive)")
print("=" * 60)
try:
    fresh_client = ClobClient(HOST, CHAIN_ID, key=PK)
    new_creds = fresh_client.create_api_key()
    print(f"  New API Key:    {new_creds.api_key}")
    print(f"  New Secret:     {new_creds.api_secret[:20]}...")
    print(f"  New Passphrase: {new_creds.api_passphrase[:20]}...")
    
    # Now try balance with the fresh key and sig type 1
    fresh_full = ClobClient(
        HOST, CHAIN_ID, key=PK,
        creds={"apiKey": new_creds.api_key, "secret": new_creds.api_secret, "passphrase": new_creds.api_passphrase},
        signature_type=1,
        funder=FUNDER,
    )
    bal = fresh_full.get_balance_allowance()
    print(f"  COLLATERAL balance (fresh key, type 1): {bal}")
    
    # And with sig type 2
    fresh_full2 = ClobClient(
        HOST, CHAIN_ID, key=PK,
        creds={"apiKey": new_creds.api_key, "secret": new_creds.api_secret, "passphrase": new_creds.api_passphrase},
        signature_type=2,
        funder=FUNDER,
    )
    bal2 = fresh_full2.get_balance_allowance()
    print(f"  COLLATERAL balance (fresh key, type 2): {bal2}")
    
except Exception as e:
    print(f"  Error: {e}")

print()

# ============================================================
# 9. RAW REST API — Balance Allowance endpoint directly
# ============================================================
print("=" * 60)
print("9. RAW REST API — direct balance-allowance call")
print("=" * 60)

# Build L2 headers manually using the client's internals
if creds:
    try:
        from py_clob_client.signing.hmac import build_hmac_signature
        from py_clob_client.headers import create_level_2_headers
        
        for sig_str in ["EOA", "POLY_PROXY", "GNOSIS_SAFE"]:
            url = f"{HOST}/balance-allowance?asset_type=COLLATERAL&signature_type={sig_str}"
            # Use built-in client for the request
            print(f"  Trying raw GET {url}")
            
            # We'll use the client's internal request method
            test_client = ClobClient(
                HOST, CHAIN_ID, key=PK,
                creds={"apiKey": creds.api_key, "secret": creds.api_secret, "passphrase": creds.api_passphrase},
                signature_type={"EOA": 0, "POLY_PROXY": 1, "GNOSIS_SAFE": 2}[sig_str],
                funder=FUNDER if sig_str != "EOA" else EOA,
            )
            bal = test_client.get_balance_allowance()
            print(f"    {sig_str}: {bal}")
    except Exception as e:
        print(f"  Error: {e}")

print()

# ============================================================
# 10. CHECK OPEN ORDERS & TRADES
# ============================================================
print("=" * 60)
print("10. OPEN ORDERS & TRADES (per sig type)")
print("=" * 60)

if creds:
    for sig_type in [0, 1, 2]:
        sig_name = sig_type_names[sig_type]
        funder_addr = EOA if sig_type == 0 else FUNDER
        try:
            client = ClobClient(
                HOST, CHAIN_ID, key=PK,
                creds={"apiKey": creds.api_key, "secret": creds.api_secret, "passphrase": creds.api_passphrase},
                signature_type=sig_type,
                funder=funder_addr,
            )
            orders = client.get_orders()
            trades = client.get_trades()
            print(f"  {sig_name}: orders={orders}, trades={trades}")
        except Exception as e:
            print(f"  {sig_name}: Error — {e}")

print()

# ============================================================
# 11. NOTIFICATION CHECK  
# ============================================================
print("=" * 60)
print("11. NOTIFICATION CHECK")
print("=" * 60)
if creds:
    try:
        client = ClobClient(
            HOST, CHAIN_ID, key=PK,
            creds={"apiKey": creds.api_key, "secret": creds.api_secret, "passphrase": creds.api_passphrase},
            signature_type=1,
            funder=FUNDER,
        )
        notifs = client.get_notifications()
        print(f"  Notifications: {notifs}")
    except Exception as e:
        print(f"  Error: {e}")

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
