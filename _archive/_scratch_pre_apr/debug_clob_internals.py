"""Debug CLOB client internals — check what address it's actually using."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

HOST = "https://clob.polymarket.com"
KEY = os.environ["POLYMARKET_PRIVATE_KEY"]
CHAIN_ID = 137
FUNDER = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"

print("=" * 60)
print("CLOB CLIENT INTERNALS DEBUG")
print("=" * 60)

for sig_type in [0, 1, 2]:
    label = {0: "EOA", 1: "POLY_PROXY", 2: "GNOSIS_SAFE"}[sig_type]
    print(f"\n--- Sig Type {sig_type} ({label}) ---")

    client = ClobClient(HOST, key=KEY, chain_id=CHAIN_ID, signature_type=sig_type, funder=FUNDER)

    # Inspect internal state
    print(f"  client.chain_id: {client.chain_id}")
    print(f"  client.funder: {getattr(client, 'funder', 'N/A')}")
    print(f"  client.signature_type: {getattr(client, 'signature_type', 'N/A')}")

    # Check signer
    signer = getattr(client, 'signer', None)
    if signer:
        print(f"  client.signer type: {type(signer).__name__}")
        print(f"  client.signer.address: {getattr(signer, 'address', 'N/A')}")
        # Check if signer has a funder method
        for attr in dir(signer):
            if 'fund' in attr.lower() or 'proxy' in attr.lower() or 'address' in attr.lower():
                val = getattr(signer, attr, None)
                if not callable(val):
                    print(f"  client.signer.{attr}: {val}")

    # Check key_derivation_signer
    kds = getattr(client, 'key_derivation_signer', None)
    if kds:
        print(f"  client.key_derivation_signer type: {type(kds).__name__}")
        for attr in dir(kds):
            if not attr.startswith('_') and not callable(getattr(kds, attr, None)):
                try:
                    val = getattr(kds, attr)
                    if isinstance(val, (str, int, bool)):
                        print(f"  kds.{attr}: {val}")
                except:
                    pass

    # Derive API key and inspect
    try:
        creds = client.derive_api_key()
        print(f"  API Key: {creds.api_key}")
        print(f"  API Secret: {creds.api_secret[:20]}...")
        print(f"  API Passphrase: {creds.api_passphrase[:20]}...")
    except Exception as e:
        print(f"  derive_api_key ERROR: {e}")

print()
print("=" * 60)

# Now let's check what the CLOB server thinks our address is
print("\n--- CLOB Server: Who Am I? ---")
for sig_type in [1]:
    client = ClobClient(HOST, key=KEY, chain_id=CHAIN_ID, signature_type=sig_type, funder=FUNDER)
    creds = client.derive_api_key()
    client.set_api_creds(creds)

    # Try to get API keys (list registered keys)
    try:
        keys = client.get_api_keys()
        print(f"  Registered API keys: {json.dumps(keys, indent=2)[:500]}")
    except Exception as e:
        print(f"  get_api_keys: {e}")

    # Check what endpoints are available
    try:
        # Try /auth/api-keys
        import base64
        import hashlib
        import hmac
        import time
        from urllib.request import Request, urlopen

        timestamp = str(int(time.time()))
        method = "GET"
        path = "/auth/api-keys"
        msg = f"{timestamp}{method}{path}"

        sig = base64.urlsafe_b64encode(
            hmac.new(
                base64.urlsafe_b64decode(creds.api_secret),
                msg.encode(),
                hashlib.sha256
            ).digest()
        ).decode()

        headers = {
            "POLY_API_KEY": creds.api_key,
            "POLY_SIGNATURE": sig,
            "POLY_TIMESTAMP": timestamp,
            "POLY_PASSPHRASE": creds.api_passphrase,
        }

        req = Request(f"{HOST}{path}", headers=headers)
        resp = urlopen(req, timeout=10)
        data = json.loads(resp.read())
        print(f"  /auth/api-keys response: {json.dumps(data, indent=2)[:500]}")
    except Exception as e:
        print(f"  /auth/api-keys: {e}")

    # Try /auth/derive-api-key with explicit request to see nonce
    try:
        # Check if create_or_derive is different
        creds2 = client.create_or_derive_api_creds()
        print(f"  create_or_derive creds: key={creds2.api_key}")
    except Exception as e:
        print(f"  create_or_derive: {e}")

print()
# Direct REST: /auth/derive-api-key
print("--- Manual derive-api-key calls ---")
from eth_account import Account
from eth_account.messages import encode_defunct

acct = Account.from_key(KEY)

for sig_type_val in [0, 1, 2]:
    label = {0: "EOA", 1: "POLY_PROXY", 2: "GNOSIS_SAFE"}[sig_type_val]

    # The CLOB API derive endpoint
    # POST /auth/derive-api-key with body: {timestamp, nonce, signature}
    import time
    ts = int(time.time())
    nonce = 0

    # Sign message: "polymarket-api-key" + timestamp + nonce
    # Actually the message format depends on the SDK
    # Let's check what the SDK sends
    print(f"\n  Sig {sig_type_val} ({label}):")
    try:
        c = ClobClient(HOST, key=KEY, chain_id=CHAIN_ID, signature_type=sig_type_val, funder=FUNDER)

        # Look at the headers module
        import inspect

        import py_clob_client.headers

        # Find create_level_1_headers or similar
        for name in dir(py_clob_client.headers):
            if 'header' in name.lower() or 'key' in name.lower() or 'derive' in name.lower():
                obj = getattr(py_clob_client.headers, name)
                if callable(obj):
                    sig_src = inspect.getsource(obj)
                    print(f"    {name}:")
                    for line in sig_src.split('\n')[:5]:
                        print(f"      {line}")
                    print()
    except Exception as e:
        print(f"    inspect error: {e}")
    break  # Only need to check once
