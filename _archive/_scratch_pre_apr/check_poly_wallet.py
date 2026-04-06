"""Quick check of Polymarket wallet connection and balance."""
import os
import sys

sys.path.insert(0, ".")
from shared.config_loader import load_env_file

load_env_file()

pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
funder = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
api_key = os.getenv("POLYMARKET_API_KEY", "")
api_secret = os.getenv("POLYMARKET_API_SECRET", "")
api_pass = os.getenv("POLYMARKET_API_PASSPHRASE", "")
chain = os.getenv("POLYMARKET_CHAIN_ID", "137")
sig = os.getenv("POLYMARKET_SIGNATURE_TYPE", "0")

print("=" * 60)
print("  POLYMARKET WALLET CHECK")
print("=" * 60)

if len(pk) > 10:
    print(f"  Private Key:    {pk[:6]}...{pk[-4:]}")
else:
    print(f"  Private Key:    MISSING ({len(pk)} chars)")

print(f"  Funder Address: {funder or 'MISSING'}")

if len(api_key) > 8:
    print(f"  API Key:        {api_key[:10]}...")
else:
    print(f"  API Key:        {api_key or 'MISSING'}")

if len(api_secret) > 8:
    print(f"  API Secret:     {api_secret[:10]}...")
else:
    print(f"  API Secret:     {api_secret or 'MISSING'}")

if len(api_pass) > 4:
    print(f"  API Passphrase: {api_pass[:6]}...")
else:
    print(f"  API Passphrase: {api_pass or 'MISSING'}")

print(f"  Chain ID:       {chain}")
print(f"  Signature Type: {sig}")
print()

# Try py_clob_client
try:
    from py_clob_client.client import ClobClient

    print("  py_clob_client: INSTALLED")

    creds = None
    if api_key and api_secret and api_pass:
        creds = {
            "api_key": api_key,
            "api_secret": api_secret,
            "api_passphrase": api_pass,
        }

    client = ClobClient(
        "https://clob.polymarket.com",
        key=pk,
        chain_id=int(chain),
        signature_type=int(sig),
        funder=funder or None,
        creds=creds,
    )
    print("  ClobClient:     INITIALIZED")

    # Check balance/allowance
    try:
        ba = client.get_balance_allowance()
        print(f"  Balance/Allow:  {ba}")
    except Exception as e:
        print(f"  Balance check:  {type(e).__name__}: {e}")

    # Try to get USDC balance on-chain via the proxy wallet
    try:
        import httpx

        # Query Polygonscan for USDC balance
        # USDC on Polygon: 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174 (PoS bridged)
        # or USDC native: 0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359
        addr = funder if funder else ""
        if addr:
            # Use Polymarket's own endpoint to check
            resp = httpx.get(
                f"https://clob.polymarket.com/balance",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            print(f"  CLOB /balance:  HTTP {resp.status_code} -> {resp.text[:200]}")
    except Exception as e:
        print(f"  On-chain check: {type(e).__name__}: {e}")

except ImportError:
    print("  py_clob_client: NOT INSTALLED")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

print("=" * 60)
