"""Check Polymarket profile/portfolio for our EOA and the mystery funder."""
import json, urllib.request

EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
OLD_FUNDER = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"
DEST = "0x89404369C1D90145462e38BA479970a3e1e6736E"

addresses = {
    "EOA (your key)": EOA,
    "Old Funder (0xF4Ba)": OLD_FUNDER,
    "Dest (0x8940)": DEST,
}

# Polymarket has a profile API and a Gamma API
for name, addr in addresses.items():
    print(f"\n{'='*60}")
    print(f"Checking: {name} = {addr}")
    print(f"{'='*60}")
    
    # Try Polymarket profile API
    urls = [
        f"https://clob.polymarket.com/profile/{addr}",
        f"https://gamma-api.polymarket.com/users/{addr}",
        f"https://clob.polymarket.com/data/positions?user={addr}",
    ]
    
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
                print(f"\n  {url.split('/')[-2]}/{url.split('/')[-1][:30]}:")
                if isinstance(data, dict):
                    # Print non-empty fields
                    for k, v in data.items():
                        if v and v != "0" and v != [] and v != {}:
                            val_str = str(v)
                            if len(val_str) > 100:
                                val_str = val_str[:100] + "..."
                            print(f"    {k}: {val_str}")
                elif isinstance(data, list):
                    print(f"    {len(data)} items")
                    for item in data[:5]:
                        print(f"    - {str(item)[:100]}")
                else:
                    print(f"    {str(data)[:200]}")
        except urllib.error.HTTPError as e:
            print(f"\n  {url.split('/')[-2]}/{url.split('/')[-1][:30]}: HTTP {e.code}")
            try:
                body = e.read().decode()[:200]
                print(f"    {body}")
            except:
                pass
        except Exception as e:
            print(f"\n  {url.split('/')[-2]}/{url.split('/')[-1][:30]}: {str(e)[:100]}")

# Also check CLOB orders for our API key
print(f"\n{'='*60}")
print("Checking CLOB orders with our API creds...")
print(f"{'='*60}")
import os
from dotenv import load_dotenv
load_dotenv()
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds
    
    PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY")
    
    for sig_type in [0, 1, 2]:
        client = ClobClient(
            "https://clob.polymarket.com",
            key=PRIVATE_KEY,
            chain_id=137,
            signature_type=sig_type,
            funder=EOA,
        )
        creds = client.derive_api_key()
        creds_obj = ApiCreds(
            api_key=creds.api_key,
            api_secret=creds.api_secret,
            api_passphrase=creds.api_passphrase,
        )
        client2 = ClobClient(
            "https://clob.polymarket.com",
            key=PRIVATE_KEY,
            chain_id=137,
            signature_type=sig_type,
            funder=EOA,
            creds=creds_obj,
        )
        
        # Try to get open orders
        try:
            orders = client2.get_orders()
            print(f"\n  Sig type {sig_type} orders: {orders}")
        except Exception as e:
            print(f"\n  Sig type {sig_type} orders error: {str(e)[:100]}")
        
        # Try to get trades
        try:
            trades = client2.get_trades()
            print(f"  Sig type {sig_type} trades: {trades}")
        except Exception as e:
            print(f"  Sig type {sig_type} trades error: {str(e)[:100]}")
            
except Exception as e:
    print(f"  Error: {e}")
