"""Deep dive into Polymarket wallet/positions for both EOA and PROXY addresses."""
import json
import urllib.request
import time

EOA = "0x4BFC40EA4051f84E90eA0a25998578f6191Acad9"
PROXY = "0xF4BaEe5f82823e10141715610D4e050A3dCeEDD8"

def api_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read())

# 1. Profile lookup
for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
    print(f"\n{'='*60}")
    print(f"PROFILE: {name} ({addr})")
    print("=" * 60)
    try:
        data = api_get(f"https://gamma-api.polymarket.com/profiles/{addr}")
        print(json.dumps(data, indent=2)[:2000])
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(0.5)

# 2. Data API positions
for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
    print(f"\n{'='*60}")
    print(f"POSITIONS (data-api): {name} ({addr})")
    print("=" * 60)
    try:
        data = api_get(f"https://data-api.polymarket.com/positions?user={addr}")
        if isinstance(data, list):
            print(f"  Position count: {len(data)}")
            total = 0
            for p in data[:30]:
                size = p.get("size", 0)
                cur_val = p.get("currentValue", 0) or 0
                title = p.get("title", p.get("question", "unknown"))
                if isinstance(title, str) and len(title) > 60:
                    title = title[:60] + "..."
                print(f"  [{title}] size={size} value=${cur_val}")
                total += float(cur_val)
            print(f"  TOTAL POSITION VALUE: ${total:.2f}")
        else:
            print(json.dumps(data, indent=2)[:2000])
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(0.5)

# 3. CLOB orders
for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
    print(f"\n{'='*60}")
    print(f"OPEN ORDERS (clob): {name} ({addr})")
    print("=" * 60)
    try:
        data = api_get(f"https://clob.polymarket.com/orders?owner={addr}")
        if isinstance(data, list):
            print(f"  Open order count: {len(data)}")
            for o in data[:10]:
                print(f"  Order: {o.get('id', '?')[:20]} side={o.get('side')} price={o.get('price')} size={o.get('size')}")
        else:
            print(json.dumps(data, indent=2)[:1000])
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(0.5)

# 4. CLOB trade history
for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
    print(f"\n{'='*60}")
    print(f"TRADE HISTORY (clob): {name} ({addr})")
    print("=" * 60)
    try:
        data = api_get(f"https://clob.polymarket.com/trades?maker_address={addr}")
        if isinstance(data, list):
            print(f"  Trade count: {len(data)}")
            for t in data[:10]:
                print(f"  Trade: side={t.get('side')} price={t.get('price')} size={t.get('size')} market={t.get('market', '?')[:30]}")
        else:
            print(json.dumps(data, indent=2)[:1000])
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(0.5)

# 5. Gamma API activity/portfolio
for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
    print(f"\n{'='*60}")
    print(f"GAMMA PORTFOLIO: {name} ({addr})")
    print("=" * 60)
    try:
        data = api_get(f"https://gamma-api.polymarket.com/query?query_type=portfolio&address={addr}")
        print(json.dumps(data, indent=2)[:2000])
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(0.5)

# 6. Check conditional token balances (CTF) via Polymarket subgraph
for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
    print(f"\n{'='*60}")
    print(f"CTF TOKEN BALANCES (subgraph): {name} ({addr})")
    print("=" * 60)
    try:
        query = json.dumps({
            "query": """
            {
              userPositions(where: {user: "%s"}, first: 20) {
                id
                condition { id }
                balance
                wrappedBalance
              }
            }
            """ % addr.lower()
        }).encode()
        req = urllib.request.Request(
            "https://api.thegraph.com/subgraphs/name/polymarket/polymarket-matic",
            query,
            {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        )
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        print(json.dumps(data, indent=2)[:2000])
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(0.5)

# 7. Check Polymarket rewards/referral API
print(f"\n{'='*60}")
print("ADDITIONAL API ENDPOINTS")
print("=" * 60)

# Check for proxy mapping
for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
    print(f"\nProxy mapping check for {name}:")
    try:
        data = api_get(f"https://gamma-api.polymarket.com/nonce?address={addr}")
        print(f"  Nonce response: {json.dumps(data)[:500]}")
    except Exception as e:
        print(f"  {e}")

# Check the relay endpoint for proxy info
for name, addr in [("EOA", EOA), ("PROXY", PROXY)]:
    print(f"\nRelay proxy check for {name}:")
    try:
        data = api_get(f"https://relayer-api.polymarket.com/address/{addr}")
        print(f"  Relay response: {json.dumps(data)[:500]}")
    except Exception as e:
        print(f"  {e}")

print("\n\nDONE.")
