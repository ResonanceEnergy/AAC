#!/usr/bin/env python3
"""Full system API key audit."""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
from shared.config_loader import load_env_file

load_env_file()

apis = [
    ("BINANCE_API_KEY", "Binance"),
    ("BINANCE_SECRET_KEY", "Binance Secret"),
    ("COINBASE_API_KEY", "Coinbase Pro"),
    ("COINBASE_SECRET_KEY", "Coinbase Secret"),
    ("KRAKEN_API_KEY", "Kraken"),
    ("KRAKEN_SECRET_KEY", "Kraken Secret"),
    ("NDAX_API_KEY", "NDAX"),
    ("MOOMOO_API_KEY", "Moomoo"),
    ("POLYGON_API_KEY", "Polygon.io"),
    ("FINNHUB_API_KEY", "Finnhub"),
    ("ALPHAVANTAGE_API_KEY", "Alpha Vantage"),
    ("TWELVE_DATA_API_KEY", "Twelve Data"),
    ("IEX_CLOUD_API_KEY", "IEX Cloud"),
    ("EODHD_API_KEY", "EODHD"),
    ("INTRINIO_API_KEY", "Intrinio"),
    ("TRADESTIE_API_KEY", "TradeStie"),
    ("WALLSTREETODDS_API_KEY", "WallStreetOdds"),
    ("TRADIER_API_KEY", "Tradier"),
    ("FRED_API_KEY", "FRED"),
    ("WHALE_ALERT_API_KEY", "Whale Alert"),
    ("SANTIMENT_API_KEY", "Santiment"),
    ("UNUSUAL_WHALES_API_KEY", "Unusual Whales"),
    ("COINMARKETCAP_API_KEY", "CoinMarketCap"),
    ("NEWS_API_KEY", "NewsAPI"),
    ("XAI_API_KEY", "xAI Grok"),
    ("OPENAI_API_KEY", "OpenAI"),
    ("ANTHROPIC_API_KEY", "Anthropic"),
    ("GOOGLE_AI_API_KEY", "Google AI"),
    ("REDDIT_CLIENT_ID", "Reddit Client"),
    ("REDDIT_CLIENT_SECRET", "Reddit Secret"),
    ("ETHERSCAN_API_KEY", "Etherscan"),
    ("INFURA_PROJECT_ID", "Infura"),
    ("TELEGRAM_BOT_TOKEN", "Telegram"),
    ("SLACK_WEBHOOK_URL", "Slack"),
    ("SMTP_USER", "Email/SMTP"),
    ("TWITTER_BEARER_TOKEN", "Twitter/X"),
    ("COINGECKO_API_KEY", "CoinGecko"),
    ("CLAWHUB_API_KEY", "ClawHub"),
    ("MASSIVE_API_KEY", "Massive"),
    ("IBKR_USERNAME", "IBKR Username"),
    ("IBKR_ACCOUNT", "IBKR Account"),
]

configured = []
missing = []
for var, name in apis:
    val = os.environ.get(var, "").strip()
    if val:
        preview = val[:8] + "..." if len(val) > 12 else val
        configured.append((name, var, preview))
    else:
        missing.append((name, var))

# Secrets directory
secrets_dir = Path("secrets")
secret_files = []
if secrets_dir.exists():
    for f in sorted(secrets_dir.iterdir()):
        if f.is_file() and f.suffix == ".txt":
            content = f.read_text().strip()
            status = "HAS KEY" if content else "EMPTY"
            preview = content[:8] + "..." if len(content) > 12 else content
            secret_files.append((f.name, status, preview))

# _FILE pointers
file_vars = []
for var, name in apis:
    file_var = f"{var}_FILE"
    file_val = os.environ.get(file_var, "").strip()
    if file_val:
        p = Path(file_val)
        exists = p.exists()
        has_content = bool(p.read_text().strip()) if exists else False
        file_vars.append((file_var, file_val, exists, has_content))

SEP = "=" * 70
DASH = "-" * 66

print(SEP)
print("AAC FULL API KEY AUDIT")
print(SEP)

print(f"\n  CONFIGURED ({len(configured)}):")
print(f"  {DASH}")
for name, var, preview in sorted(configured):
    print(f"  [OK] {name:<22} {var:<32} {preview}")

print(f"\n  MISSING ({len(missing)}):")
print(f"  {DASH}")
for name, var in sorted(missing):
    print(f"  [ ] {name:<22} {var}")

if secret_files:
    print(f"\n  SECRETS FILES ({len(secret_files)}):")
    print(f"  {DASH}")
    for fname, status, preview in secret_files:
        marker = "OK" if status == "HAS KEY" else "  "
        pv = preview if status == "HAS KEY" else ""
        print(f"  [{marker}] {fname:<40} {status:<10} {pv}")

if file_vars:
    print(f"\n  FILE-BACKED POINTERS ({len(file_vars)}):")
    print(f"  {DASH}")
    for fvar, fval, exists, has_content in file_vars:
        if exists and has_content:
            st = "OK"
        elif exists:
            st = "EMPTY FILE"
        else:
            st = "FILE MISSING"
        mk = "OK" if exists and has_content else "  "
        print(f"  [{mk}] {fvar:<40} -> {fval} ({st})")

# Cross-check
env_keys_set = {var for var, _ in apis if os.environ.get(var, "").strip()}
print(f"\n  CROSS-CHECK ISSUES:")
print(f"  {DASH}")
issues = 0
for sf_name, sf_status, _ in secret_files:
    if sf_status == "HAS KEY":
        env_var = sf_name.replace(".txt", "").upper()
        if env_var not in env_keys_set:
            print(f"  [!] Secret file {sf_name} has key but {env_var} not in env")
            issues += 1
if not issues:
    print("  None - all secret files are reflected in env vars")

print(f"\n  SUMMARY: {len(configured)} configured, {len(missing)} missing, {len(secret_files)} secret files")
print(SEP)
