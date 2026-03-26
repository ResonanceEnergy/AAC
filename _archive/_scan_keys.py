#!/usr/bin/env python3
"""Scan codebase for API keys that might exist but aren't in .env."""
import os
import re
import sys
from pathlib import Path

ROOT = Path("c:/dev/AAC_fresh")
SKIP = {".venv", "__pycache__", ".git", "node_modules", ".egg-info", "archive", "build", ".context"}

# Patterns for API keys
KEY_PATTERNS = [
    (r'["\']([a-f0-9]{32})["\']', "hex32"),
    (r'["\']([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})["\']', "uuid"),
    (r'["\']([A-Za-z0-9_-]{30,60})["\']', "generic_long"),
]

# Check secrets directory
print("=== SECRETS DIRECTORY ===")
secrets_dir = ROOT / "secrets"
if secrets_dir.exists():
    for f in sorted(secrets_dir.rglob("*.txt")):
        content = f.read_text().strip()
        print(f"  {f.name}: {'HAS KEY' if content else 'EMPTY'} {content[:20] + '...' if len(content) > 20 else content}")
else:
    print("  No secrets directory")

# Check config directory
print("\n=== CONFIG DIRECTORY ===")
config_dir = ROOT / "config"
if config_dir.exists():
    for f in sorted(config_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}")

# Check .env files across all dev repos
print("\n=== OTHER .ENV FILES IN C:\\dev ===")
dev_root = Path("c:/dev")
for env_file in sorted(dev_root.glob("*/.env")):
    if "AAC_fresh" not in str(env_file):
        print(f"  {env_file}")

# Check for Google AI key in other repos
print("\n=== GOOGLE AI KEY SEARCH (all c:\\dev) ===")
for env_file in dev_root.glob("*/.env"):
    try:
        content = env_file.read_text()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if any(k in line.upper() for k in ["GOOGLE_AI", "GOOGLE_API", "GEMINI", "GOOGLE_CLOUD"]):
                print(f"  {env_file}: {line[:60]}...")
    except Exception:
        pass

# Check for Reddit keys in other repos
print("\n=== REDDIT KEY SEARCH (all c:\\dev) ===")
for env_file in dev_root.glob("*/.env"):
    try:
        content = env_file.read_text()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if "REDDIT" in line.upper():
                print(f"  {env_file}: {line[:60]}...")
    except Exception:
        pass

# Check for Telegram keys in other repos
print("\n=== TELEGRAM KEY SEARCH (all c:\\dev) ===")
for env_file in dev_root.glob("*/.env"):
    try:
        content = env_file.read_text()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if "TELEGRAM" in line.upper():
                print(f"  {env_file}: {line[:60]}...")
    except Exception:
        pass

# Check for Infura keys in other repos
print("\n=== INFURA KEY SEARCH (all c:\\dev) ===")
for env_file in dev_root.glob("*/.env"):
    try:
        content = env_file.read_text()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if "INFURA" in line.upper():
                print(f"  {env_file}: {line[:60]}...")
    except Exception:
        pass

# Check for Slack keys in other repos
print("\n=== SLACK/SMTP/TWITTER SEARCH (all c:\\dev) ===")
for env_file in dev_root.glob("*/.env"):
    try:
        content = env_file.read_text()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if any(k in line.upper() for k in ["SLACK", "SMTP", "TWITTER", "X_BEARER", "ZOHO"]):
                val = line.split("=", 1)
                if len(val) == 2 and val[1].strip():
                    print(f"  {env_file}: {line[:70]}...")
    except Exception:
        pass

# Check for CoinMarketCap, Whale Alert, Tradier in other repos
print("\n=== COINMARKETCAP/WHALE_ALERT/TRADIER SEARCH (all c:\\dev) ===")
for env_file in dev_root.glob("*/.env"):
    try:
        content = env_file.read_text()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if any(k in line.upper() for k in ["COINMARKETCAP", "CMC_", "WHALE_ALERT", "TRADIER", "TWELVE_DATA"]):
                val = line.split("=", 1)
                if len(val) == 2 and val[1].strip():
                    print(f"  {env_file}: {line[:70]}...")
    except Exception:
        pass

# Check for Binance/Coinbase/Kraken API keys in other repos  
print("\n=== EXCHANGE KEYS SEARCH (all c:\\dev) ===")
for env_file in dev_root.glob("*/.env"):
    try:
        content = env_file.read_text()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if any(k in line.upper() for k in ["BINANCE", "COINBASE", "KRAKEN"]):
                val = line.split("=", 1)
                if len(val) == 2 and val[1].strip():
                    print(f"  {env_file}: {line[:70]}...")
    except Exception:
        pass

# Also check DIGITAL LABOUR .env for any relevant keys
print("\n=== DIGITAL LABOUR .ENV ===")
dl_env = Path("c:/dev/DIGITAL LABOUR/DIGITAL LABOUR/.env")
if dl_env.exists():
    content = dl_env.read_text()
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        key = line.split("=", 1)[0].strip()
        val = line.split("=", 1)[1].strip() if "=" in line else ""
        if val and any(k in key.upper() for k in [
            "GOOGLE", "GEMINI", "REDDIT", "TELEGRAM", "SLACK", "SMTP", "ZOHO",
            "TWITTER", "X_BEARER", "X_API", "INFURA", "COINMARKET", "WHALE",
            "TRADIER", "TWELVE", "BINANCE", "COINBASE", "KRAKEN"
        ]):
            print(f"  {key}={val[:30]}...")
else:
    print("  Not found")

print("\n=== DONE ===")
