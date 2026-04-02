#!/usr/bin/env python3
"""Validate newly added API keys."""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, ".")
from shared.config_loader import load_env_file

load_env_file()

results = []

# 1. Google AI (Gemini)
key = os.environ.get("GOOGLE_AI_API_KEY", "")
if key:
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])[:3]]
            results.append(("Google AI (Gemini)", "OK", f"Models: {', '.join(models)}"))
    except Exception as e:
        results.append(("Google AI (Gemini)", "FAIL", str(e)[:80]))
else:
    results.append(("Google AI (Gemini)", "MISSING", ""))

# 2. OpenAI
key = os.environ.get("OPENAI_API_KEY", "")
if key:
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            count = len(data.get("data", []))
            results.append(("OpenAI", "OK", f"{count} models available"))
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:100]
        results.append(("OpenAI", f"HTTP {e.code}", body))
    except Exception as e:
        results.append(("OpenAI", "FAIL", str(e)[:80]))
else:
    results.append(("OpenAI", "MISSING", ""))

# 3. Anthropic
key = os.environ.get("ANTHROPIC_API_KEY", "")
if key:
    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({"model": "claude-sonnet-4-20250514", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}).encode(),
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            results.append(("Anthropic", "OK", f"Model: {data.get('model', '?')}"))
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:100]
        results.append(("Anthropic", f"HTTP {e.code}", body))
    except Exception as e:
        results.append(("Anthropic", "FAIL", str(e)[:80]))
else:
    results.append(("Anthropic", "MISSING", ""))

# 4. xAI Grok
key = os.environ.get("XAI_API_KEY", "")
if key:
    try:
        req = urllib.request.Request(
            "https://api.x.ai/v1/models",
            headers={"Authorization": f"Bearer {key}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            models = [m.get("id", "") for m in data.get("data", [])[:3]]
            results.append(("xAI Grok", "OK", f"Models: {', '.join(models)}"))
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:100]
        results.append(("xAI Grok", f"HTTP {e.code}", body))
    except Exception as e:
        results.append(("xAI Grok", "FAIL", str(e)[:80]))
else:
    results.append(("xAI Grok", "MISSING", ""))

# 5. Twitter/X Bearer
key = os.environ.get("TWITTER_BEARER_TOKEN", "")
if key:
    try:
        req = urllib.request.Request(
            "https://api.twitter.com/2/tweets/search/recent?query=bitcoin&max_results=10",
            headers={"Authorization": f"Bearer {key}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            count = len(data.get("data", []))
            results.append(("Twitter/X", "OK", f"{count} tweets returned"))
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:100]
        results.append(("Twitter/X", f"HTTP {e.code}", body))
    except Exception as e:
        results.append(("Twitter/X", "FAIL", str(e)[:80]))
else:
    results.append(("Twitter/X", "MISSING", ""))

# 6. SMTP (just check config is set)
smtp_user = os.environ.get("SMTP_USER", "")
smtp_pass = os.environ.get("SMTP_PASSWORD", "")
smtp_host = os.environ.get("SMTP_HOST", "")
if smtp_user and smtp_pass:
    results.append(("SMTP", "OK", f"Host: {smtp_host}, User: {smtp_user}"))
else:
    results.append(("SMTP", "MISSING", ""))

print("=" * 60)
print("KEY VALIDATION RESULTS")
print("=" * 60)
for name, status, detail in results:
    icon = "OK" if status == "OK" else "!!" if "FAIL" in status or "MISSING" in status else "??"
    print(f"  [{icon}] {name:<20} {status:<10} {detail}")
print("=" * 60)
