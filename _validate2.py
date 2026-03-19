#!/usr/bin/env python3
"""Validate keys and write results to val_out.txt."""
import os, sys, json, urllib.request, urllib.error
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from shared.config_loader import load_env_file
load_env_file()

out = []

def test_key(name, url, headers=None, method="GET", data=None):
    try:
        req = urllib.request.Request(url, headers=headers or {}, data=data, method=method)
        with urllib.request.urlopen(req, timeout=10) as resp:
            d = json.loads(resp.read())
            return "OK", d
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}", e.read().decode()[:80]
    except Exception as e:
        return "FAIL", str(e)[:80]

# Google AI
k = os.environ.get("GOOGLE_AI_API_KEY", "")
if k:
    s, d = test_key("Google AI", f"https://generativelanguage.googleapis.com/v1beta/models?key={k}")
    detail = ", ".join(m.get("name","") for m in d.get("models",[])[:3]) if s == "OK" else str(d)
    out.append(f"[{'OK' if s=='OK' else '!!'}] Google AI          {s:<10} {detail[:60]}")
else:
    out.append("[!!] Google AI          MISSING")

# OpenAI
k = os.environ.get("OPENAI_API_KEY", "")
if k:
    s, d = test_key("OpenAI", "https://api.openai.com/v1/models", {"Authorization": f"Bearer {k}"})
    detail = f"{len(d.get('data',[]))} models" if s == "OK" else str(d)[:60]
    out.append(f"[{'OK' if s=='OK' else '!!'}] OpenAI              {s:<10} {detail}")
else:
    out.append("[!!] OpenAI              MISSING")

# Anthropic
k = os.environ.get("ANTHROPIC_API_KEY", "")
if k:
    body = json.dumps({"model":"claude-sonnet-4-20250514","max_tokens":5,"messages":[{"role":"user","content":"hi"}]}).encode()
    s, d = test_key("Anthropic", "https://api.anthropic.com/v1/messages",
        {"x-api-key": k, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
        method="POST", data=body)
    detail = d.get("model","?") if s == "OK" else str(d)[:60]
    out.append(f"[{'OK' if s=='OK' else '!!'}] Anthropic            {s:<10} {detail}")
else:
    out.append("[!!] Anthropic            MISSING")

# xAI Grok
k = os.environ.get("XAI_API_KEY", "")
if k:
    s, d = test_key("xAI", "https://api.x.ai/v1/models", {"Authorization": f"Bearer {k}"})
    detail = ", ".join(m.get("id","") for m in d.get("data",[])[:3]) if s == "OK" else str(d)[:60]
    out.append(f"[{'OK' if s=='OK' else '!!'}] xAI Grok             {s:<10} {detail[:60]}")
else:
    out.append("[!!] xAI Grok             MISSING")

# Twitter/X
k = os.environ.get("TWITTER_BEARER_TOKEN", "")
if k:
    s, d = test_key("Twitter", "https://api.twitter.com/2/tweets/search/recent?query=bitcoin&max_results=10",
        {"Authorization": f"Bearer {k}"})
    detail = f"{len(d.get('data',[]))} tweets" if s == "OK" else str(d)[:60]
    out.append(f"[{'OK' if s=='OK' else '!!'}] Twitter/X            {s:<10} {detail}")
else:
    out.append("[!!] Twitter/X            MISSING")

# SMTP
u = os.environ.get("SMTP_USER",""); p = os.environ.get("SMTP_PASSWORD","")
if u and p:
    out.append(f"[OK] SMTP                OK         Host: {os.environ.get('SMTP_HOST','?')}, User: {u}")
else:
    out.append("[!!] SMTP                MISSING")

Path(__file__).parent.joinpath("val_out.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
