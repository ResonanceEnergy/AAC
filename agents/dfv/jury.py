from __future__ import annotations

"""Three-LLM jury — replaces single-Gemini second opinion.

Panelists: Gemini (Google), Grok (xAI), OpenAI. Each gets the same gate
verdict + proposal and votes AGREE | DISAGREE | CONCERN | UNCLEAR. Majority
view + dissent count surface in the dashboard 2nd-op column.

Gracefully degrades when keys missing: a jury of 1 is just the existing
behavior. Never overrides the gate verdict (advisory only).
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

VOTE_TOKENS = {"AGREE", "DISAGREE", "CONCERN", "UNCLEAR"}


def _first_token_vote(text: str) -> str:
    if not text:
        return "UNCLEAR"
    first = text.strip().split()[0].upper().rstrip(".,:!?")
    return first if first in VOTE_TOKENS else "UNCLEAR"


# ── Panelist: Gemini ──────────────────────────────────────────────
def _ask_gemini(model: str, prompt: str, system: str) -> dict[str, Any]:
    try:
        from integrations.google_clients import GeminiClient
    except ImportError:
        return {"ok": False, "error": "GeminiClient unavailable"}
    try:
        gem = GeminiClient(model=model)
        if not gem.configured:
            return {"ok": False, "error": "GEMINI_API_KEY missing"}
        result = gem.ask(prompt, system=system, temperature=0.1)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "unknown")}
    text = (result.get("text") or "").strip()
    return {"ok": True, "text": text[:600], "vote": _first_token_vote(text), "model": model}


# ── Panelist: Grok (xAI) ──────────────────────────────────────────
def _ask_grok(model: str, prompt: str, system: str) -> dict[str, Any]:
    key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY", "")
    if not key:
        return {"ok": False, "error": "XAI_API_KEY missing"}
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 250,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.x.ai/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    try:
        text = (payload["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return {"ok": False, "error": "malformed response"}
    return {"ok": True, "text": text[:600], "vote": _first_token_vote(text), "model": model}


# ── Panelist: OpenAI ──────────────────────────────────────────────
def _ask_openai(model: str, prompt: str, system: str) -> dict[str, Any]:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return {"ok": False, "error": "OPENAI_API_KEY missing"}
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 250,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    try:
        text = (payload["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return {"ok": False, "error": "malformed response"}
    return {"ok": True, "text": text[:600], "vote": _first_token_vote(text), "model": model}


_PANELIST_FNS = {
    "gemini": _ask_gemini,
    "grok": _ask_grok,
    "openai": _ask_openai,
}


def jury_review(
    *,
    symbol: str,
    proposal: dict[str, Any],
    decision: Any,
    doctrine: dict[str, Any],
) -> dict[str, Any] | None:
    """Run the configured jury panel. Returns aggregated verdict + per-panelist
    transcripts. None if disabled."""
    cfg_jury = doctrine.get("jury") or {}
    cfg_legacy = doctrine.get("second_opinion") or {}
    if not cfg_jury.get("enabled", True) and not cfg_legacy.get("enabled", True):
        return None
    panelists = list(cfg_jury.get("panelists") or ["gemini"])
    models = cfg_jury.get("models") or {}
    # Defaults if doctrine block missing
    defaults = {"gemini": "gemini-2.5-flash", "grok": "grok-3-mini", "openai": "gpt-4o-mini"}
    # Legacy second_opinion.model overrides gemini default if present
    if "model" in cfg_legacy:
        defaults["gemini"] = cfg_legacy["model"]

    gate_lines = "\n".join(
        f"  {g.gate_id} {g.name}: {g.outcome.upper()} ({g.severity}) — {g.note}"
        for g in (decision.gates or [])
    )
    prompt = (
        f"DFV proposal review.\n"
        f"Symbol: {symbol or '(none)'}\n"
        f"Proposal: {proposal}\n"
        f"DFV gate verdict: {decision.verdict} — {decision.summary}\n"
        f"Gates:\n{gate_lines}\n\n"
        f"In ≤60 words: do you AGREE, DISAGREE, or CONCERN? "
        f"Start your answer with one of those three words, then a one-line reason."
    )
    system = (
        "You are a second-opinion reviewer for DFV (Roaring Kitty). "
        "Be terse. Flag thesis/sizing/invalidation risk. Never bypass hard rules."
    )

    transcripts: list[dict[str, Any]] = []
    for name in panelists:
        fn = _PANELIST_FNS.get(name.lower())
        if not fn:
            transcripts.append({"provider": name, "ok": False, "error": "unknown panelist"})
            continue
        model = models.get(name, defaults.get(name, ""))
        result = fn(model, prompt, system)
        result["provider"] = name
        transcripts.append(result)

    votes = [t.get("vote") for t in transcripts if t.get("ok")]
    n_ok = len(votes)
    # Majority view
    if n_ok == 0:
        majority = "UNAVAILABLE"
    else:
        from collections import Counter
        counts = Counter(votes)
        majority, _ = counts.most_common(1)[0]
    dissent = [v for v in votes if v != majority]

    return {
        "provider": "jury",
        "ok": n_ok > 0,
        "panelists": [t.get("provider") for t in transcripts],
        "votes": votes,
        "majority": majority,
        "dissent_count": len(dissent),
        "quorum": int(cfg_jury.get("quorum", 1)),
        "transcripts": transcripts,
        # Back-compat: surface a single "verdict" so old callers keep working.
        "verdict": majority,
        "text": " | ".join(
            f"{t.get('provider')}={t.get('vote', '—')}" for t in transcripts
        ),
    }
