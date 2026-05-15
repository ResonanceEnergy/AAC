from __future__ import annotations

"""DFV LLM voice — Ollama tool-calling loop with the DFV persona.

Wires four things together:
  1. `persona.md` is injected as the system prompt (so the model speaks as DFV).
  2. The local Ollama model (per `config/rag.yaml`) drives the conversation.
  3. A tool registry exposes DFV's structured memory + decision engine to the
     model: `decide`, `review`, `set_thesis`, `get_thesis`, `list_theses`,
     `list_watchlist`, `recent_decisions`, `recall_memory` (semantic RAG).
  4. Persistence: every Q/A turn is appended to a rolling history file.

Public surface:
    from agents.dfv.llm import ask
    answer = ask("Should we add to GME at 18?")
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import structlog

_log = structlog.get_logger().bind(component="dfv.llm")

REPO_ROOT = Path(__file__).resolve().parents[2]
PERSONA_PATH = REPO_ROOT / "agents" / "dfv" / "persona.md"
HISTORY_PATH = REPO_ROOT / "agents" / "dfv" / "memory" / "llm_history.jsonl"
DEFAULT_MAX_STEPS = 6
DEFAULT_TEMPERATURE = 0.4  # DFV has voice — not as deterministic as code-search


# ── Tool implementations (thin wrappers around the structured stores) ─────


def _dfv():
    from agents.dfv.decision_engine import DFV
    return DFV()


def _decide_proposal(
    symbol: str,
    side: str = "long",
    size_pct_book: float = 0.0,
    invalidation: str = "",
    catalyst_days_out: int | None = None,
    cash_pct_after: float = 0.20,
    correlation_factor_pct: float = 0.0,
    estimated_slippage_pct: float = 0.005,
) -> dict[str, Any]:
    """Run a structured trade proposal through the seven gates."""
    proposal = {
        "symbol": symbol.upper(),
        "side": side,
        "size_pct_book": float(size_pct_book),
        "invalidation": invalidation,
        "catalyst_days_out": catalyst_days_out,
        "cash_pct_after": float(cash_pct_after),
        "correlation_factor_pct": float(correlation_factor_pct),
        "estimated_slippage_pct": float(estimated_slippage_pct),
    }
    d = _dfv().evaluate(proposal)
    return d.to_dict()


def _review_text(text: str) -> dict[str, Any]:
    """Run free-text prompt through the seven gates (FOMO + thesis triggers)."""
    return _dfv().review_prompt(text).to_dict()


def _list_theses() -> dict[str, Any]:
    dfv = _dfv()
    return {
        "count": len(dfv.thesis.all()),
        "theses": [
            {
                "symbol": s,
                "conviction": r.get("conviction"),
                "horizon": r.get("horizon"),
                "updated": r.get("updated"),
                "revision": r.get("revision"),
            }
            for s, r in sorted(dfv.thesis.all().items())
        ],
    }


def _get_thesis(symbol: str) -> dict[str, Any]:
    dfv = _dfv()
    rec = dfv.thesis.get(symbol)
    return rec or {"error": f"No thesis on file for {symbol.upper()}"}


def _set_thesis(
    symbol: str,
    thesis: str,
    conviction: int,
    horizon: str,
    catalysts: list[str] | str,
    invalidation: str,
    target: str = "",
    max_pct_book: float = 0.03,
) -> dict[str, Any]:
    dfv = _dfv()
    if isinstance(catalysts, str):
        catalysts = [c.strip() for c in catalysts.split("|") if c.strip()]
    rec = dfv.thesis.set(
        symbol,
        thesis=thesis,
        conviction=int(conviction),
        horizon=horizon,
        catalysts=list(catalysts),
        invalidation=invalidation,
        target={"raw": target},
        sizing={"max_pct_book": float(max_pct_book)},
    )
    # Auto-index the new thesis into RAG memory (best-effort)
    try:
        from agents.dfv import rag as dfv_rag
        dfv_rag.index_thesis(symbol, rec)
    except Exception as e:  # noqa: BLE001
        _log.warning("dfv.llm.index_thesis_failed", error=str(e))
    return rec


def _list_watchlist() -> dict[str, Any]:
    dfv = _dfv()
    return {"watchlist": dfv.watchlist.all()}


def _recent_decisions(n: int = 10) -> dict[str, Any]:
    dfv = _dfv()
    return {"decisions": dfv.decisions.tail(int(n))}


def _recall_memory(query: str, k: int = 5, kind: str | None = None,
                   symbol: str | None = None) -> dict[str, Any]:
    """Semantic search over DFV's brief / thesis history."""
    from agents.dfv import rag as dfv_rag
    hits = dfv_rag.search(query, k=k, kind=kind, symbol=symbol)
    return {"count": len(hits), "results": hits}


_GROK_DEFAULT_MODEL = "grok-3-mini"
_GROK_MAX_OUTPUT_CHARS = 6000


def _grok_ask(
    prompt: str,
    system: str = "",
    model: str = _GROK_DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Call xAI Grok over HTTPS. Returns {ok, model, text, error?}."""
    import os
    import httpx

    # Prefer .env value (in-repo, fresh) over a stale OS env XAI_API_KEY.
    key = ""
    try:
        from dotenv import dotenv_values
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            vals = dotenv_values(str(env_path))
            key = (vals.get("XAI_API_KEY") or vals.get("GROK_API_KEY") or "").strip()
    except ImportError:
        pass
    if not key:
        key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY") or ""
    if not key:
        return {"ok": False, "error": "XAI_API_KEY not set", "text": ""}

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        if r.status_code >= 400:
            return {
                "ok": False,
                "model": model,
                "error": f"HTTP {r.status_code}: {r.text[:400]}",
                "text": "",
            }
        data = r.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            or ""
        )
        if len(text) > _GROK_MAX_OUTPUT_CHARS:
            text = text[:_GROK_MAX_OUTPUT_CHARS] + "\n\u2026[truncated]"
        usage = data.get("usage", {})
        return {
            "ok": True,
            "model": model,
            "text": text,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
    except httpx.HTTPError as e:
        return {"ok": False, "model": model, "error": f"http: {e}", "text": ""}
    except (ValueError, KeyError) as e:
        return {"ok": False, "model": model, "error": f"parse: {e}", "text": ""}


TOOLS: dict[str, Callable[..., Any]] = {
    "decide_proposal": _decide_proposal,
    "review_text": _review_text,
    "list_theses": _list_theses,
    "get_thesis": _get_thesis,
    "set_thesis": _set_thesis,
    "list_watchlist": _list_watchlist,
    "recent_decisions": _recent_decisions,
    "recall_memory": _recall_memory,
    "grok_ask": _grok_ask,
}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "decide_proposal",
            "description": (
                "Run a structured trade proposal through DFV's seven gates. "
                "Use this whenever the user proposes a specific trade (ticker + size)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "enum": ["long", "short"], "default": "long"},
                    "size_pct_book": {"type": "number", "description": "0.05 = 5% of book"},
                    "invalidation": {"type": "string"},
                    "catalyst_days_out": {"type": "integer"},
                    "cash_pct_after": {"type": "number", "default": 0.20},
                    "correlation_factor_pct": {"type": "number", "default": 0.0},
                    "estimated_slippage_pct": {"type": "number", "default": 0.005},
                },
                "required": ["symbol", "size_pct_book"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_text",
            "description": (
                "Run a free-text prompt through the seven gates. Detects FOMO, "
                "missing thesis, and other operator errors. Use for ambiguous user requests."
            ),
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_theses",
            "description": "List every ticker DFV has a written thesis on.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_thesis",
            "description": "Fetch the full thesis record for a single ticker.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_thesis",
            "description": (
                "Write or update a thesis. ALL fields are required by DFV doctrine "
                "(G1 Thesis + G6 Invalidation are hard gates)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "thesis": {"type": "string", "description": "Why is this mispriced? <= 200 words."},
                    "conviction": {"type": "integer", "description": "1 (starter) to 5 (max conviction)"},
                    "horizon": {"type": "string", "description": "e.g. '12-24 months'"},
                    "catalysts": {"type": "string", "description": "Pipe-separated list of catalysts."},
                    "invalidation": {"type": "string", "description": "What proves the thesis wrong."},
                    "target": {"type": "string", "description": "Fair value range + method."},
                    "max_pct_book": {"type": "number", "default": 0.03},
                },
                "required": ["symbol", "thesis", "conviction", "horizon",
                             "catalysts", "invalidation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_watchlist",
            "description": "List tickers DFV is watching but not yet long.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recent_decisions",
            "description": "Tail of the gate decision log (most recent N entries).",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer", "default": 10}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Semantic search DFV's long-term memory (past briefs + thesis revisions). "
                "Use this to answer 'when did I last see this setup?' / 'what did I "
                "decide on TICKER before?' / 'what did the brief say about credit spreads?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5},
                    "kind": {"type": "string", "enum": ["brief", "thesis", "decision", "note"]},
                    "symbol": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grok_ask",
            "description": (
                "Ask xAI Grok over HTTPS. Use sparingly -- costs money and adds latency. "
                "Good for: (a) live web / X / news context the local model lacks, "
                "(b) a second opinion on a thesis or trade idea, (c) long-form reasoning "
                "or recent events qwen2.5-coder is weak at. Treat the answer as ONE INPUT "
                "-- still apply the seven gates yourself. Do NOT pass account numbers, "
                "private position sizes, or PII in the prompt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "User question for Grok."},
                    "system": {"type": "string", "description": "Optional system message to steer Grok."},
                    "model": {"type": "string", "default": "grok-3-mini",
                              "description": "xAI model id (grok-3-mini, grok-3, grok-4 if entitled)."},
                    "temperature": {"type": "number", "default": 0.3},
                    "max_tokens": {"type": "integer", "default": 1024},
                },
                "required": ["prompt"],
            },
        },
    },
]


# ── Persona system prompt ─────────────────────────────────────────────────


def _load_persona() -> str:
    try:
        return PERSONA_PATH.read_text(encoding="utf-8")
    except OSError as e:
        _log.warning("dfv.llm.persona_read_failed", error=str(e))
        return "You are DFV (Roaring Kitty), the Prime Operator on AAC."


def _system_prompt() -> str:
    persona = _load_persona()
    return (
        persona
        + "\n\n"
        + "## Operating instructions for this session\n\n"
        + "You are running inside the AAC platform. You have access to your own structured "
        + "memory and decision engine via the tools below. **Use them.** Never make up theses, "
        + "conviction levels, or recent decisions — call the tools to read the real records.\n\n"
        + "Workflow:\n"
        + "1. If the user proposes a trade, FIRST call `get_thesis` for the ticker. "
        + "If no thesis exists, refuse and either call `set_thesis` (after gathering the "
        + "required fields from the user) or tell the user to write one.\n"
        + "2. If the user asks an open question about positioning or history, call "
        + "`recall_memory` and `list_theses` before answering.\n"
        + "3. For any structured trade idea, ALWAYS run `decide_proposal` and quote the "
        + "verdict + which gates passed/failed.\n"
        + "4. For any vague or hype-y prompt, run `review_text` first.\n"
        + "5. `grok_ask` is your escape hatch to xAI Grok -- use it when you need live "
        + "web/X context, recent news, or a stronger second opinion. Never trust Grok "
        + "blindly: its answer is one input, the gates still rule. Never pass account "
        + "numbers or private position sizes into Grok prompts.\n\n"
        + "Voice rules (binding):\n"
        + "- Plainspoken. Numbers > narratives.\n"
        + "- Lead with the headline in one line.\n"
        + "- Show the work in a tight table.\n"
        + "- State the decision with which gates cleared/failed.\n"
        + "- End with the next action (autonomous vs needs human OK).\n"
        + "- Never approve a trade because of FOMO.\n"
        + "- Cash is a position. Dry powder is sacred.\n"
    )


# ── JSON-in-text tool-call extractor (qwen2.5-coder fallback) ─────────────


_TOOL_CALL_HINT_RE = re.compile(r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"')


def _extract_text_tool_calls(text: str, allowed: set[str]) -> list[dict[str, Any]]:
    """Many local models emit tool calls as JSON inside content. Salvage them."""
    cleaned = re.sub(r"```(?:json|tool_call)?\s*", "", text)
    cleaned = cleaned.replace("```", "")
    calls: list[dict[str, Any]] = []
    for m in _TOOL_CALL_HINT_RE.finditer(cleaned):
        name = m.group(1)
        if name not in allowed:
            continue
        start = cleaned.rfind("{", 0, m.start())
        if start < 0:
            continue
        depth = 0
        end = -1
        in_str = False
        esc = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            continue
        try:
            obj = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("name") != name:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        calls.append({"function": {"name": name, "arguments": args or {}}})
    # Dedupe identical calls
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for c in calls:
        key = json.dumps(c, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


# ── Persistence ──────────────────────────────────────────────────────────


def _append_history(prompt: str, answer: str, tool_calls: list[dict[str, Any]]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt": prompt[:1000],
        "answer": answer[:4000],
        "tool_call_count": len(tool_calls),
        "tools_called": [tc.get("function", {}).get("name") for tc in tool_calls],
    }
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _recent_history(n: int = 3) -> str:
    """Compact summary of last N turns for prompt injection."""
    if not HISTORY_PATH.exists():
        return ""
    try:
        lines = HISTORY_PATH.read_text(encoding="utf-8").strip().splitlines()[-n:]
    except OSError:
        return ""
    if not lines:
        return ""
    out = ["## Previous DFV sessions (most recent last)"]
    for ln in lines:
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        ts = (r.get("ts") or "")[:10]
        prompt = (r.get("prompt") or "").replace("\n", " ")[:120]
        ans = (r.get("answer") or "").replace("\n", " ")[:200]
        out.append(f"- [{ts}] Q: {prompt} -> A: {ans}")
    return "\n".join(out)


# ── Main entrypoint ──────────────────────────────────────────────────────


@dataclass
class AskResult:
    answer: str
    steps: int
    tool_calls: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
        }


def ask(
    prompt: str,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    temperature: float = DEFAULT_TEMPERATURE,
    use_history: bool = True,
    verbose: bool = False,
) -> AskResult:
    """Ask DFV a question. He'll use his tools and answer in his own voice."""
    import ollama  # noqa: PLC0415

    from shared.aac_rag.config import RagConfig

    cfg = RagConfig.load()
    client = ollama.Client(host=cfg.generation_endpoint)
    allowed = set(TOOLS.keys())

    system = _system_prompt()
    if use_history:
        prior = _recent_history(3)
        if prior:
            system = system + "\n\n" + prior

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    tool_log: list[dict[str, Any]] = []
    answer = ""

    for step in range(max_steps):
        if verbose:
            _log.info("dfv.llm.step", step=step + 1)
        resp = client.chat(
            model=cfg.generation_model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            options={"temperature": temperature},
        )
        msg = resp["message"]
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls and content:
            extracted = _extract_text_tool_calls(content, allowed=allowed)
            if extracted:
                tool_calls = extracted
                content = ""

        assistant_turn: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_turn["tool_calls"] = tool_calls
        messages.append(assistant_turn)

        if not tool_calls:
            answer = content
            break

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name not in TOOLS:
                result_str = json.dumps({"error": f"Unknown tool {name!r}"})
            else:
                try:
                    result = TOOLS[name](**args) if isinstance(args, dict) else TOOLS[name]()
                    result_str = json.dumps(result, default=str)[:12000]
                except TypeError as e:
                    result_str = json.dumps({"error": f"bad args: {e}"})
                except Exception as e:  # noqa: BLE001 — survival path
                    result_str = json.dumps({"error": str(e)})
            tool_log.append({"name": name, "args": args, "result_preview": result_str[:400]})
            messages.append({"role": "tool", "name": name, "content": result_str})
            if verbose:
                _log.info("dfv.llm.tool", name=name, preview=result_str[:200])

    if not answer:
        answer = "(DFV: model exhausted tool steps without a final answer.)"

    _append_history(prompt, answer, tool_log)
    return AskResult(answer=answer, steps=step + 1, tool_calls=tool_log)
