"""Agent runtime: Ollama tool-calling loop.

Three agents share this runtime — only their system prompt + tool subset
differs. The loop:

  1. Send messages + tools to Ollama.
  2. If the model returns tool_calls, dispatch each, append `tool` messages.
  3. Repeat until the model returns content with no tool_calls (or max_steps).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from shared.aac_rag.config import RagConfig

from .history import append_history, recent_summary
from .tools import TOOL_SCHEMAS, dispatch

_log = structlog.get_logger().bind(component="aac_agents.runtime")


@dataclass
class AgentSpec:
    name: str
    system_prompt: str
    allowed_tools: set[str] = field(default_factory=set)
    max_steps: int = 6
    temperature: float = 0.2


# ── Agent definitions ───────────────────────────────────────────────────────


_RESEARCHER = AgentSpec(
    name="researcher",
    system_prompt=(
        "You are the AAC Researcher Agent. You answer questions about the "
        "Accelerated Arbitrage Corp (AAC) trading platform codebase using the "
        "rag_search and rag_ask tools.\n\n"
        "RULES:\n"
        "- ALWAYS call rag_search first to find relevant files before answering.\n"
        "- Cite file paths in backticks.\n"
        "- If the index has no answer, say so plainly. Do NOT invent code.\n"
        "- Keep answers concise: 3-8 sentences plus a short bullet list of cited files.\n"
        "- Prefer rag_search (cheaper) over rag_ask (recursive LLM call)."
    ),
    allowed_tools={"rag_search", "rag_ask"},
    max_steps=4,
)


_MONITOR = AgentSpec(
    name="monitor",
    system_prompt=(
        "You are the AAC Monitor Agent. You produce situational briefings by "
        "fusing the watchlist, the upcoming financial calendar, live "
        "positions, and current news.\n\n"
        "RULES:\n"
        "- ALWAYS call get_watchlist and calendar_upcoming(watchlist_only=True) first.\n"
        "- Call get_positions to ground the brief in real holdings (skip if it errors).\n"
        "- Call get_news (no symbol) for top-of-mind market headlines.\n"
        "- Group output by date, then by importance (CRITICAL > HIGH > MEDIUM).\n"
        "- For each event, note which watchlist symbols are exposed.\n"
        "- Flag any CRITICAL events or any event within 3 days as ACTION ITEMS.\n"
        "- End with a 1-line headline: 'Top concern this week: ...'\n"
        "- Be terse. Tabular when possible. No filler."
    ),
    allowed_tools={
        "get_watchlist", "calendar_upcoming", "calendar_by_symbol", "calendar_by_kind",
        "get_positions", "get_news",
    },
    max_steps=6,
)


_PLANNER = AgentSpec(
    name="planner",
    system_prompt=(
        "You are the AAC Planner Agent. You orchestrate research, calendar, "
        "portfolio, and market-data tools to produce actionable trading-day "
        "plans.\n\n"
        "RULES:\n"
        "- Decompose the user's request into 2-5 sub-questions.\n"
        "- Call get_positions and get_account_value FIRST to know what's actually held.\n"
        "- Use calendar_* tools for time-sensitive context (events in next N days).\n"
        "- Use get_option_chain when the plan involves rolls, strikes, or new puts.\n"
        "- Use rag_search to find relevant code/strategies/risk-rules in the repo.\n"
        "- Use get_news for symbols that have upcoming events or open positions.\n"
        "- Cite file paths (backticks) and event dates.\n"
        "- Produce a numbered action plan with owners (engine vs human) and timing."
    ),
    allowed_tools=set(t["function"]["name"] for t in TOOL_SCHEMAS),  # all
    max_steps=10,
    temperature=0.3,
)


AGENTS: dict[str, AgentSpec] = {
    "researcher": _RESEARCHER,
    "monitor": _MONITOR,
    "planner": _PLANNER,
}


# ── Runtime loop ─────────────────────────────────────────────────────────────


def _filter_tools(spec: AgentSpec) -> list[dict[str, Any]]:
    return [t for t in TOOL_SCHEMAS if t["function"]["name"] in spec.allowed_tools]


def run_agent(
    agent_name: str,
    user_message: str,
    *,
    verbose: bool = False,
    use_history: bool = True,
) -> dict[str, Any]:
    """Run an agent on a user message. Returns {answer, steps, tool_calls}."""
    if agent_name not in AGENTS:
        raise ValueError(
            f"Unknown agent '{agent_name}'. Valid: {sorted(AGENTS.keys())}"
        )
    spec = AGENTS[agent_name]
    cfg = RagConfig.load()

    import ollama  # noqa: PLC0415

    client = ollama.Client(host=cfg.generation_endpoint)
    tools = _filter_tools(spec)

    system_prompt = spec.system_prompt
    if use_history:
        prior = recent_summary(agent_name, n=3)
        if prior:
            system_prompt = f"{spec.system_prompt}\n\n{prior}"

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    tool_call_log: list[dict[str, Any]] = []
    final_answer = ""

    for step in range(spec.max_steps):
        if verbose:
            _log.info("agent_step", agent=agent_name, step=step + 1)

        resp = client.chat(
            model=cfg.generation_model,
            messages=messages,
            tools=tools,
            options={"temperature": spec.temperature},
        )
        msg = resp["message"]
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls") or []

        # Fallback: many local models (qwen2.5-coder, llama3) emit tool calls as
        # JSON inside the content field instead of via the native tool_calls slot.
        if not tool_calls and content:
            extracted = _extract_text_tool_calls(content, allowed=spec.allowed_tools)
            if extracted:
                tool_calls = extracted
                content = ""  # don't echo the raw JSON back as content

        # Persist the assistant turn (with tool_calls if any)
        assistant_turn: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_turn["tool_calls"] = tool_calls
        messages.append(assistant_turn)

        if not tool_calls:
            final_answer = content
            break

        # Dispatch each tool call and feed results back
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if verbose:
                _log.info("tool_call", agent=agent_name, tool=name, args=args)
            result_json = dispatch(name, args)
            tool_call_log.append({
                "step": step + 1,
                "tool": name,
                "arguments": args if isinstance(args, dict) else _safe_json_load(args),
                "result_preview": result_json[:400],
            })
            messages.append({
                "role": "tool",
                "name": name,
                "content": result_json,
            })
    else:
        final_answer = (
            "(agent halted: max_steps reached without final answer; partial tool "
            "results above)"
        )

    result = {
        "agent": agent_name,
        "answer": final_answer,
        "steps_used": len(tool_call_log) + (1 if final_answer else 0),
        "tool_calls": tool_call_log,
    }
    if use_history and final_answer:
        append_history(agent_name, user_message, final_answer, len(tool_call_log))
    return result


def _safe_json_load(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


# Match a JSON object that looks like {"name": "...", "arguments": {...}}.
# Use a depth-aware scanner instead of a fragile regex for nested braces.
_TOOL_CALL_HINT_RE = re.compile(r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"')


def _extract_text_tool_calls(text: str, *, allowed: set[str]) -> list[dict[str, Any]]:
    """Pull JSON tool-call objects out of a free-text model response.

    Handles raw JSON, ```json fenced blocks, and multiple calls in one response.
    Only returns calls whose `name` is in `allowed`.
    """
    # Strip code fences so brace scanning works uniformly
    cleaned = re.sub(r"```(?:json|tool_call)?\s*", "", text)
    cleaned = cleaned.replace("```", "")

    calls: list[dict[str, Any]] = []
    for m in _TOOL_CALL_HINT_RE.finditer(cleaned):
        name = m.group(1)
        if name not in allowed:
            continue
        # Walk back to the opening brace of the enclosing object
        start = cleaned.rfind("{", 0, m.start())
        if start < 0:
            continue
        # Walk forward, tracking brace depth, to find the matching close
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
        blob = cleaned[start : end + 1]
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("name") != name:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        calls.append({"function": {"name": name, "arguments": args}})

    # Dedupe identical (name, args) pairs preserving order
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for c in calls:
        key = json.dumps(c, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique
