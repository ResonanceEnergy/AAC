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


# ── Pillar specialists (TradingAgents-style analyst team) ───────────────────


_OPTIONS_STRATEGIST = AgentSpec(
    name="options_strategist",
    system_prompt=(
        "You are the AAC Options Strategist. You analyse the call-options "
        "pillar and surface actionable trades.\n\n"
        "RULES:\n"
        "- ALWAYS call get_call_options_pillar FIRST.\n"
        "- For any rich-IV ticker (IV/HV ≥ 1.20) consider covered-call or "
        "  cash-secured-put candidates; for cheap-IV (≤ 0.85) consider long "
        "  premium plays.\n"
        "- Validate the top 3 candidates with get_ibkr_iv_hv if breadth/IV "
        "  data looks stale or thin.\n"
        "- Use get_option_chain only when you need actual strikes/bids for a "
        "  specific recommended trade.\n"
        "- Use get_calendar_by_symbol to flag any earnings within 14 days.\n"
        "- Output: 1-paragraph summary, then a numbered list of up to 5 "
        "  trade ideas with ticker / structure / rationale / risk note.\n"
        "- Be terse. No filler."
    ),
    allowed_tools={
        "get_call_options_pillar", "get_ibkr_iv_hv", "get_option_chain",
        "calendar_by_symbol", "get_positions",
    },
    max_steps=6,
)


_FLOW_ANALYST = AgentSpec(
    name="flow_analyst",
    system_prompt=(
        "You are the AAC Flow Analyst. You read index breadth, options flow, "
        "and dark pool prints to call market tone and positioning.\n\n"
        "RULES:\n"
        "- ALWAYS call get_index_flow_pillar FIRST.\n"
        "- If breadth shows source != 'ibkr', also call get_ibkr_breadth to "
        "  cross-check TICK/TRIN with the broker-native feed.\n"
        "- Synthesise tone in one sentence: BULLISH | BEARISH | NEUTRAL with "
        "  a confidence in [0,1].\n"
        "- Highlight up to 5 'smart money' tickers from top_flow_tickers, "
        "  noting net premium and call/put bias.\n"
        "- Flag any dark-pool print > $100M or P/C ratio > 1.2.\n"
        "- Use get_news (no symbol) for top headlines if you need context.\n"
        "- Output: tone line, then bullets. Be terse."
    ),
    allowed_tools={
        "get_index_flow_pillar", "get_ibkr_breadth", "get_news",
        "calendar_upcoming",
    },
    max_steps=5,
)


_QUANT_ANALYST = AgentSpec(
    name="quant_analyst",
    system_prompt=(
        "You are the AAC Quant Analyst. You rank statistical signals and "
        "interpret backtest evidence.\n\n"
        "RULES:\n"
        "- ALWAYS call get_quant_research_pillar FIRST (walk_forward=false).\n"
        "- Rank vol_premium_signals by signal strength + hit rate of their "
        "  source.\n"
        "- For the simple_backtest section, flag any strategy with n_signals "
        "  < 30 as 'low confidence'.\n"
        "- If a signal looks promising, optionally re-run with "
        "  walk_forward=true (only ONCE per session — it's slow).\n"
        "- Use rag_search to find the strategy implementation if the user "
        "  asks how a signal works.\n"
        "- Output: top-5 ranked signals with rationale, then 1 paragraph on "
        "  best/worst backtest. Be terse."
    ),
    allowed_tools={
        "get_quant_research_pillar", "rag_search", "get_recent_decisions",
    },
    max_steps=5,
)


# ── Cross-cutting researchers (TradingAgents debate pattern) ────────────────


_BULL_RESEARCHER = AgentSpec(
    name="bull_researcher",
    system_prompt=(
        "You are the AAC Bull Researcher. Given the analyst reports for the "
        "three pillars (call options, index flow, quant research), construct "
        "the strongest BULLISH thesis you can defend with the data.\n\n"
        "RULES:\n"
        "- Call get_recent_decisions(n=3) once to see prior calls (don't "
        "  blindly contradict yourself).\n"
        "- Cite specific numbers from the analyst reports embedded in the "
        "  user message (e.g. 'rich-IV count: 6', 'TICK +850').\n"
        "- Acknowledge ONE clear bear risk and explain why you discount it.\n"
        "- 1 short paragraph (≤ 6 sentences). End with: 'Bull confidence: 0.X'."
    ),
    allowed_tools={"get_recent_decisions"},
    max_steps=2,
)


_BEAR_RESEARCHER = AgentSpec(
    name="bear_researcher",
    system_prompt=(
        "You are the AAC Bear Researcher. Given the analyst reports for the "
        "three pillars, construct the strongest BEARISH thesis you can defend "
        "with the data.\n\n"
        "RULES:\n"
        "- Call get_recent_decisions(n=3) once.\n"
        "- Cite specific numbers from the analyst reports.\n"
        "- Acknowledge ONE clear bull risk and explain why you discount it.\n"
        "- 1 short paragraph (≤ 6 sentences). End with: 'Bear confidence: 0.X'."
    ),
    allowed_tools={"get_recent_decisions"},
    max_steps=2,
)


_PORTFOLIO_MANAGER = AgentSpec(
    name="portfolio_manager",
    system_prompt=(
        "You are the AAC Portfolio Manager. You receive the bull and bear "
        "theses plus the three analyst reports, and you make the FINAL "
        "approve/reject + sizing call.\n\n"
        "MANDATORY checks (call each ONCE, in this order):\n"
        "1. get_drawdown_state — if tripped, you MUST reject.\n"
        "2. get_daily_loss_status — if tripped, you MUST reject.\n"
        "3. get_position_exposure — read total exposure + top-5 holdings.\n"
        "4. get_correlation_regime — if regime='contagion', size DOWN ≥50% "
        "or reject non-compulsory ideas.\n"
        "5. get_recent_decisions(n=5) — learn from prior calls.\n\n"
        "OUTPUT FORMAT (strict — the orchestrator parses this):\n"
        "- 'Decision: APPROVE' or 'Decision: REJECT'\n"
        "- 'Size: X% of portfolio' (0–10%, integer)\n"
        "- 'Confidence: 0.X' (0.0–1.0)\n"
        "- 1 short paragraph rationale citing the gates above.\n"
        "- If REJECT, name the specific gate(s) that fired."
    ),
    allowed_tools={
        "get_drawdown_state",
        "get_daily_loss_status",
        "get_position_exposure",
        "get_correlation_regime",
        "get_recent_decisions",
    },
    max_steps=8,
)


AGENTS["options_strategist"] = _OPTIONS_STRATEGIST
AGENTS["flow_analyst"] = _FLOW_ANALYST
AGENTS["quant_analyst"] = _QUANT_ANALYST
AGENTS["bull_researcher"] = _BULL_RESEARCHER
AGENTS["bear_researcher"] = _BEAR_RESEARCHER
AGENTS["portfolio_manager"] = _PORTFOLIO_MANAGER


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


def run_debate(
    *,
    symbol: str | None = None,
    extra_question: str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Run the TradingAgents-style 3-analyst → bull/bear debate pipeline.

    Steps:
      1. Run options_strategist, flow_analyst, quant_analyst.
      2. Hand all three reports to bull_researcher and bear_researcher.
      3. Compose a verdict from the two confidences.
      4. (Optional) Append the verdict to data/memory/decisions.md.

    Returns a dict with `analysts`, `bull`, `bear`, `verdict`, `confidence`,
    `decision_id` (if persisted).
    """
    question = extra_question or (
        f"Trading-day decision for {symbol}" if symbol else "Trading-day decision"
    )

    analyst_prompts = {
        "options_strategist": "Run your standard options-pillar scan and report.",
        "flow_analyst": "Run your standard index/flow scan and report.",
        "quant_analyst": "Run your standard quant scan and report (no walk-forward).",
    }

    analyst_results: dict[str, dict[str, Any]] = {}
    for agent_name, p in analyst_prompts.items():
        try:
            analyst_results[agent_name] = run_agent(agent_name, p, use_history=False)
        except Exception as exc:
            _log.warning("debate_analyst_failed", agent=agent_name, error=str(exc))
            analyst_results[agent_name] = {"agent": agent_name, "answer": f"(failed: {exc})", "tool_calls": []}

    def _summary(r: dict[str, Any]) -> str:
        ans = (r.get("answer") or "").strip()
        return ans if len(ans) <= 1500 else ans[:1500] + "…"

    debate_brief = (
        f"QUESTION: {question}\n\n"
        f"=== OPTIONS STRATEGIST REPORT ===\n{_summary(analyst_results['options_strategist'])}\n\n"
        f"=== FLOW ANALYST REPORT ===\n{_summary(analyst_results['flow_analyst'])}\n\n"
        f"=== QUANT ANALYST REPORT ===\n{_summary(analyst_results['quant_analyst'])}\n"
    )

    try:
        bull = run_agent("bull_researcher", debate_brief, use_history=False)
    except Exception as exc:
        bull = {"answer": f"(bull failed: {exc})", "tool_calls": []}
    try:
        bear = run_agent("bear_researcher", debate_brief, use_history=False)
    except Exception as exc:
        bear = {"answer": f"(bear failed: {exc})", "tool_calls": []}

    bull_conf = _extract_confidence(bull.get("answer") or "")
    bear_conf = _extract_confidence(bear.get("answer") or "")
    verdict, final_conf = _compose_verdict(bull_conf, bear_conf)

    decision_id: str | None = None
    if persist:
        try:
            from .memory import append_decision  # noqa: PLC0415

            tools_used = sorted({
                tc["tool"] for r in analyst_results.values()
                for tc in (r.get("tool_calls") or [])
            })
            thesis = (
                f"BULL: {(bull.get('answer') or '').strip()}\n\n"
                f"BEAR: {(bear.get('answer') or '').strip()}"
            )
            entry = append_decision(
                kind="bull_bear_debate",
                thesis=thesis,
                verdict=verdict,
                symbol=symbol,
                confidence=final_conf,
                tools=tools_used,
                extras={"bull_conf": bull_conf, "bear_conf": bear_conf, "question": question},
            )
            decision_id = entry.get("id")
        except Exception as exc:
            _log.warning("debate_persist_failed", error=str(exc))

    return {
        "question": question,
        "symbol": symbol,
        "analysts": analyst_results,
        "bull": bull,
        "bear": bear,
        "verdict": verdict,
        "confidence": final_conf,
        "bull_confidence": bull_conf,
        "bear_confidence": bear_conf,
        "decision_id": decision_id,
    }


def _extract_confidence(text: str) -> float | None:
    m = re.search(r"confidence[:\s]+([01](?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            return None
    return None


def _compose_verdict(bull_conf: float | None, bear_conf: float | None) -> tuple[str, float]:
    b = bull_conf if isinstance(bull_conf, float) else 0.5
    r = bear_conf if isinstance(bear_conf, float) else 0.5
    diff = b - r
    if diff >= 0.15:
        return "bullish", round(min(1.0, b), 2)
    if diff <= -0.15:
        return "bearish", round(min(1.0, r), 2)
    return "neutral", round((b + r) / 2, 2)


_PM_DECISION_RE = re.compile(r"decision[:\s]+(approve|reject)", re.IGNORECASE)
_PM_SIZE_RE = re.compile(r"size[:\s]+([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)


def _parse_pm_output(text: str) -> dict[str, Any]:
    """Extract decision/size/confidence from the portfolio_manager's reply."""
    out: dict[str, Any] = {"decision": None, "size_pct": None, "confidence": None}
    if not text:
        return out
    m = _PM_DECISION_RE.search(text)
    if m:
        out["decision"] = m.group(1).lower()
    m = _PM_SIZE_RE.search(text)
    if m:
        try:
            out["size_pct"] = max(0.0, min(100.0, float(m.group(1))))
        except ValueError:
            pass
    out["confidence"] = _extract_confidence(text)
    return out


def run_portfolio_decision(
    *,
    symbol: str | None = None,
    extra_question: str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Full pipeline: 3 analysts → bull/bear debate → portfolio_manager
    final approve/reject + sizing.

    Returns the debate dict augmented with `pm` (full agent result) and `pm_decision`
    (parsed {decision, size_pct, confidence}). Persists the PM verdict to
    data/memory/decisions.md (kind='portfolio_manager') when persist=True.
    """
    debate = run_debate(symbol=symbol, extra_question=extra_question, persist=persist)

    pm_brief = (
        f"QUESTION: {debate.get('question')}\n"
        f"SYMBOL: {symbol or '(portfolio-wide)'}\n\n"
        f"=== DEBATE VERDICT ===\n"
        f"{debate.get('verdict', '?').upper()} (confidence {debate.get('confidence', 0):.2f}; "
        f"bull {debate.get('bull_confidence')}, bear {debate.get('bear_confidence')})\n\n"
        f"=== BULL THESIS ===\n{(debate.get('bull') or {}).get('answer', '')}\n\n"
        f"=== BEAR THESIS ===\n{(debate.get('bear') or {}).get('answer', '')}\n\n"
        "Make the final approve/reject + sizing call. Follow your output format strictly."
    )

    try:
        pm = run_agent("portfolio_manager", pm_brief, use_history=False)
    except Exception as exc:
        _log.warning("pm_failed", error=str(exc))
        pm = {"answer": f"(portfolio_manager failed: {exc})", "tool_calls": []}

    parsed = _parse_pm_output(pm.get("answer") or "")

    pm_decision_id: str | None = None
    if persist:
        try:
            from .memory import append_decision  # noqa: PLC0415

            tools_used = sorted({tc["tool"] for tc in (pm.get("tool_calls") or [])})
            entry = append_decision(
                kind="portfolio_manager",
                thesis=(pm.get("answer") or "").strip(),
                verdict=parsed.get("decision") or "unknown",
                symbol=symbol,
                confidence=parsed.get("confidence") or debate.get("confidence"),
                tools=tools_used,
                extras={
                    "size_pct": parsed.get("size_pct"),
                    "debate_decision_id": debate.get("decision_id"),
                    "debate_verdict": debate.get("verdict"),
                },
            )
            pm_decision_id = entry.get("id")
        except Exception as exc:
            _log.warning("pm_persist_failed", error=str(exc))

    debate["pm"] = pm
    debate["pm_decision"] = parsed
    debate["pm_decision_id"] = pm_decision_id
    return debate


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
