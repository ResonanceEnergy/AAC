"""AAC Agent Swarm — local LLM agents over the RAG index + calendar.

Three agents share one runtime:
  * researcher — answers code/architecture questions using RAG over the codebase
  * monitor    — synthesizes calendar + watchlist into daily situational briefings
  * planner    — multi-step orchestrator that calls both RAG + calendar tools

All agents run on local Ollama (qwen2.5-coder:7b) with native tool-calling.

Public API:
    from shared.aac_agents import run_agent
    result = run_agent("researcher", "How does the IBKR connector authenticate?")
    print(result["answer"])

CLI:
    python -m shared.aac_agents researcher "How does the IBKR connector authenticate?"
    python -m shared.aac_agents monitor "What's on my radar this week?"
    python -m shared.aac_agents planner "Plan my Friday based on watchlist + calendar"
"""

from __future__ import annotations

from .runtime import AGENTS, run_agent

__all__ = ["AGENTS", "run_agent"]
