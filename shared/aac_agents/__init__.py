"""AAC Agent Swarm — local LLM agents over the RAG index + calendar.

Generic agents (one runtime, system prompt + tool subset differ):
  * researcher — answers code/architecture questions using RAG
  * monitor    — synthesizes calendar + watchlist into daily briefings
  * planner    — multi-step orchestrator with the full tool set

Pillar specialists (TradingAgents-style analyst team):
  * options_strategist — call-options pillar (IV/HV, covered calls, puts)
  * flow_analyst       — index breadth + UW flow + dark pool
  * quant_analyst      — vol-premium signals + backtest hit rates

Cross-cutting researchers:
  * bull_researcher / bear_researcher — debate the analyst reports
  * portfolio_manager — final approve/reject + sizing with risk gates

All run on local Ollama (qwen2.5-coder:7b) with native tool-calling.

Public API:
    from shared.aac_agents import run_agent, run_debate, run_portfolio_decision
    res = run_agent("options_strategist", "Scan today's options")
    debate = run_debate(symbol="SPY")
    full = run_portfolio_decision(symbol="SPY")  # debate + PM

CLI:
    python -m shared.aac_agents researcher "How does the IBKR connector authenticate?"
"""

from __future__ import annotations

from .runtime import AGENTS, run_agent, run_debate, run_portfolio_decision

__all__ = ["AGENTS", "run_agent", "run_debate", "run_portfolio_decision"]
