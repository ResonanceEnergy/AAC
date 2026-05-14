"""CLI for the AAC agent swarm.

Examples:
    python -m shared.aac_agents researcher "How does the IBKR connector authenticate?"
    python -m shared.aac_agents monitor "Brief me on this week"
    python -m shared.aac_agents planner "What should I focus on Friday?"
    python -m shared.aac_agents list
"""

from __future__ import annotations

import argparse
import json
import sys

from .runtime import AGENTS, run_agent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aac_agents", description="AAC local agent swarm")
    p.add_argument("agent", help="Agent name or 'list' to enumerate.")
    p.add_argument("prompt", nargs="?", default="", help="User message for the agent.")
    p.add_argument("--verbose", "-v", action="store_true", help="Log tool calls.")
    p.add_argument("--json", action="store_true", help="Emit full JSON result.")
    args = p.parse_args(argv)

    if args.agent == "list":
        for name, spec in AGENTS.items():
            print(f"  {name:<12} tools={sorted(spec.allowed_tools)}")
        return 0

    if not args.prompt:
        p.error("prompt is required (unless agent=list)")

    result = run_agent(args.agent, args.prompt, verbose=args.verbose)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    print(f"\n=== {result['agent'].upper()} ===")
    if result["tool_calls"]:
        print(f"\n[{len(result['tool_calls'])} tool call(s)]")
        for tc in result["tool_calls"]:
            args_repr = json.dumps(tc["arguments"], default=str)[:120]
            print(f"  step {tc['step']}: {tc['tool']}({args_repr})")
    print("\n--- ANSWER ---")
    print(result["answer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
