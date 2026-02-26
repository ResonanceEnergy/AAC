#!/usr/bin/env python3
"""
PowerShell Integration Wrapper for BigBrain Agents
==================================================

This script provides a simple interface for PowerShell scripts to call
BigBrainIntelligence agents without complex inline Python code.
"""

import sys
import asyncio
import os
from typing import List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from BigBrainIntelligence.agents import get_agent


async def run_agent_scan(agent_name: str) -> List[str]:
    """Run an agent scan and return formatted results."""
    try:
        agent = get_agent(agent_name)
        if not agent:
            return [f"ERROR: Agent '{agent_name}' not found"]

        findings = await agent.run_scan()
        results = []

        for finding in findings:
            if agent_name == 'api_scanner':
                # Format: title|confidence|source
                source = finding.data.get('source', 'api')
                results.append(f"{finding.title}|{finding.confidence}|{source}")
            elif agent_name == 'data_gap_finder':
                # Format: area|confidence
                area = finding.data.get('area', 'unknown')
                results.append(f"{area}|{finding.confidence}")
            elif agent_name == 'access_arbitrage':
                # Format: opportunity|confidence|edge
                opportunity = finding.data.get('opportunity', 'unknown')
                edge = finding.data.get('edge_estimate', 0.5)
                results.append(f"{opportunity}|{finding.confidence}|{edge}")
            elif agent_name == 'network_mapper':
                # Format: target|confidence|connections
                target = finding.data.get('target', 'unknown')
                connections = finding.data.get('connections', 0)
                results.append(f"{target}|{finding.confidence}|{connections}")
            else:
                # Generic format
                results.append(f"{finding.title}|{finding.confidence}")

        return results

    except Exception as e:
        return [f"ERROR: {str(e)}"]


def main():
    """Main entry point for PowerShell integration."""
    if len(sys.argv) < 2:
        print("ERROR: Agent name required")
        sys.exit(1)

    agent_name = sys.argv[1]

    try:
        results = asyncio.run(run_agent_scan(agent_name))
        for result in results:
            print(result)
    except Exception as e:
        print(f"ERROR: {str(e)}")


if __name__ == "__main__":
    main()