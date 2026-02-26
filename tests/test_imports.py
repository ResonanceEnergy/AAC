#!/usr/bin/env python3
"""
Simple test script for master agent imports
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Testing imports...")

try:
    from BigBrainIntelligence.agents import AGENT_REGISTRY
    print(f"✓ Research agents imported: {len(AGENT_REGISTRY)} agents")
except Exception as e:
    print(f"✗ Research agents import failed: {e}")

try:
    from shared.department_super_agents import DEPARTMENT_SUPER_AGENTS
    super_count = sum(len(agents) for agents in DEPARTMENT_SUPER_AGENTS.values())
    print(f"✓ Super agents imported: {super_count} agents")
except Exception as e:
    print(f"✗ Super agents import failed: {e}")

try:
    from master_agent_file import get_all_agents
    agents = get_all_agents()
    print(f"✓ Master agent file imported: {agents['total_overall']} total agents")
except Exception as e:
    print(f"✗ Master agent file import failed: {e}")

print("Import test complete.")