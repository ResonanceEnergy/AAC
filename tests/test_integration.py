#!/usr/bin/env python3
"""
Quick Integration Test
"""

import asyncio
from agent_based_trading_integration import AACAgentIntegration

async def test_integration():
    print("Testing AAC Agent Integration...")

    try:
        integration = AACAgentIntegration()
        success = await integration.initialize_integration()
        status = 'SUCCESS' if success else 'FAILED'
        print(f'Integration initialization: {status}')

        if success:
            orchestrator = await integration.create_integrated_contest()
            await orchestrator.initialize_contest()
            print(f'Contest initialization: SUCCESS ({len(orchestrator.agents)} agents created)')

            # Get status without running rounds
            status = await orchestrator.get_contest_status()
            active_agents = status.get('active_agents', 0)
            print(f'Contest status: {active_agents} active agents')

            await integration.shutdown_integration()
            print("Integration test completed successfully")
            return True
        else:
            print("Integration test failed")
            return False

    except Exception as e:
        print(f"Integration test failed with error: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_integration())
    exit(0 if result else 1)