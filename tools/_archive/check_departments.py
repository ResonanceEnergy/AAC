"""CLI tool to inspect department connections and doctrine-pack health."""

import asyncio
from aac.doctrine.doctrine_integration import get_doctrine_integration
import logging

logger = logging.getLogger(__name__)

async def check_departments():
    """Check departments."""
    logger.info('🔍 Checking Department Connections...')

    doctrine = get_doctrine_integration()
    await doctrine.initialize()  # Initialize first
    status = doctrine.orchestrator.get_system_status()  # Use orchestrator's method

    logger.info(f'\n[MONITOR] SYSTEM STATUS: {status["barren_wuffet_state"]}')
    logger.info(f'Last Check: {status.get("last_check", "Never")}')
    logger.info(f'Monitoring Active: {status["monitoring_active"]}')

    logger.info('\n🏢 DEPARTMENT CONNECTIONS:')
    departments = status.get('departments', {})
    for dept_name, dept_info in departments.items():
        packs = dept_info.get('doctrine_packs', [])
        health = dept_info.get('health', 'unknown')
        logger.info(f'  • {dept_name}: {len(packs)} doctrine packs, health={health}')
        for pack in packs[:2]:  # Show first 2 packs
            logger.info(f'    - Pack {pack["pack_id"]}: {pack["name"]}')
        if len(packs) > 2:
            logger.info(f'    ... and {len(packs)-2} more packs')

    logger.info(f'\n✅ Found {len(departments)} connected departments!')

if __name__ == "__main__":
    asyncio.run(check_departments())