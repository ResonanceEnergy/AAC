"""CLI tool to inspect department connections and doctrine-pack health."""

import asyncio
from aac.doctrine.doctrine_integration import get_doctrine_integration

async def check_departments():
    print('🔍 Checking Department Connections...')

    doctrine = get_doctrine_integration()
    await doctrine.initialize()  # Initialize first
    status = doctrine.orchestrator.get_system_status()  # Use orchestrator's method

    print(f'\n[MONITOR] SYSTEM STATUS: {status["az_prime_state"]}')
    print(f'Last Check: {status.get("last_check", "Never")}')
    print(f'Monitoring Active: {status["monitoring_active"]}')

    print('\n🏢 DEPARTMENT CONNECTIONS:')
    departments = status.get('departments', {})
    for dept_name, dept_info in departments.items():
        packs = dept_info.get('doctrine_packs', [])
        health = dept_info.get('health', 'unknown')
        print(f'  • {dept_name}: {len(packs)} doctrine packs, health={health}')
        for pack in packs[:2]:  # Show first 2 packs
            print(f'    - Pack {pack["pack_id"]}: {pack["name"]}')
        if len(packs) > 2:
            print(f'    ... and {len(packs)-2} more packs')

    print(f'\n✅ Found {len(departments)} connected departments!')

if __name__ == "__main__":
    asyncio.run(check_departments())