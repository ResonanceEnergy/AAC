"""CLI tool to run a compliance check and surface warning-level metrics."""

import asyncio
from aac.doctrine.doctrine_integration import DoctrineOrchestrator
import logging

logger = logging.getLogger(__name__)

async def main():
    """Main."""
    orch = DoctrineOrchestrator()
    result = await orch.run_compliance_check()
    logger.info("\n=== WARNING METRICS ===")
    for m in result.get('details', []):
        if m.get('status') == 'warning':
            logger.info(f"{m['metric']}: value={m.get('value')} | message={m.get('message')}")

if __name__ == "__main__":
    asyncio.run(main())
