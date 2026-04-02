#!/usr/bin/env python3
"""Gap Analysis Tool for AAC Doctrine System"""

import asyncio
import logging
import sys

logger = logging.getLogger(__name__)
sys.path.insert(0, '.')

from aac.doctrine.doctrine_engine import DOCTRINE_PACKS
from aac.doctrine.doctrine_integration import DoctrineOrchestrator


async def main():
    """Main."""
    logger.info("=" * 70)
    logger.info("AAC DOCTRINE GAP ANALYSIS")
    logger.info("=" * 70)

    orch = DoctrineOrchestrator()

    # Gather metrics from hardcoded DOCTRINE_PACKS (used for lookups)
    hardcoded_metrics = set()
    for pack_id, pack_info in DOCTRINE_PACKS.items():
        for m in pack_info.get('key_metrics', []):
            hardcoded_metrics.add(m)

    # Gather metrics from adapters
    adapter_metrics = set()
    for dept, adapter in orch.adapters.items():
        try:
            metrics = await adapter.get_metrics()
            adapter_metrics.update(metrics.keys())
            logger.info(f"  {dept.value}: {len(metrics)} metrics")
        except Exception as e:
            logger.info(f"  {dept.value}: ERROR - {e}")

    logger.info(f"\n{'='*70}")
    logger.info(f"METRIC SOURCE ANALYSIS")
    logger.info(f"{'='*70}")
    logger.info(f"  Hardcoded DOCTRINE_PACKS: {len(hardcoded_metrics)} metrics")
    logger.info(f"  Adapters provide:         {len(adapter_metrics)} metrics")

    # Gap analysis
    in_doctrine_not_adapter = hardcoded_metrics - adapter_metrics
    in_adapter_not_doctrine = adapter_metrics - hardcoded_metrics
    matched = hardcoded_metrics & adapter_metrics

    logger.info(f"\n{'='*70}")
    logger.info(f"GAP ANALYSIS RESULTS")
    logger.info(f"{'='*70}")

    if in_doctrine_not_adapter:
        logger.info(f"\n[CROSS] IN DOCTRINE_PACKS BUT NOT IN ADAPTERS ({len(in_doctrine_not_adapter)}):")
        for m in sorted(in_doctrine_not_adapter):
            logger.info(f"    MISSING IN ADAPTER: {m}")
    else:
        logger.info(f"\n✅ All DOCTRINE_PACKS metrics have adapter coverage!")

    if in_adapter_not_doctrine:
        logger.info(f"\n[WARN]️  IN ADAPTERS BUT NOT IN DOCTRINE_PACKS ({len(in_adapter_not_doctrine)}):")
        for m in sorted(in_adapter_not_doctrine):
            logger.info(f"    EXTRA (not in packs): {m}")
    else:
        logger.info(f"\n✅ All adapter metrics are defined in DOCTRINE_PACKS!")

    logger.info(f"\n{'='*70}")
    logger.info(f"COVERAGE SUMMARY")
    logger.info(f"{'='*70}")
    coverage = 100 * len(matched) / len(hardcoded_metrics) if hardcoded_metrics else 0
    logger.info(f"  Matched:    {len(matched)}/{len(hardcoded_metrics)} = {coverage:.1f}%")
    logger.info(f"  Missing:    {len(in_doctrine_not_adapter)}")
    logger.info(f"  Extra:      {len(in_adapter_not_doctrine)}")

    total_gaps = len(in_doctrine_not_adapter) + len(in_adapter_not_doctrine)

    if total_gaps == 0:
        logger.info("\n[CELEBRATION] PERFECT ALIGNMENT - NO GAPS!")
    else:
        logger.info(f"\n[WARN]️  {total_gaps} total gaps need resolution")

    # List all matched metrics by pack
    logger.info(f"\n{'='*70}")
    logger.info(f"MATCHED METRICS BY PACK")
    logger.info(f"{'='*70}")
    for pack_id, pack_info in sorted(DOCTRINE_PACKS.items()):
        pack_metrics = set(pack_info.get('key_metrics', []))
        pack_matched = pack_metrics & adapter_metrics
        pack_missing = pack_metrics - adapter_metrics
        status = "✅" if not pack_missing else "[WARN]️"
        logger.info(f"\n  {status} Pack {pack_id}: {pack_info['name']}")
        logger.info(f"     Matched: {len(pack_matched)}/{len(pack_metrics)}")
        if pack_missing:
            for m in sorted(pack_missing):
                logger.info(f"       MISSING: {m}")

    return total_gaps

if __name__ == "__main__":
    gaps = asyncio.run(main())
    sys.exit(0 if gaps == 0 else 1)
