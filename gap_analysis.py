#!/usr/bin/env python3
"""Gap Analysis Tool for AAC Doctrine System"""

import sys
import asyncio
sys.path.insert(0, '.')

from aac.doctrine.doctrine_integration import DoctrineOrchestrator
from aac.doctrine.doctrine_engine import DOCTRINE_PACKS

async def main():
    print("=" * 70)
    print("AAC DOCTRINE GAP ANALYSIS")
    print("=" * 70)
    
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
            print(f"  {dept.value}: {len(metrics)} metrics")
        except Exception as e:
            print(f"  {dept.value}: ERROR - {e}")
    
    print(f"\n{'='*70}")
    print(f"METRIC SOURCE ANALYSIS")
    print(f"{'='*70}")
    print(f"  Hardcoded DOCTRINE_PACKS: {len(hardcoded_metrics)} metrics")
    print(f"  Adapters provide:         {len(adapter_metrics)} metrics")
    
    # Gap analysis
    in_doctrine_not_adapter = hardcoded_metrics - adapter_metrics
    in_adapter_not_doctrine = adapter_metrics - hardcoded_metrics
    matched = hardcoded_metrics & adapter_metrics
    
    print(f"\n{'='*70}")
    print(f"GAP ANALYSIS RESULTS")
    print(f"{'='*70}")
    
    if in_doctrine_not_adapter:
        print(f"\n❌ IN DOCTRINE_PACKS BUT NOT IN ADAPTERS ({len(in_doctrine_not_adapter)}):")
        for m in sorted(in_doctrine_not_adapter):
            print(f"    MISSING IN ADAPTER: {m}")
    else:
        print(f"\n✅ All DOCTRINE_PACKS metrics have adapter coverage!")
    
    if in_adapter_not_doctrine:
        print(f"\n⚠️  IN ADAPTERS BUT NOT IN DOCTRINE_PACKS ({len(in_adapter_not_doctrine)}):")
        for m in sorted(in_adapter_not_doctrine):
            print(f"    EXTRA (not in packs): {m}")
    else:
        print(f"\n✅ All adapter metrics are defined in DOCTRINE_PACKS!")
    
    print(f"\n{'='*70}")
    print(f"COVERAGE SUMMARY")
    print(f"{'='*70}")
    coverage = 100 * len(matched) / len(hardcoded_metrics) if hardcoded_metrics else 0
    print(f"  Matched:    {len(matched)}/{len(hardcoded_metrics)} = {coverage:.1f}%")
    print(f"  Missing:    {len(in_doctrine_not_adapter)}")
    print(f"  Extra:      {len(in_adapter_not_doctrine)}")
    
    total_gaps = len(in_doctrine_not_adapter) + len(in_adapter_not_doctrine)
    
    if total_gaps == 0:
        print("\n🎉 PERFECT ALIGNMENT - NO GAPS!")
    else:
        print(f"\n⚠️  {total_gaps} total gaps need resolution")
    
    # List all matched metrics by pack
    print(f"\n{'='*70}")
    print(f"MATCHED METRICS BY PACK")
    print(f"{'='*70}")
    for pack_id, pack_info in sorted(DOCTRINE_PACKS.items()):
        pack_metrics = set(pack_info.get('key_metrics', []))
        pack_matched = pack_metrics & adapter_metrics
        pack_missing = pack_metrics - adapter_metrics
        status = "✅" if not pack_missing else "⚠️"
        print(f"\n  {status} Pack {pack_id}: {pack_info['name']}")
        print(f"     Matched: {len(pack_matched)}/{len(pack_metrics)}")
        if pack_missing:
            for m in sorted(pack_missing):
                print(f"       MISSING: {m}")
    
    return total_gaps

if __name__ == "__main__":
    gaps = asyncio.run(main())
    sys.exit(0 if gaps == 0 else 1)
