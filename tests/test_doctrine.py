import asyncio
from aac.doctrine.doctrine_integration import DoctrineOrchestrator

async def test_doctrine():
    """Test doctrine orchestrator initialization."""
    try:
        orchestrator = DoctrineOrchestrator()
        success = await orchestrator.initialize()
        if success:
            print("✅ Doctrine orchestrator initialized successfully")
            # Test metrics collection
            metrics = await orchestrator.collect_all_metrics()
            print(f"[MONITOR] Collected {len(metrics)} metrics from all departments")
            print("Sample metrics:")
            for key, value in list(metrics.items())[:10]:
                print(f"  {key}: {value}")
            
            # Run compliance check
            print("\n🔍 Running compliance check...")
            compliance_result = await orchestrator.run_compliance_check()
            print(f"📈 Compliance Score: {compliance_result['compliance_score']}%")
            print(f"🛡️  AZ Prime State: {compliance_result['az_prime_state']}")
            print(f"⚠️  Violations: {compliance_result['violations']}")
        else:
            print("[CROSS] Doctrine orchestrator initialization failed")
    except Exception as e:
        print(f"[CROSS] Error testing doctrine orchestrator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_doctrine())