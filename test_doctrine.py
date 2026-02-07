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
            print(f"📊 Collected {len(metrics)} metrics from all departments")
            print("Sample metrics:")
            for key, value in list(metrics.items())[:10]:
                print(f"  {key}: {value}")
        else:
            print("❌ Doctrine orchestrator initialization failed")
    except Exception as e:
        print(f"❌ Error testing doctrine orchestrator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_doctrine())