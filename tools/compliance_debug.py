import asyncio
from aac.doctrine.doctrine_integration import DoctrineOrchestrator

async def main():
    orch = DoctrineOrchestrator()
    result = await orch.run_compliance_check()
    print("\n=== WARNING METRICS ===")
    for m in result.get('details', []):
        if m.get('status') == 'warning':
            print(f"{m['metric']}: value={m.get('value')} | message={m.get('message')}")

if __name__ == "__main__":
    asyncio.run(main())
