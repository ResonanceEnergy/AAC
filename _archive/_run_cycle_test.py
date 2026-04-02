"""Quick test: run one full OODA cycle."""
import asyncio
import os
import sys

# Force UTF-8 on Windows console
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from full_activation import FullActivationEngine


async def main():
    engine = FullActivationEngine()
    await engine.initialize()
    result = await engine.run_cycle()

    print("\n\n=== CYCLE RESULT ===")
    snapshot = result.get("snapshot", {})
    crypto = snapshot.get("crypto", {})
    print(f"Crypto assets fetched: {len(crypto)}")
    for ticker, data in sorted(crypto.items()):
        print(f"  {ticker}: ${data['price']:,.2f}  ({data['change_24h']:+.1f}%)")
    print(f"Signals generated: {len(result.get('signals', []))}")
    print(f"Approved: {len(result.get('approved', []))}")
    print(f"Executions: {len(result.get('executions', []))}")
    doctrine = result.get("doctrine", {})
    print(f"Doctrine state: {doctrine.get('state', 'N/A')}")
    print(f"NCL level: {doctrine.get('ncl_level', 'N/A')} ({doctrine.get('ncl_score', 0):.0f}%)")
    print(f"NCC online: {result.get('agent_findings_count', 0)} agent findings")
    print(f"BW state: {doctrine.get('state', 'N/A')}")

    # Graceful cleanup
    await engine.shutdown()
    print("\nSTATUS: OPERATIONAL")

if __name__ == "__main__":
    asyncio.run(main())
