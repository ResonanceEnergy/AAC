import asyncio
from aac.doctrine.doctrine_integration import DoctrineOrchestrator

async def test_az_prime_transitions():
    print('Testing AZ Prime State Machine transitions...')

    orchestrator = DoctrineOrchestrator()
    await orchestrator.initialize()

    # Test with metrics that should trigger CAUTION state
    caution_metrics = {
        'max_drawdown_pct': 6.0,  # Should trigger CAUTION (>5%)
        'fill_rate': 100.0,
        'time_to_fill_p95': 200.0,
        'slippage_bps': 1.0,
        'partial_fill_rate': 0.0,
        'adverse_selection_cost': 0.5,
        'market_impact_bps': 1.0,
        'liquidity_available_pct': 600.0,
    }

    print('Testing CAUTION state trigger...')
    result = await orchestrator.run_compliance_check()
    print(f'Compliance: {result["compliance_score"]}% | State: {result["az_prime_state"]}')

if __name__ == "__main__":
    asyncio.run(test_az_prime_transitions())