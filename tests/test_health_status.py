import asyncio
from CentralAccounting.financial_analysis_engine import get_financial_analysis_engine
from CryptoIntelligence.crypto_intelligence_engine import get_crypto_intelligence_engine
from BigBrainIntelligence.agents import get_research_agent_manager

async def test_health():
    print('Testing health status methods...')

    # Test Financial Analysis Engine
    fae = await get_financial_analysis_engine()
    fae_health = await fae.get_health_status()
    print(f'Financial Engine Health: {fae_health["status"]} ({fae_health["health_score"]:.3f})')

    # Test Crypto Intelligence Engine
    cie = await get_crypto_intelligence_engine()
    cie_health = await cie.get_health_status()
    print(f'Crypto Engine Health: {cie_health["status"]} ({cie_health["overall_health_score"]:.3f})')

    # Test Research Agent Manager
    ram = await get_research_agent_manager()
    ram_health = await ram.get_all_agents_health_status()
    print(f'Research Agents: {ram_health["healthy_agents"]}/{ram_health["total_agents"]} healthy')

    print('✅ All health status methods working!')

if __name__ == "__main__":
    asyncio.run(test_health())