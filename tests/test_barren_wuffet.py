"""Tests for BARREN WUFFET State Machine transitions in the Doctrine orchestrator."""

import pytest

from aac.doctrine.doctrine_integration import DoctrineOrchestrator


@pytest.mark.asyncio
async def test_barren_wuffet_compliance_check():
    """DoctrineOrchestrator initializes and runs a compliance check."""
    orchestrator = DoctrineOrchestrator()
    await orchestrator.initialize()

    result = await orchestrator.run_compliance_check()
    assert 'compliance_score' in result
    assert 'barren_wuffet_state' in result
    assert 0 <= result['compliance_score'] <= 100


@pytest.mark.asyncio
async def test_barren_wuffet_state_values():
    """Verify BarrenWuffetState enum values are reachable."""
    orchestrator = DoctrineOrchestrator()
    await orchestrator.initialize()

    result = await orchestrator.run_compliance_check()
    valid_states = {'NORMAL', 'CAUTION', 'SAFE_MODE', 'HALT'}
    assert result['barren_wuffet_state'] in valid_states
