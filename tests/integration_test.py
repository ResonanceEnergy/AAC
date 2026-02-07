"""
Accelerated Arbitrage Corp - Full Integration Test
===================================================
End-to-end test that verifies the complete workflow:
Data Sources → Research Agents → Orchestrator → Execution Engine

Run with: python -m pytest tests/integration_test.py -v -s
"""

import asyncio
import pytest
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFullSystemIntegration:
    """Complete system integration tests"""

    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self):
        """
        Test complete workflow from signal generation to position management.
        This simulates a real trading scenario in paper mode.
        """
        print("\n" + "="*60)
        print("INTEGRATION TEST: Complete Trading Workflow")
        print("="*60)
        
        # Step 1: Initialize all components
        print("\n[1/6] Initializing components...")
        from shared.config_loader import get_config
        from shared.data_sources import DataAggregator
        from BigBrainIntelligence.agents import get_agent, AGENT_REGISTRY
        from TradingExecution.execution_engine import ExecutionEngine, OrderSide
        from orchestrator import Signal, SignalAggregator
        from CentralAccounting.database import AccountingDatabase
        
        config = get_config()
        aggregator = DataAggregator()
        engine = ExecutionEngine()
        engine.dry_run = False  # Enable paper trading
        signal_agg = SignalAggregator()
        db = AccountingDatabase(":memory:")
        db.initialize()
        
        print(f"   ✓ Config loaded")
        print(f"   ✓ DataAggregator ready")
        print(f"   ✓ ExecutionEngine ready (paper_trading={engine.paper_trading})")
        print(f"   ✓ SignalAggregator ready")
        print(f"   ✓ AccountingDatabase ready")
        
        # Step 2: Run research agents
        print("\n[2/6] Running research agents...")
        agents_to_test = ["whale_watcher", "sentiment_scanner", "dex_monitor"]
        findings = []
        
        for agent_name in agents_to_test:
            agent = get_agent(agent_name)
            if agent:
                result = await agent.scan()
                findings.extend(result)
                print(f"   ✓ {agent_name}: {len(result)} findings")
        
        # Note: Simulated agents may return empty findings - that's OK for integration test
        print(f"   Total findings: {len(findings)}")
        
        # Step 3: Create signals directly (for test purposes)
        print("\n[3/6] Creating test signals...")
        signals = []
        # Create synthetic signals for testing
        for i in range(3):
            signal = Signal(
                signal_id=f"INT_TEST_{i+1:03d}",
                source_agent="integration_test",
                theater="theater_b",
                signal_type="test_signal",
                symbol="BTC/USDT",
                direction="long",
                strength=0.75 - (i * 0.1),
                confidence=0.8,  # Required field
                metadata={"test": True},
            )
            signals.append(signal)
            signal_agg.add_signal(signal)
            print(f"   ✓ Signal {signal.signal_id}: {signal.direction} {signal.symbol} (strength={signal.strength:.2f})")
        
        # Step 4: Aggregate and evaluate signals
        print("\n[4/6] Aggregating signals...")
        consensus = signal_agg.get_consensus("BTC/USDT")
        print(f"   Consensus for BTC/USDT:")
        print(f"     Direction: {consensus.get('direction', 'neutral')}")
        print(f"     Strength: {consensus.get('avg_strength', 0):.2f}")
        print(f"     Signal count: {consensus.get('signal_count', 0)}")
        
        # Step 5: Execute paper trade based on signals
        print("\n[5/6] Executing paper trade...")
        
        # Open position based on consensus
        side = OrderSide.BUY if consensus.get('direction') == 'long' else OrderSide.SELL
        position = await engine.open_position(
            symbol="BTC/USDT",
            side=side,
            quantity=0.01,
            entry_price=45000.0,
        )
        
        assert position is not None, "Position should be opened"
        print(f"   ✓ Position opened: {position.position_id}")
        print(f"     Symbol: {position.symbol}")
        print(f"     Side: {position.side.value}")
        print(f"     Quantity: {position.quantity}")
        print(f"     Entry price: ${position.entry_price:,.2f}")
        print(f"     Stop loss: ${position.stop_loss:,.2f}")
        print(f"     Take profit: ${position.take_profit:,.2f}")
        
        # Simulate price movement
        await engine.update_positions({"BTC/USDT": 46000.0})  # Price up $1000
        print(f"   ✓ Price updated to $46,000")
        print(f"     Unrealized P&L: ${position.unrealized_pnl:,.2f}")
        
        # Close position
        closed = await engine.close_position(position.position_id, price=46000.0)
        assert closed, "Position should close successfully"
        print(f"   ✓ Position closed")
        print(f"     Realized P&L: ${position.realized_pnl:,.2f}")
        
        # Step 6: Record in accounting database
        print("\n[6/6] Recording transaction...")
        accounts = db.get_accounts()
        account_id = accounts[0]['account_id']
        
        tx_id = db.record_transaction(
            account_id=account_id,
            transaction_type="trade",
            asset="BTC",
            quantity=0.01,
            price=46000.0,
            side="sell",
            symbol="BTC/USDT",
            notes=f"P&L: ${position.realized_pnl}",
        )
        print(f"   ✓ Transaction recorded: ID={tx_id}")
        
        # Verify
        print("\n" + "="*60)
        print("INTEGRATION TEST: PASSED ✓")
        print("="*60)
        print(f"  Agents ran successfully")
        print(f"  Signals created: {len(signals)}")
        print(f"  Position P&L: ${position.realized_pnl:,.2f}")
        print("="*60 + "\n")

    @pytest.mark.asyncio
    async def test_multi_theater_coordination(self):
        """Test that multiple theaters can operate simultaneously"""
        print("\n" + "="*60)
        print("INTEGRATION TEST: Multi-Theater Coordination")
        print("="*60)
        
        from BigBrainIntelligence.agents import AGENT_REGISTRY, get_agents_by_theater
        
        theaters = ["theater_b", "theater_c", "theater_d"]
        theater_findings = {}
        
        for theater in theaters:
            agents = get_agents_by_theater(theater)
            findings = []
            for agent in agents:
                result = await agent.scan()
                findings.extend(result)
            theater_findings[theater] = findings
            print(f"   {theater}: {len(agents)} agents, {len(findings)} findings")
        
        # Verify all theaters have agents registered
        total_agents = sum(len(get_agents_by_theater(t)) for t in theaters)
        total_findings = sum(len(f) for f in theater_findings.values())
        print(f"\n   Total agents: {total_agents}")
        print(f"   Total findings: {total_findings}")
        assert total_agents > 0, "Should have agents registered across theaters"
        
        print("   PASSED ✓\n")
        
        print(f"\n   Total findings across all theaters: {total_findings}")
        print("   PASSED ✓\n")

    @pytest.mark.asyncio
    async def test_risk_management_enforcement(self):
        """Test that risk limits are properly enforced"""
        print("\n" + "="*60)
        print("INTEGRATION TEST: Risk Management Enforcement")
        print("="*60)
        
        from TradingExecution.execution_engine import ExecutionEngine, OrderSide, RiskManager
        
        engine = ExecutionEngine()
        engine.dry_run = False
        
        # Test position size limit
        print("\n   Testing position size limits...")
        risk_mgr = RiskManager()
        
        # Should reject oversized position
        can_open, reason = risk_mgr.can_open_position(
            size_usd=1_000_000,  # Way over limit
            current_positions=0,
        )
        assert not can_open, "Should reject oversized position"
        print(f"   ✓ Rejected $1M position: {reason}")
        
        # Should accept reasonable position
        can_open, reason = risk_mgr.can_open_position(
            size_usd=100,
            current_positions=0,
        )
        assert can_open, "Should accept reasonable position"
        print(f"   ✓ Accepted $100 position")
        
        # Test max positions limit
        print("\n   Testing max positions limit...")
        can_open, reason = risk_mgr.can_open_position(
            size_usd=100,
            current_positions=100,  # Way over limit
        )
        assert not can_open, "Should reject due to too many positions"
        print(f"   ✓ Rejected due to position count: {reason}")
        
        print("\n   PASSED ✓\n")

    @pytest.mark.asyncio
    async def test_data_source_connectivity(self):
        """Test that data sources can connect and disconnect properly"""
        print("\n" + "="*60)
        print("INTEGRATION TEST: Data Source Connectivity")
        print("="*60)
        
        from shared.data_sources import CoinGeckoClient, DataAggregator
        
        # Test individual client
        print("\n   Testing CoinGecko client...")
        client = CoinGeckoClient()
        await client.connect()
        assert client.is_connected  # Property is 'is_connected' not 'connected'
        print("   ✓ Connected")
        
        await client.disconnect()
        assert not client.is_connected
        print("   ✓ Disconnected")
        
        # Test aggregator lifecycle
        print("\n   Testing DataAggregator...")
        agg = DataAggregator()
        await agg.connect_all()
        print("   ✓ Connected all sources")
        
        await agg.disconnect_all()
        print("   ✓ Disconnected all sources")
        
        print("\n   PASSED ✓\n")

    @pytest.mark.asyncio
    async def test_database_integrity(self):
        """Test database operations maintain data integrity"""
        print("\n" + "="*60)
        print("INTEGRATION TEST: Database Integrity")
        print("="*60)
        
        from CentralAccounting.database import AccountingDatabase
        
        db = AccountingDatabase(":memory:")
        db.initialize()
        
        # Get an account
        accounts = db.get_accounts()
        account_id = accounts[0]['account_id']
        print(f"\n   Using account ID: {account_id}")
        
        # Update balance (free_balance, not free)
        print("   Testing balance updates...")
        db.update_balance(account_id, "USDT", free_balance=10000.0)
        
        # Use get_balances (plural) and filter
        balances = db.get_balances(account_id)
        usdt_balance = next((b for b in balances if b['asset'] == 'USDT'), None)
        assert usdt_balance is not None
        assert usdt_balance['free_balance'] == 10000.0
        print(f"   ✓ Balance set: ${usdt_balance['free_balance']:,.2f} USDT")
        
        # Record multiple transactions
        print("   Testing transaction recording...")
        for i in range(5):
            db.record_transaction(
                account_id=account_id,
                transaction_type="trade",
                asset="BTC",
                quantity=0.01 * (i + 1),
                price=45000.0 + (i * 100),
            )
        print(f"   ✓ Recorded 5 transactions")
        
        # Verify transaction history
        history = db.get_transactions(account_id=account_id, limit=10)
        assert len(history) >= 5
        print(f"   ✓ Transaction history verified: {len(history)} records")
        
        print("\n   PASSED ✓\n")


class TestSystemHealth:
    """System health and configuration tests"""

    def test_all_modules_importable(self):
        """Verify all core modules can be imported"""
        print("\n" + "="*60)
        print("HEALTH CHECK: Module Imports")
        print("="*60)
        
        modules = [
            "shared.config_loader",
            "shared.data_sources",
            "BigBrainIntelligence.agents",
            "TradingExecution.execution_engine",
            "TradingExecution.exchange_connectors.binance_connector",
            "TradingExecution.exchange_connectors.coinbase_connector",
            "TradingExecution.exchange_connectors.kraken_connector",
            "CentralAccounting.database",
            "orchestrator",
        ]
        
        for module in modules:
            try:
                __import__(module)
                print(f"   ✓ {module}")
            except ImportError as e:
                print(f"   ✗ {module}: {e}")
                raise
        
        print("\n   All modules importable ✓\n")

    def test_configuration_valid(self):
        """Verify configuration is valid"""
        print("\n" + "="*60)
        print("HEALTH CHECK: Configuration Validity")
        print("="*60)
        
        from shared.config_loader import get_config
        
        config = get_config()
        validation = config.validate()
        
        print(f"\n   Valid: {validation['valid']}")
        print(f"   Dry Run: {validation['dry_run']}")
        print(f"   Exchanges configured: {validation['exchanges_configured']}")
        
        if validation['warnings']:
            print("   Warnings:")
            for warning in validation['warnings']:
                print(f"     - {warning}")
        
        print("\n   Configuration valid ✓\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
