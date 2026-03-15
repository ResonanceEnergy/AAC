"""Tests for CryptoIntelligence/mev_protection_system.py — MEV detection."""

import pytest
from CryptoIntelligence.mev_protection_system import (
    MEVAttackType,
    ProtectionLevel,
    RiskLevel,
    MEVAlert,
    SandwichDetection,
    SandwichDetector,
    TransactionProtectionPlan,
)


class TestSandwichDetector:
    def _make_txs(self, front_addr="0xbot", victim_addr="0xvictim"):
        """Create a classic sandwich attack pattern."""
        return [
            {
                "hash": "0xfront",
                "from_addr": front_addr,
                "to_addr": "0xpool",
                "token_in": "ETH",
                "token_out": "USDC",
                "amount_in": 10,
                "amount_out": 2000,
                "gas_price": 100,
                "tx_index": 0,
            },
            {
                "hash": "0xvictim",
                "from_addr": victim_addr,
                "to_addr": "0xpool",
                "token_in": "ETH",
                "token_out": "USDC",
                "amount_in": 5,
                "amount_out": 900,
                "gas_price": 50,
                "tx_index": 1,
            },
            {
                "hash": "0xback",
                "from_addr": front_addr,
                "to_addr": "0xpool",
                "token_in": "USDC",
                "token_out": "ETH",
                "amount_in": 2100,
                "amount_out": 11,
                "gas_price": 40,
                "tx_index": 2,
            },
        ]

    def test_detects_sandwich(self):
        txs = self._make_txs()
        result = SandwichDetector.detect_sandwich(txs, "0xpool", expected_price=200)
        assert result is not None
        assert result.is_sandwich is True
        assert result.victim_loss_usd >= 0

    def test_no_sandwich_different_attackers(self):
        txs = self._make_txs(front_addr="0xbot")
        txs[2]["from_addr"] = "0xother"
        result = SandwichDetector.detect_sandwich(txs, "0xpool", expected_price=200)
        assert result is None

    def test_no_sandwich_too_few_txs(self):
        result = SandwichDetector.detect_sandwich([{"hash": "a"}], "0xp", 1.0)
        assert result is None


class TestDataclasses:
    def test_mev_alert(self):
        alert = MEVAlert(
            attack_type=MEVAttackType.SANDWICH,
            risk_level=RiskLevel.HIGH,
            tx_hash="0xabc",
            victim_address="0xv",
            attacker_address="0xa",
            estimated_loss_usd=150.0,
            block_number=12345,
            chain="ethereum",
            details={"pair": "ETH/USDC"},
        )
        assert alert.attack_type == MEVAttackType.SANDWICH
        assert alert.estimated_loss_usd == 150.0

    def test_protection_plan(self):
        plan = TransactionProtectionPlan(
            original_tx={"to": "0xpool"},
            risk_level=RiskLevel.HIGH,
            vulnerability_score=75.0,
            recommended_protection=ProtectionLevel.FLASHBOTS,
            max_slippage=0.5,
            recommended_slippage=0.3,
            use_private_rpc=True,
            split_trades=True,
            num_splits=3,
            use_twap=True,
            twap_duration_minutes=15,
            estimated_savings_usd=80.0,
        )
        assert plan.recommended_protection == ProtectionLevel.FLASHBOTS
        assert plan.num_splits == 3


class TestEnums:
    def test_attack_types(self):
        assert MEVAttackType.SANDWICH.value == "sandwich"
        assert MEVAttackType.FRONTRUN.value == "frontrun"

    def test_protection_levels(self):
        assert ProtectionLevel.NONE.value == "none"
        assert ProtectionLevel.MAXIMUM.value == "maximum"
