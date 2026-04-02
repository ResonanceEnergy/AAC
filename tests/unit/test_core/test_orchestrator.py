"""Unit tests for core.orchestrator — QuantumSignal and QuantumSignalAggregator."""
from datetime import datetime, timedelta

import pytest

from core.orchestrator import QuantumSignal, QuantumSignalAggregator


def _make_signal(
    symbol="BTC/USDT",
    direction="long",
    theater="theater_c",
    strength=0.8,
    confidence=0.9,
    quantum_advantage=0.1,
    cross_temporal_score=0.2,
    expires_at=None,
    signal_id="sig1",
):
    return QuantumSignal(
        signal_id=signal_id,
        source_agent="test_agent",
        theater=theater,
        signal_type="quantum",
        symbol=symbol,
        direction=direction,
        strength=strength,
        confidence=confidence,
        quantum_advantage=quantum_advantage,
        cross_temporal_score=cross_temporal_score,
        expires_at=expires_at,
    )


class TestQuantumSignal:
    def test_score_calculation(self):
        sig = _make_signal(strength=1.0, confidence=1.0, quantum_advantage=1.0, cross_temporal_score=1.0)
        assert sig.score == pytest.approx(1.0)

    def test_score_weighted(self):
        sig = _make_signal(strength=0.5, confidence=0.5, quantum_advantage=0.0, cross_temporal_score=0.0)
        # 0.5*0.4 + 0.5*0.4 + 0*0.1 + 0*0.1 = 0.4
        assert sig.score == pytest.approx(0.4)

    def test_not_expired(self):
        sig = _make_signal(expires_at=datetime.now() + timedelta(hours=1))
        assert not sig.is_expired()

    def test_expired(self):
        sig = _make_signal(expires_at=datetime.now() - timedelta(seconds=1))
        assert sig.is_expired()

    def test_no_expiry(self):
        sig = _make_signal(expires_at=None)
        assert not sig.is_expired()


class TestQuantumSignalAggregator:
    def test_empty_consensus(self):
        agg = QuantumSignalAggregator()
        result = agg.get_consensus("BTC/USDT")
        assert result["direction"] == "neutral"
        assert result["confidence"] == 0
        assert result["signal_count"] == 0

    def test_single_long_signal(self):
        agg = QuantumSignalAggregator()
        agg.add_signal(_make_signal(direction="long", confidence=0.9))
        result = agg.get_consensus("BTC/USDT")
        assert result["signal_count"] == 1
        assert result["confidence"] == pytest.approx(0.9)

    def test_multiple_signals_consensus(self):
        agg = QuantumSignalAggregator()
        agg.add_signal(_make_signal(signal_id="s1", direction="long", confidence=0.8))
        agg.add_signal(_make_signal(signal_id="s2", direction="long", confidence=0.9))
        result = agg.get_consensus("BTC/USDT")
        assert result["signal_count"] == 2
        assert result["direction"] in ("long", "neutral")

    def test_conflicting_signals(self):
        agg = QuantumSignalAggregator()
        agg.add_signal(_make_signal(signal_id="s1", direction="long", strength=0.5))
        agg.add_signal(_make_signal(signal_id="s2", direction="short", strength=0.5))
        result = agg.get_consensus("BTC/USDT")
        assert result["direction"] == "neutral"

    def test_expired_signals_excluded(self):
        agg = QuantumSignalAggregator()
        agg.add_signal(_make_signal(
            signal_id="expired",
            expires_at=datetime.now() - timedelta(seconds=1),
        ))
        result = agg.get_consensus("BTC/USDT")
        assert result["signal_count"] == 0

    def test_get_top_opportunities_empty(self):
        agg = QuantumSignalAggregator()
        assert agg.get_top_opportunities() == []

    def test_get_top_opportunities(self):
        agg = QuantumSignalAggregator()
        agg.add_signal(_make_signal(symbol="BTC/USDT", direction="long", confidence=0.9, strength=0.8))
        agg.add_signal(_make_signal(symbol="ETH/USDT", direction="short", confidence=0.7, strength=0.6, signal_id="s2"))
        opps = agg.get_top_opportunities(min_score=0.5)
        assert len(opps) >= 1
        assert opps[0]["symbol"] in ("BTC/USDT", "ETH/USDT")

    def test_correlation_matrix_updated(self):
        agg = QuantumSignalAggregator()
        agg.add_signal(_make_signal())
        key = "theater_c:BTC/USDT"
        assert key in agg.quantum_correlation_matrix
        assert agg.quantum_correlation_matrix[key]["count"] == 1
