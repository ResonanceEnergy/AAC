"""tests/test_execution_path.py — Sprint 2.5 end-to-end execution tests.

Tests the full path: TradeSignal → SignalExecutor → OrderConfirmation
and PositionTracker → PositionSnapshot list.

All IBKR and paper simulator calls are mocked — no real network traffic.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_signal(
    ticker: str = "SPY",
    direction_name: str = "LONG_PUT",
    confidence: float = 0.75,
    entry: float = 510.0,
    size: float = 0.05,
    strategy: str = "test",
):
    """Build a TradeSignal for tests — uses the real dataclass."""
    from shared.signal import AssetClass, Direction, TradeSignal

    return TradeSignal(
        ticker=ticker,
        direction=Direction[direction_name],
        confidence=confidence,
        entry=entry,
        stop=entry * 1.03,
        target=entry * 0.90,
        size=size,
        strategy=strategy,
        regime="CRISIS",
        asset_class=AssetClass.OPTION,
    )


# ── OrderConfirmation / ConfirmationStatus import helper ─────────────────────

def _import_executor():
    from TradingExecution.signal_executor import (
        ConfirmationStatus,
        OrderConfirmation,
        SignalExecutor,
    )

    return SignalExecutor, OrderConfirmation, ConfirmationStatus


# ═══════════════════════════════════════════════════════════════════════════════
# Test: risk gate (no real connections needed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskGate:
    """Unit tests for _risk_check without any exchange connections."""

    def test_low_confidence_rejected(self):
        from TradingExecution.signal_executor import _risk_check

        sig = _make_signal(confidence=0.30)
        approved, reason = _risk_check(sig, account_value_usd=10_000)
        assert not approved
        assert "confidence" in reason

    def test_oversized_rejected(self):
        from TradingExecution.signal_executor import _risk_check

        sig = _make_signal(size=0.50)  # 50% — way above 10% limit
        approved, reason = _risk_check(sig, account_value_usd=10_000)
        assert not approved
        assert "size" in reason

    def test_flat_direction_rejected(self):
        from TradingExecution.signal_executor import _risk_check

        sig = _make_signal(direction_name="FLAT")
        approved, reason = _risk_check(sig, account_value_usd=10_000)
        assert not approved
        assert "FLAT" in reason

    def test_zero_account_rejected(self):
        from TradingExecution.signal_executor import _risk_check

        sig = _make_signal()
        approved, reason = _risk_check(sig, account_value_usd=0)
        assert not approved

    def test_valid_signal_approved(self):
        from TradingExecution.signal_executor import _risk_check

        sig = _make_signal(confidence=0.80, size=0.05)
        approved, reason = _risk_check(sig, account_value_usd=10_000)
        assert approved
        assert reason == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Test: paper execution path
# ═══════════════════════════════════════════════════════════════════════════════

class TestPaperExecution:
    """SignalExecutor in paper mode — OrderSimulator is mocked."""

    @pytest.fixture
    def paper_executor(self):
        """SignalExecutor forced into paper mode, already marked as connected."""
        from TradingExecution.signal_executor import SignalExecutor

        exe = SignalExecutor(paper=True, account_value_usd=10_000)
        exe._connected = True  # skip real connect for unit tests
        return exe

    @pytest.mark.asyncio
    async def test_paper_filled_order(self, paper_executor):
        """Simulator fills → confirmation status is FILLED."""
        from PaperTradingDivision.order_simulator import SimulatedOrder
        from TradingExecution.signal_executor import ConfirmationStatus

        from datetime import datetime as dt

        filled_order = SimulatedOrder(
            order_id="sim_abc123",
            symbol="SPY",
            side="sell",
            quantity=10.0,
            order_type="market",
            price=510.0,
            status="filled",
            filled_quantity=10.0,
            avg_fill_price=510.0,
            created_at=dt.now(),
        )

        mock_sim = AsyncMock()
        mock_sim.submit_order = AsyncMock(return_value="sim_abc123")
        mock_sim.get_order_status = AsyncMock(return_value=filled_order)
        paper_executor._paper_sim = mock_sim

        sig = _make_signal()
        conf = await paper_executor.execute(sig)

        assert conf.status == ConfirmationStatus.FILLED
        assert conf.order_id == "sim_abc123"
        assert conf.filled_quantity == 10.0
        assert conf.avg_fill_price == 510.0
        assert conf.paper is True
        assert conf.exchange == "paper"
        assert conf.filled_at is not None

    @pytest.mark.asyncio
    async def test_paper_pending_order(self, paper_executor):
        """Simulator still pending → confirmation status is SUBMITTED."""
        from PaperTradingDivision.order_simulator import SimulatedOrder
        from TradingExecution.signal_executor import ConfirmationStatus

        from datetime import datetime as dt

        pending_order = SimulatedOrder(
            order_id="sim_def456",
            symbol="QQQ",
            side="sell",
            quantity=5.0,
            order_type="market",
            price=None,
            status="pending",
            filled_quantity=0.0,
            avg_fill_price=0.0,
            created_at=dt.now(),
        )

        mock_sim = AsyncMock()
        mock_sim.submit_order = AsyncMock(return_value="sim_def456")
        mock_sim.get_order_status = AsyncMock(return_value=pending_order)
        paper_executor._paper_sim = mock_sim

        sig = _make_signal(ticker="QQQ")
        conf = await paper_executor.execute(sig)

        assert conf.status == ConfirmationStatus.SUBMITTED
        assert conf.filled_at is None

    @pytest.mark.asyncio
    async def test_paper_risk_rejected_signal(self, paper_executor):
        """Signal below confidence threshold → REJECTED before hitting simulator."""
        from TradingExecution.signal_executor import ConfirmationStatus

        sig = _make_signal(confidence=0.10)
        conf = await paper_executor.execute(sig)

        assert conf.status == ConfirmationStatus.REJECTED
        assert "confidence" in conf.rejection_reason

    @pytest.mark.asyncio
    async def test_paper_executor_not_connected(self):
        """Calling execute() before connect() returns ERROR."""
        from TradingExecution.signal_executor import ConfirmationStatus, SignalExecutor

        exe = SignalExecutor(paper=True, account_value_usd=10_000)
        # Do NOT call connect()
        sig = _make_signal()
        conf = await exe.execute(sig)

        assert conf.status == ConfirmationStatus.ERROR
        assert "not connected" in conf.rejection_reason

    @pytest.mark.asyncio
    async def test_paper_simulator_exception_returns_error(self, paper_executor):
        """If simulator raises, confirmation status is ERROR (never propagates)."""
        from TradingExecution.signal_executor import ConfirmationStatus

        mock_sim = AsyncMock()
        mock_sim.submit_order = AsyncMock(side_effect=RuntimeError("sim exploded"))
        paper_executor._paper_sim = mock_sim

        sig = _make_signal()
        conf = await paper_executor.execute(sig)

        assert conf.status == ConfirmationStatus.ERROR
        assert "sim exploded" in conf.rejection_reason


# ═══════════════════════════════════════════════════════════════════════════════
# Test: live (IBKR) execution path
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiveExecution:
    """SignalExecutor in live mode — IBKRConnector is mocked."""

    @pytest.fixture
    def live_executor(self):
        from TradingExecution.signal_executor import SignalExecutor

        exe = SignalExecutor(paper=False, account_value_usd=50_000)
        exe._connected = True
        return exe

    def _make_exchange_order(self, order_id="ibkr_001", filled_qty=0.0, fill_price=0.0, status="submitted"):
        mock = MagicMock()
        mock.order_id = order_id
        mock.filled_quantity = filled_qty
        mock.average_fill_price = fill_price
        mock.status = status
        return mock

    @pytest.mark.asyncio
    async def test_live_order_submitted(self, live_executor):
        """IBKR returns submitted ack → confirmation is SUBMITTED."""
        from TradingExecution.signal_executor import ConfirmationStatus

        mock_ibkr = AsyncMock()
        mock_ibkr.create_order = AsyncMock(
            return_value=self._make_exchange_order(order_id="ibkr_001", filled_qty=0.0)
        )
        live_executor._ibkr = mock_ibkr

        sig = _make_signal(ticker="SPY")
        conf = await live_executor.execute(sig)

        assert conf.status == ConfirmationStatus.SUBMITTED
        assert conf.order_id == "ibkr_001"
        assert conf.paper is False
        assert conf.exchange == "ibkr"
        mock_ibkr.create_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_live_order_filled_immediately(self, live_executor):
        """IBKR returns fill in same ack (market order) → FILLED."""
        from TradingExecution.signal_executor import ConfirmationStatus

        mock_ibkr = AsyncMock()
        mock_ibkr.create_order = AsyncMock(
            return_value=self._make_exchange_order(
                order_id="ibkr_002", filled_qty=5.0, fill_price=508.5
            )
        )
        live_executor._ibkr = mock_ibkr

        sig = _make_signal(ticker="SPY")
        conf = await live_executor.execute(sig)

        assert conf.status == ConfirmationStatus.FILLED
        assert conf.filled_quantity == 5.0
        assert conf.avg_fill_price == 508.5

    @pytest.mark.asyncio
    async def test_live_ibkr_raises_returns_error(self, live_executor):
        """If IBKR connector raises, status is ERROR — never propagates."""
        from TradingExecution.signal_executor import ConfirmationStatus

        mock_ibkr = AsyncMock()
        mock_ibkr.create_order = AsyncMock(side_effect=ConnectionError("TWS unavailable"))
        live_executor._ibkr = mock_ibkr

        sig = _make_signal()
        conf = await live_executor.execute(sig)

        assert conf.status == ConfirmationStatus.ERROR
        assert "TWS unavailable" in conf.rejection_reason

    @pytest.mark.asyncio
    async def test_get_order_status_filled(self, live_executor):
        """get_order_status() returns FILLED when exchange confirms fill."""
        from TradingExecution.signal_executor import ConfirmationStatus

        mock_ibkr = AsyncMock()
        mock_ibkr.get_order = AsyncMock(
            return_value=self._make_exchange_order(
                order_id="ibkr_003", filled_qty=10.0, fill_price=509.0, status="filled"
            )
        )
        live_executor._ibkr = mock_ibkr

        conf = await live_executor.get_order_status("ibkr_003", "SPY")
        assert conf is not None
        assert conf.status == ConfirmationStatus.FILLED
        assert conf.filled_quantity == 10.0

    @pytest.mark.asyncio
    async def test_batch_execute_multiple_signals(self, live_executor):
        """execute_batch returns one confirmation per signal."""
        from TradingExecution.signal_executor import ConfirmationStatus

        mock_ibkr = AsyncMock()
        mock_ibkr.create_order = AsyncMock(
            return_value=self._make_exchange_order(order_id="ibkr_batch", filled_qty=0.0)
        )
        live_executor._ibkr = mock_ibkr

        signals = [_make_signal(ticker=t) for t in ["SPY", "QQQ", "HYG"]]
        results = await live_executor.execute_batch(signals)

        assert len(results) == 3
        assert all(r.status == ConfirmationStatus.SUBMITTED for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: PositionTracker
# ═══════════════════════════════════════════════════════════════════════════════

class TestPositionTracker:
    """PositionTracker reads positions from IBKR — connector mocked."""

    def _raw_position(self, symbol: str, qty: float, upnl: float = -50.0) -> dict:
        return {
            "symbol": symbol,
            "sec_type": "OPT",
            "quantity": qty,
            "avg_cost": 1.50,
            "market_price": 1.20,
            "market_value": qty * 1.20,
            "unrealized_pnl": upnl,
            "realized_pnl": 0.0,
            "exchange": "SMART",
            "currency": "USD",
        }

    @pytest.mark.asyncio
    async def test_refresh_populates_cache(self):
        from TradingExecution.position_tracker import PositionTracker

        tracker = PositionTracker(paper=False)
        mock_ibkr = AsyncMock()
        mock_ibkr.get_positions = AsyncMock(return_value=[
            self._raw_position("SPY", -10),
            self._raw_position("QQQ", -5),
        ])
        tracker._ibkr = mock_ibkr

        positions = await tracker.refresh()

        assert len(positions) == 2
        assert tracker.get("SPY") is not None
        assert tracker.get("SPY").quantity == -10
        assert tracker.get("UNKNOWN") is None

    @pytest.mark.asyncio
    async def test_total_exposure(self):
        from TradingExecution.position_tracker import PositionTracker

        tracker = PositionTracker(paper=False)
        mock_ibkr = AsyncMock()
        mock_ibkr.get_positions = AsyncMock(return_value=[
            self._raw_position("SPY", -10),   # market_value = -12.0
            self._raw_position("QQQ", -5),    # market_value = -6.0
        ])
        tracker._ibkr = mock_ibkr
        await tracker.refresh()

        assert tracker.total_exposure() == pytest.approx(18.0)

    @pytest.mark.asyncio
    async def test_refresh_error_returns_cached(self):
        """If IBKR raises, stale cache is returned unchanged."""
        from TradingExecution.position_tracker import PositionTracker

        tracker = PositionTracker(paper=False)
        mock_ibkr = AsyncMock()
        # First call succeeds
        mock_ibkr.get_positions = AsyncMock(return_value=[self._raw_position("SPY", -10)])
        tracker._ibkr = mock_ibkr
        await tracker.refresh()

        # Second call fails
        mock_ibkr.get_positions = AsyncMock(side_effect=RuntimeError("IBKR down"))
        positions = await tracker.refresh()

        assert len(positions) == 1  # stale cache returned
        assert positions[0].symbol == "SPY"

    @pytest.mark.asyncio
    async def test_summary_structure(self):
        from TradingExecution.position_tracker import PositionTracker

        tracker = PositionTracker(paper=False)
        mock_ibkr = AsyncMock()
        mock_ibkr.get_positions = AsyncMock(return_value=[self._raw_position("IWM", -8)])
        tracker._ibkr = mock_ibkr
        await tracker.refresh()

        summary = tracker.summary()
        assert "position_count" in summary
        assert "total_exposure_usd" in summary
        assert "positions" in summary
        assert summary["position_count"] == 1

    @pytest.mark.asyncio
    async def test_no_connection_returns_empty(self):
        """Before connecting, refresh returns empty list (no crash)."""
        from TradingExecution.position_tracker import PositionTracker

        tracker = PositionTracker(paper=False)
        # _ibkr is None — never connected
        positions = await tracker.refresh()
        assert positions == []


# ═══════════════════════════════════════════════════════════════════════════════
# Test: OrderConfirmation dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrderConfirmation:
    def test_to_dict_keys(self):
        from TradingExecution.signal_executor import ConfirmationStatus, OrderConfirmation

        conf = OrderConfirmation(
            signal_ticker="SPY",
            order_id="ibkr_999",
            status=ConfirmationStatus.FILLED,
            filled_quantity=10.0,
            avg_fill_price=508.0,
            exchange="ibkr",
            paper=False,
        )
        d = conf.to_dict()
        required = {
            "signal_ticker", "order_id", "status", "filled_quantity",
            "avg_fill_price", "exchange", "paper", "submitted_at",
        }
        assert required <= d.keys()
        assert d["status"] == "filled"

    def test_submitted_at_auto_populated(self):
        from TradingExecution.signal_executor import ConfirmationStatus, OrderConfirmation

        conf = OrderConfirmation(
            signal_ticker="GLD",
            order_id="x",
            status=ConfirmationStatus.SUBMITTED,
        )
        assert conf.submitted_at  # not empty
