from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from shared.circuit_breaker import (
    CircuitBreaker,
    CircuitMetrics,
    CircuitOpenError,
    CircuitState,
    FailureType,
)


# ---------------------------------------------------------------------------
# Enum & dataclass
# ---------------------------------------------------------------------------


class TestCircuitStateEnum:
    def test_values(self):
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_count(self):
        assert len(list(CircuitState)) == 3


class TestFailureTypeEnum:
    def test_values(self):
        assert FailureType.LATENCY_SPIKE.value == "latency_spike"
        assert FailureType.ERROR_RATE.value == "error_rate"
        assert FailureType.SLIPPAGE.value == "slippage"
        assert FailureType.CONNECTION_LOSS.value == "connection_loss"
        assert FailureType.DATA_CORRUPTION.value == "data_corruption"

    def test_count(self):
        assert len(list(FailureType)) == 5


class TestCircuitOpenError:
    def test_is_exception(self):
        assert issubclass(CircuitOpenError, Exception)

    def test_can_raise(self):
        with pytest.raises(CircuitOpenError, match="boom"):
            raise CircuitOpenError("boom")


class TestCircuitMetrics:
    def test_defaults(self):
        m = CircuitMetrics()
        assert m.request_count == 0
        assert m.error_count == 0
        assert m.latency_p95 == 0.0
        assert m.last_failure_time is None
        assert m.consecutive_failures == 0

    def test_error_rate_zero_requests(self):
        m = CircuitMetrics()
        assert m.error_rate == 0.0

    def test_error_rate_partial(self):
        m = CircuitMetrics(request_count=10, error_count=3)
        assert m.error_rate == pytest.approx(0.3)

    def test_error_rate_all_errors(self):
        m = CircuitMetrics(request_count=4, error_count=4)
        assert m.error_rate == 1.0


# ---------------------------------------------------------------------------
# CircuitBreaker init
# ---------------------------------------------------------------------------


class TestCircuitBreakerInit:
    def test_defaults(self):
        cb = CircuitBreaker("test")
        assert cb.name == "test"
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 60
        assert cb.state is CircuitState.CLOSED
        assert isinstance(cb.metrics, CircuitMetrics)
        assert cb.metrics.request_count == 0
        assert isinstance(cb.last_state_change, datetime)

    def test_custom_thresholds(self):
        cb = CircuitBreaker("ext", failure_threshold=2, recovery_timeout=10)
        assert cb.failure_threshold == 2
        assert cb.recovery_timeout == 10


# ---------------------------------------------------------------------------
# call() — success path
# ---------------------------------------------------------------------------


class TestCallSuccess:
    @pytest.mark.asyncio
    async def test_returns_func_result(self):
        cb = CircuitBreaker("ok")

        async def fn(x, y):
            return x + y

        assert await cb.call(fn, 2, 3) == 5

    @pytest.mark.asyncio
    async def test_passes_kwargs(self):
        cb = CircuitBreaker("ok")

        async def fn(*, a, b):
            return a * b

        assert await cb.call(fn, a=4, b=5) == 20

    @pytest.mark.asyncio
    async def test_increments_request_count(self):
        cb = CircuitBreaker("ok")

        async def fn():
            return 1

        await cb.call(fn)
        await cb.call(fn)
        assert cb.metrics.request_count == 2
        assert cb.metrics.error_count == 0

    @pytest.mark.asyncio
    async def test_resets_consecutive_failures(self):
        cb = CircuitBreaker("ok")

        async def good():
            return 1

        async def bad():
            raise ValueError("nope")

        with pytest.raises(ValueError):
            await cb.call(bad)
        assert cb.metrics.consecutive_failures == 1
        await cb.call(good)
        assert cb.metrics.consecutive_failures == 0


# ---------------------------------------------------------------------------
# call() — failure path
# ---------------------------------------------------------------------------


class TestCallFailure:
    @pytest.mark.asyncio
    async def test_propagates_exception(self):
        cb = CircuitBreaker("e", failure_threshold=10)

        async def fn():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await cb.call(fn)

    @pytest.mark.asyncio
    async def test_increments_counters(self):
        cb = CircuitBreaker("e", failure_threshold=10)

        async def fn():
            raise ValueError("x")

        with pytest.raises(ValueError):
            await cb.call(fn)
        assert cb.metrics.request_count == 1
        assert cb.metrics.error_count == 1
        assert cb.metrics.consecutive_failures == 1
        assert cb.metrics.last_failure_time is not None

    @pytest.mark.asyncio
    async def test_state_stays_closed_below_threshold(self):
        cb = CircuitBreaker("e", failure_threshold=3)

        async def fn():
            raise ValueError("x")

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(fn)
        assert cb.state is CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_trips_at_threshold(self):
        cb = CircuitBreaker("e", failure_threshold=3)

        async def fn():
            raise ValueError("x")

        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(fn)
        assert cb.state is CircuitState.OPEN
        assert cb.metrics.consecutive_failures == 3

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_immediately(self):
        cb = CircuitBreaker("e", failure_threshold=1, recovery_timeout=60)

        async def bad():
            raise RuntimeError("x")

        with pytest.raises(RuntimeError):
            await cb.call(bad)
        assert cb.state is CircuitState.OPEN

        async def good():
            return 1

        with pytest.raises(CircuitOpenError, match="OPEN"):
            await cb.call(good)
        # rejection should NOT count as a request
        # (request_count was 1 from the initial call)
        assert cb.metrics.request_count == 1


# ---------------------------------------------------------------------------
# Recovery / HALF_OPEN
# ---------------------------------------------------------------------------


class TestHalfOpenRecovery:
    @pytest.mark.asyncio
    async def test_half_open_after_timeout_then_success_closes(self):
        cb = CircuitBreaker("r", failure_threshold=1, recovery_timeout=1)

        async def bad():
            raise RuntimeError("x")

        with pytest.raises(RuntimeError):
            await cb.call(bad)
        assert cb.state is CircuitState.OPEN

        # simulate timeout elapsed by rewinding last_state_change
        cb.last_state_change = datetime.now() - timedelta(seconds=2)

        async def good():
            return 42

        result = await cb.call(good)
        assert result == 42
        assert cb.state is CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self):
        cb = CircuitBreaker("r", failure_threshold=1, recovery_timeout=1)

        async def bad():
            raise RuntimeError("x")

        with pytest.raises(RuntimeError):
            await cb.call(bad)
        cb.last_state_change = datetime.now() - timedelta(seconds=2)

        with pytest.raises(RuntimeError):
            await cb.call(bad)
        assert cb.state is CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_before_timeout_keeps_rejecting(self):
        cb = CircuitBreaker("r", failure_threshold=1, recovery_timeout=300)

        async def bad():
            raise RuntimeError("x")

        with pytest.raises(RuntimeError):
            await cb.call(bad)
        assert cb.state is CircuitState.OPEN

        async def good():
            return 1

        with pytest.raises(CircuitOpenError):
            await cb.call(good)
        assert cb.state is CircuitState.OPEN


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_from_open(self):
        cb = CircuitBreaker("r", failure_threshold=1)
        cb.state = CircuitState.OPEN
        cb.metrics.consecutive_failures = 5
        cb.metrics.error_count = 5
        cb.metrics.request_count = 5

        cb.reset()
        assert cb.state is CircuitState.CLOSED
        assert cb.metrics.consecutive_failures == 0
        assert cb.metrics.error_count == 0
        assert cb.metrics.request_count == 0

    def test_reset_replaces_metrics_object(self):
        cb = CircuitBreaker("r")
        original = cb.metrics
        cb.reset()
        assert cb.metrics is not original


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    def test_transition_updates_state_and_timestamp(self):
        cb = CircuitBreaker("h")
        before = cb.last_state_change
        cb._transition(CircuitState.OPEN)
        assert cb.state is CircuitState.OPEN
        assert cb.last_state_change >= before

    def test_trip_sets_open(self):
        cb = CircuitBreaker("h", failure_threshold=2)
        cb.metrics.consecutive_failures = 2
        cb._trip()
        assert cb.state is CircuitState.OPEN

    def test_should_attempt_reset_true_when_elapsed(self):
        cb = CircuitBreaker("h", recovery_timeout=1)
        cb.last_state_change = datetime.now() - timedelta(seconds=5)
        assert cb._should_attempt_reset() is True

    def test_should_attempt_reset_false_when_recent(self):
        cb = CircuitBreaker("h", recovery_timeout=300)
        cb.last_state_change = datetime.now()
        assert cb._should_attempt_reset() is False

    def test_seconds_until_reset_clamped_to_zero(self):
        cb = CircuitBreaker("h", recovery_timeout=10)
        cb.last_state_change = datetime.now() - timedelta(seconds=100)
        assert cb._seconds_until_reset() == 0.0

    def test_seconds_until_reset_partial(self):
        cb = CircuitBreaker("h", recovery_timeout=60)
        cb.last_state_change = datetime.now() - timedelta(seconds=10)
        sec = cb._seconds_until_reset()
        assert 45.0 <= sec <= 50.0

    def test_record_success_resets_consecutive(self):
        cb = CircuitBreaker("h")
        cb.metrics.consecutive_failures = 3
        cb._record_success()
        assert cb.metrics.consecutive_failures == 0
        assert cb.metrics.request_count == 1

    def test_record_success_in_half_open_closes(self):
        cb = CircuitBreaker("h")
        cb.state = CircuitState.HALF_OPEN
        cb._record_success()
        assert cb.state is CircuitState.CLOSED

    def test_record_failure_increments(self):
        cb = CircuitBreaker("h")
        cb._record_failure(ValueError("x"))
        assert cb.metrics.request_count == 1
        assert cb.metrics.error_count == 1
        assert cb.metrics.consecutive_failures == 1
        assert isinstance(cb.metrics.last_failure_time, datetime)


# ---------------------------------------------------------------------------
# Integration sequences
# ---------------------------------------------------------------------------


class TestIntegrationSequences:
    @pytest.mark.asyncio
    async def test_full_lifecycle_close_open_halfopen_close(self):
        cb = CircuitBreaker("life", failure_threshold=2, recovery_timeout=1)

        async def bad():
            raise ValueError("x")

        async def good():
            return "ok"

        assert cb.state is CircuitState.CLOSED
        with pytest.raises(ValueError):
            await cb.call(bad)
        with pytest.raises(ValueError):
            await cb.call(bad)
        assert cb.state is CircuitState.OPEN

        # rewind for HALF_OPEN attempt
        cb.last_state_change = datetime.now() - timedelta(seconds=2)
        result = await cb.call(good)
        assert result == "ok"
        assert cb.state is CircuitState.CLOSED
        assert cb.metrics.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_alternating_success_failure_never_trips(self):
        cb = CircuitBreaker("alt", failure_threshold=3)

        async def bad():
            raise ValueError("x")

        async def good():
            return 1

        for _ in range(5):
            with pytest.raises(ValueError):
                await cb.call(bad)
            await cb.call(good)
        assert cb.state is CircuitState.CLOSED
        assert cb.metrics.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_manual_reset_after_trip_allows_new_calls(self):
        cb = CircuitBreaker("manual", failure_threshold=1, recovery_timeout=999)

        async def bad():
            raise RuntimeError("x")

        async def good():
            return 7

        with pytest.raises(RuntimeError):
            await cb.call(bad)
        with pytest.raises(CircuitOpenError):
            await cb.call(good)

        cb.reset()
        assert await cb.call(good) == 7
        assert cb.state is CircuitState.CLOSED
