"""Tests for shared/production_safeguards.py — circuit breaker + rate limiter."""

import asyncio
import time

import pytest

from shared.production_safeguards import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenException,
    CircuitBreakerState,
    ExchangeSafeguards,
    RateLimitConfig,
    RateLimiter,
)


@pytest.fixture
def cb_config():
    """Cb config."""
    return CircuitBreakerConfig(failure_threshold=3, recovery_timeout=0.5)


@pytest.fixture
def cb(cb_config):
    """Cb."""
    return CircuitBreaker(config=cb_config)


@pytest.fixture
def rate_config():
    """Rate config."""
    return RateLimitConfig(requests_per_minute=60, burst_size=5, window_size=60.0)


@pytest.fixture
def rl(rate_config):
    """Rl."""
    return RateLimiter(config=rate_config)


# ── Circuit Breaker ────────────────────────────────────────────────────


class TestCircuitBreaker:
    """TestCircuitBreaker class."""

    @pytest.mark.asyncio
    async def test_starts_closed(self, cb):
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_success_stays_closed(self, cb):
        result = await cb.call(self._success)
        assert result == "ok"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self, cb):
        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(self._fail)
        assert cb.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_open_rejects_calls(self, cb):
        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(self._fail)
        with pytest.raises(CircuitBreakerOpenException):
            await cb.call(self._success)

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self, cb):
        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(self._fail)
        # Wait for recovery timeout
        await asyncio.sleep(0.6)
        result = await cb.call(self._success)
        assert result == "ok"
        assert cb.state == CircuitBreakerState.CLOSED

    @staticmethod
    async def _success():
        return "ok"

    @staticmethod
    async def _fail():
        raise ValueError("boom")


# ── Rate Limiter ───────────────────────────────────────────────────────


class TestRateLimiter:
    """TestRateLimiter class."""

    @pytest.mark.asyncio
    async def test_allows_burst(self, rl):
        results = [await rl.acquire() for _ in range(5)]
        assert all(results)

    @pytest.mark.asyncio
    async def test_rejects_after_burst(self, rl):
        for _ in range(5):
            await rl.acquire()
        assert await rl.acquire() is False

    @pytest.mark.asyncio
    async def test_tokens_refill(self, rl):
        for _ in range(5):
            await rl.acquire()
        # Need enough time for at least 1 token to refill (1 token/sec at 60 rpm)
        await asyncio.sleep(1.2)
        assert await rl.acquire() is True


# ── Exchange Safeguards ────────────────────────────────────────────────


class TestExchangeSafeguards:
    """TestExchangeSafeguards class."""
    def test_create_factory(self):
        sg = ExchangeSafeguards.create(
            "binance",
            RateLimitConfig(requests_per_minute=120),
            CircuitBreakerConfig(failure_threshold=5),
        )
        assert sg.exchange_name == "binance"
        assert sg.rate_limiter is not None
        assert sg.circuit_breaker is not None
