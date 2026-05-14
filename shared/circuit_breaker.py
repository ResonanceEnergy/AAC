"""
Circuit Breaker
===============
Standard CLOSED/OPEN/HALF_OPEN circuit breaker for protecting external API calls.

Usage::

    cb = CircuitBreaker("coingecko", failure_threshold=5, recovery_timeout=60)

    async def fetch():
        return await cb.call(some_async_func, arg1, arg2)

When consecutive failures reach ``failure_threshold``, the circuit OPENS and
all calls fail fast with ``CircuitOpenError`` until ``recovery_timeout`` seconds
have passed.  After that it enters HALF_OPEN, allows one attempt, and either
resets to CLOSED (success) or re-opens (failure).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

_log = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing fast — calls rejected immediately
    HALF_OPEN = "half_open"  # Probing recovery — one attempt allowed


class FailureType(Enum):
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE = "error_rate"
    SLIPPAGE = "slippage"
    CONNECTION_LOSS = "connection_loss"
    DATA_CORRUPTION = "data_corruption"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is OPEN."""


@dataclass
class CircuitMetrics:
    """Running health metrics for a single circuit."""
    request_count: int = 0
    error_count: int = 0
    latency_p95: float = 0.0
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count


class CircuitBreaker:
    """
    CLOSED/OPEN/HALF_OPEN circuit breaker.

    Thread-safe for async usage.  Not thread-safe for concurrent sync use
    from multiple OS threads (use asyncio — single-threaded event loop).

    Parameters
    ----------
    name:
        Human-readable name for logging.
    failure_threshold:
        Consecutive failures required to trip the circuit.
    recovery_timeout:
        Seconds to wait in OPEN state before trying HALF_OPEN.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self.last_state_change = datetime.now()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute *func* with circuit breaker protection.

        Raises ``CircuitOpenError`` if the circuit is OPEN and the recovery
        timeout has not elapsed yet.
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition(CircuitState.HALF_OPEN)
            else:
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is OPEN — "
                    f"{self.metrics.consecutive_failures} consecutive failures. "
                    f"Retry after {self._seconds_until_reset():.0f}s."
                )

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as exc:
            self._record_failure(exc)
            if self.state == CircuitState.HALF_OPEN:
                self._transition(CircuitState.OPEN)
            elif self.metrics.consecutive_failures >= self.failure_threshold:
                self._trip()
            raise

    def reset(self) -> None:
        """Manually reset circuit to CLOSED state (e.g. after a deployment fix)."""
        self.metrics = CircuitMetrics()
        self._transition(CircuitState.CLOSED)
        _log.info("Circuit '%s' manually reset to CLOSED", self.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: CircuitState) -> None:
        _log.info(
            "Circuit '%s': %s → %s",
            self.name,
            self.state.value,
            new_state.value,
        )
        self.state = new_state
        self.last_state_change = datetime.now()

    def _trip(self) -> None:
        _log.warning(
            "Circuit '%s' TRIPPED after %d consecutive failures",
            self.name,
            self.metrics.consecutive_failures,
        )
        self._transition(CircuitState.OPEN)

    def _record_success(self) -> None:
        self.metrics.request_count += 1
        self.metrics.consecutive_failures = 0
        if self.state == CircuitState.HALF_OPEN:
            self._transition(CircuitState.CLOSED)

    def _record_failure(self, error: Exception) -> None:
        self.metrics.request_count += 1
        self.metrics.error_count += 1
        self.metrics.consecutive_failures += 1
        self.metrics.last_failure_time = datetime.now()

    def _should_attempt_reset(self) -> bool:
        return (datetime.now() - self.last_state_change).total_seconds() >= self.recovery_timeout

    def _seconds_until_reset(self) -> float:
        elapsed = (datetime.now() - self.last_state_change).total_seconds()
        return max(0.0, self.recovery_timeout - elapsed)
