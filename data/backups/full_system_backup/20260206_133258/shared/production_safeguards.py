"""
Production Safeguards Module
============================
Implements rate limiting, circuit breakers, and other production safety measures
for AAC 2100 live trading operations.
"""

import asyncio
import time
import logging
import os
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    expected_exception: tuple = (Exception,)  # Exception types to count as failures


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60
    burst_size: int = 10
    window_size: float = 60.0  # Seconds


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation"""
    config: CircuitBreakerConfig
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    next_attempt_time: float = 0.0

    def __post_init__(self):
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() < self.next_attempt_time:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker is OPEN. Next attempt at {self.next_attempt_time}"
                    )
                else:
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF_OPEN state")

            try:
                result = await func(*args, **kwargs)

                # Success - reset failure count
                if self.state == CircuitBreakerState.HALF_OPEN:
                    logger.info("Circuit breaker recovered - closing")
                    self.state = CircuitBreakerState.CLOSED

                self.failure_count = 0
                return result

            except self.config.expected_exception as e:
                self._record_failure()
                raise e

    def _record_failure(self):
        """Record a failure and potentially open the circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout
            logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} failures. "
                f"Next attempt at {self.next_attempt_time}"
            )


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass


@dataclass
class RateLimiter:
    """Token bucket rate limiter"""
    config: RateLimitConfig
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self):
        self.tokens = self.config.burst_size
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a token. Returns True if allowed, False if rate limited."""
        async with self._lock:
            now = time.time()

            # Refill tokens based on time passed
            time_passed = now - self.last_refill
            refill_amount = time_passed * (self.config.requests_per_minute / self.config.window_size)
            self.tokens = min(self.config.burst_size, self.tokens + refill_amount)
            self.last_refill = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            else:
                return False

    async def wait_for_token(self) -> None:
        """Wait until a token is available"""
        while not await self.acquire():
            await asyncio.sleep(0.1)  # Small delay before retrying


@dataclass
class ExchangeSafeguards:
    """Production safeguards for exchange interactions"""
    exchange_name: str
    rate_limiter: RateLimiter
    circuit_breaker: CircuitBreaker

    @classmethod
    def create(cls, exchange_name: str, rate_config: RateLimitConfig, circuit_config: CircuitBreakerConfig):
        """Create safeguards for an exchange"""
        return cls(
            exchange_name=exchange_name,
            rate_limiter=RateLimiter(rate_config),
            circuit_breaker=CircuitBreaker(circuit_config)
        )

    @asynccontextmanager
    async def safe_call(self):
        """Context manager for safe exchange API calls"""
        # Rate limiting
        await self.rate_limiter.wait_for_token()

        # Circuit breaker protection
        try:
            yield
        except Exception as e:
            logger.error(f"Exchange {self.exchange_name} call failed: {e}")
            raise


class ProductionSafeguards:
    """Central production safeguards manager"""

    def __init__(self):
        self.exchange_safeguards: Dict[str, ExchangeSafeguards] = {}
        self._initialized = False

    def initialize_from_config(self, config: Dict[str, Any]):
        """Initialize safeguards from configuration"""
        if self._initialized:
            return

        # Default configurations
        default_rate_config = RateLimitConfig(
            requests_per_minute=int(config.get('RATE_LIMIT_REQUESTS_PER_MINUTE', 60)),
            burst_size=int(config.get('RATE_LIMIT_BURST_SIZE', 10))
        )

        default_circuit_config = CircuitBreakerConfig(
            failure_threshold=int(config.get('CIRCUIT_BREAKER_FAILURE_THRESHOLD', 5)),
            recovery_timeout=float(config.get('CIRCUIT_BREAKER_RECOVERY_TIMEOUT', 60.0))
        )

        # Exchange-specific safeguards
        exchanges = ['binance', 'coinbase', 'kraken']
        for exchange in exchanges:
            self.exchange_safeguards[exchange] = ExchangeSafeguards.create(
                exchange, default_rate_config, default_circuit_config
            )

        self._initialized = True
        logger.info("Production safeguards initialized")

    async def get_exchange_safeguards(self, exchange_name: str) -> ExchangeSafeguards:
        """Get safeguards for a specific exchange"""
        if exchange_name not in self.exchange_safeguards:
            raise ValueError(f"No safeguards configured for exchange: {exchange_name}")
        return self.exchange_safeguards[exchange_name]

    async def execute_with_safeguards(
        self,
        exchange_name: str,
        operation: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """Execute an exchange operation with all safeguards applied"""
        safeguards = await self.get_exchange_safeguards(exchange_name)

        async with safeguards.safe_call():
            # Execute the operation through circuit breaker
            return await safeguards.circuit_breaker.call(operation, *args, **kwargs)

    def get_safeguards_status(self) -> Dict[str, Any]:
        """Get status of all safeguards"""
        status = {
            'exchanges': {},
            'overall_health': 'healthy'
        }

        for exchange_name, safeguards in self.exchange_safeguards.items():
            exchange_status = {
                'circuit_breaker_state': safeguards.circuit_breaker.state.value,
                'failure_count': safeguards.circuit_breaker.failure_count,
                'rate_limiter_tokens': safeguards.rate_limiter.tokens,
                'healthy': safeguards.circuit_breaker.state == CircuitBreakerState.CLOSED
            }

            status['exchanges'][exchange_name] = exchange_status

            if not exchange_status['healthy']:
                status['overall_health'] = 'degraded'

        return status


# Global safeguards instance
_production_safeguards = ProductionSafeguards()


def get_production_safeguards() -> ProductionSafeguards:
    """Get the global production safeguards instance"""
    return _production_safeguards


async def initialize_production_safeguards(config: Optional[Dict[str, Any]] = None):
    """Initialize production safeguards with configuration"""
    if config is None:
        # Load from environment
        config = dict(os.environ)

    _production_safeguards.initialize_from_config(config)


# Convenience functions for easy integration
async def safe_exchange_call(
    exchange_name: str,
    operation: Callable[..., Awaitable[Any]],
    *args,
    **kwargs
) -> Any:
    """Convenience function for safe exchange calls"""
    return await _production_safeguards.execute_with_safeguards(
        exchange_name, operation, *args, **kwargs
    )


def get_safeguards_health() -> Dict[str, Any]:
    """Get current health status of all safeguards"""
    return _production_safeguards.get_safeguards_status()