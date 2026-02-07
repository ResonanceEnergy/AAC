"""
Shared Utilities
================
Common utility functions for retry logic, error handling, rate limiting, etc.
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================
# RETRY LOGIC
# ============================================

class RetryStrategy(Enum):
    """Retry strategies for failed operations"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True  # Add randomness to prevent thundering herd
    retryable_exceptions: tuple = (Exception,)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        if self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** (attempt - 1))
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * attempt
        else:  # FIXED
            delay = self.base_delay
        
        # Cap at max delay
        delay = min(delay, self.max_delay)
        
        # Add jitter (±25%)
        if self.jitter:
            delay = delay * (0.75 + random.random() * 0.5)
        
        return delay


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None,
):
    """
    Decorator for retrying failed function calls.
    
    Usage:
        @retry(max_attempts=3, base_delay=1.0)
        async def fetch_data():
            ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=strategy,
        retryable_exceptions=retryable_exceptions,
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {attempt} attempts: {e}"
                        )
                        raise
                    
                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{config.max_attempts}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {attempt} attempts: {e}"
                        )
                        raise
                    
                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{config.max_attempts}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    time.sleep(delay)
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# ============================================
# CIRCUIT BREAKER
# ============================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Usage:
        breaker = CircuitBreaker(name="binance_api")
        
        async def call_api():
            if not breaker.can_execute():
                raise CircuitOpenError("Service unavailable")
            try:
                result = await api_call()
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
    """
    name: str
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes before closing
    timeout: float = 30.0  # Seconds before trying again
    
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: Optional[datetime] = field(default=None)
    
    def can_execute(self) -> bool:
        """Check if circuit allows execution"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit {self.name} entering half-open state")
                    return True
            return False
        
        # HALF_OPEN - allow limited requests
        return True
    
    def record_success(self):
        """Record successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit {self.name} closed - service recovered")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name} reopened - service still failing")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit {self.name} opened after {self.failure_count} failures"
            )
    
    def reset(self):
        """Reset circuit to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


# Global circuit breakers registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get or create a circuit breaker by name"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
    return _circuit_breakers[name]


def with_circuit_breaker(
    breaker_name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: float = 30.0,
):
    """
    Decorator that wraps a function with circuit breaker protection.
    
    Usage:
        @with_circuit_breaker("binance_api")
        async def fetch_data():
            return await api_call()
    """
    def decorator(func: Callable) -> Callable:
        breaker = get_circuit_breaker(
            breaker_name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
        )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not breaker.can_execute():
                raise CircuitOpenError(
                    f"Circuit {breaker_name} is open - service unavailable"
                )
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not breaker.can_execute():
                raise CircuitOpenError(
                    f"Circuit {breaker_name} is open - service unavailable"
                )
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# ============================================
# RATE LIMITING
# ============================================

@dataclass
class RateLimiter:
    """
    Token bucket rate limiter.
    
    Usage:
        limiter = RateLimiter(rate=10, per=1.0)  # 10 requests per second
        
        async def make_request():
            await limiter.acquire()
            return await api_call()
    """
    rate: int  # Number of tokens
    per: float  # Time period in seconds
    
    tokens: float = field(default=None)
    last_update: float = field(default=None)
    _lock: asyncio.Lock = field(default=None)
    
    def __post_init__(self):
        if self.tokens is None:
            self.tokens = float(self.rate)
        if self.last_update is None:
            self.last_update = time.monotonic()
        if self._lock is None:
            self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, waiting if necessary"""
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now
                
                # Add tokens based on elapsed time
                self.tokens = min(
                    self.rate,
                    self.tokens + elapsed * (self.rate / self.per)
                )
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                
                # Calculate wait time
                needed = tokens - self.tokens
                wait_time = needed * (self.per / self.rate)
                await asyncio.sleep(wait_time)


# Global rate limiters registry
_rate_limiters: Dict[str, RateLimiter] = {}


def get_rate_limiter(name: str, rate: int, per: float = 1.0) -> RateLimiter:
    """Get or create a rate limiter by name"""
    if name not in _rate_limiters:
        _rate_limiters[name] = RateLimiter(rate=rate, per=per)
    return _rate_limiters[name]


# ============================================
# VALIDATION UTILITIES
# ============================================

def validate_symbol(symbol: str) -> bool:
    """Validate trading symbol format"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Check for base/quote format
    if "/" in symbol:
        parts = symbol.split("/")
        if len(parts) != 2:
            return False
        base, quote = parts
        return bool(base and quote and base.isalpha() and quote.isalpha())
    
    # Single asset (e.g., "BTC")
    return symbol.isalpha()


def validate_quantity(quantity: float, min_qty: float = 0.0) -> bool:
    """Validate trade quantity"""
    return isinstance(quantity, (int, float)) and quantity > min_qty


def validate_price(price: float, min_price: float = 0.0) -> bool:
    """Validate price"""
    return isinstance(price, (int, float)) and price > min_price


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize string input"""
    if not isinstance(value, str):
        return ""
    # Remove null bytes and control characters
    cleaned = "".join(c for c in value if c.isprintable() or c in "\n\t")
    return cleaned[:max_length]


# ============================================
# ASYNC UTILITIES
# ============================================

async def gather_with_concurrency(
    tasks: List[Callable],
    max_concurrent: int = 5,
    return_exceptions: bool = True
) -> List[Any]:
    """
    Run async tasks with limited concurrency.
    
    Usage:
        results = await gather_with_concurrency(
            [fetch_price(s) for s in symbols],
            max_concurrent=10
        )
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(
        *[bounded_task(t) for t in tasks],
        return_exceptions=return_exceptions
    )


class AsyncTimeout:
    """
    Context manager for async timeouts.
    
    Usage:
        async with AsyncTimeout(5.0):
            await long_running_operation()
    """
    def __init__(self, timeout: float):
        self.timeout = timeout
        self._task: Optional[asyncio.Task] = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    @staticmethod
    async def run(coro, timeout: float):
        """Run coroutine with timeout"""
        return await asyncio.wait_for(coro, timeout=timeout)


# ============================================
# PRICE & CALCULATION UTILITIES
# ============================================

def calculate_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    side: str
) -> float:
    """Calculate profit/loss for a trade"""
    if side.lower() == "buy":
        return (exit_price - entry_price) * quantity
    else:  # sell/short
        return (entry_price - exit_price) * quantity


def calculate_pnl_percentage(
    entry_price: float,
    current_price: float,
    side: str
) -> float:
    """Calculate P&L as percentage"""
    if entry_price <= 0:
        return 0.0
    
    if side.lower() == "buy":
        return ((current_price - entry_price) / entry_price) * 100
    else:
        return ((entry_price - current_price) / entry_price) * 100


def round_to_precision(value: float, precision: int) -> float:
    """Round value to given decimal precision"""
    return round(value, precision)


def calculate_position_size(
    account_balance: float,
    risk_percentage: float,
    entry_price: float,
    stop_loss_price: float
) -> float:
    """
    Calculate position size based on risk management.
    
    Args:
        account_balance: Total account balance
        risk_percentage: Max risk per trade (e.g., 2.0 for 2%)
        entry_price: Entry price
        stop_loss_price: Stop loss price
    
    Returns:
        Position size in base currency
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0.0
    
    risk_amount = account_balance * (risk_percentage / 100)
    price_risk = abs(entry_price - stop_loss_price)
    
    if price_risk <= 0:
        return 0.0
    
    return risk_amount / price_risk


# ============================================
# LOGGING UTILITIES
# ============================================

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Set up a logger with standard formatting"""
    log = logging.getLogger(name)
    log.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if not log.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        log.addHandler(ch)
    
    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    
    return log


class TradeLogger:
    """Specialized logger for trade events"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_order(self, order_id: str, symbol: str, side: str, 
                  quantity: float, price: float, status: str):
        self.logger.info(
            f"ORDER | {order_id} | {side.upper()} {quantity:.8f} {symbol} "
            f"@ ${price:,.2f} | Status: {status}"
        )
    
    def log_position_open(self, position_id: str, symbol: str, side: str,
                          quantity: float, entry_price: float):
        self.logger.info(
            f"POSITION OPEN | {position_id} | {side.upper()} {quantity:.8f} {symbol} "
            f"@ ${entry_price:,.2f}"
        )
    
    def log_position_close(self, position_id: str, symbol: str, 
                           pnl: float, pnl_pct: float):
        emoji = "🟢" if pnl >= 0 else "🔴"
        self.logger.info(
            f"POSITION CLOSE | {position_id} | {symbol} | "
            f"{emoji} P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)"
        )
    
    def log_signal(self, signal_id: str, source: str, symbol: str,
                   direction: str, strength: float):
        self.logger.info(
            f"SIGNAL | {signal_id} | {source} | {direction.upper()} {symbol} "
            f"| Strength: {strength:.2f}"
        )
