"""
AAC Exception Hierarchy
========================

Centralised exception classes for the Accelerated Arbitrage Corp platform.
All domain-specific errors inherit from ``AACError`` so callers can do a
broad ``except AACError`` when desired.
"""

from __future__ import annotations


# ── Base ──────────────────────────────────────────────────────────────────────


class AACError(Exception):
    """Root exception for all AAC domain errors."""


# ── Configuration ─────────────────────────────────────────────────────────────


class ConfigurationError(AACError):
    """Raised when a required configuration value is missing or invalid."""


class EnvironmentError(ConfigurationError):
    """Raised when a required environment variable is absent."""


# ── Trading ───────────────────────────────────────────────────────────────────


class TradingError(AACError):
    """Base class for all trading-related errors."""


class OrderError(TradingError):
    """Raised when an order cannot be created, submitted, or modified."""


class InsufficientFundsError(TradingError):
    """Raised when account balance is too low for the requested operation."""


class PositionError(TradingError):
    """Raised when a position operation fails (open, close, update)."""


class RiskLimitExceededError(TradingError):
    """Raised when a proposed trade would violate risk management limits."""


# ── Exchange / Connectivity ──────────────────────────────────────────────────


class ExchangeError(AACError):
    """Raised when communication with an exchange fails."""


class ExchangeTimeoutError(ExchangeError):
    """Raised when an exchange request times out."""


class AuthenticationError(ExchangeError):
    """Raised when exchange API authentication fails."""


class RateLimitError(ExchangeError):
    """Raised when an exchange rate-limit is hit."""


# ── Market Data ──────────────────────────────────────────────────────────────


class MarketDataError(AACError):
    """Raised when market data retrieval fails."""


class StaleDataError(MarketDataError):
    """Raised when market data is too old to be trustworthy."""


class DataProviderError(MarketDataError):
    """Raised when a specific data provider returns an error."""


# ── Strategy ─────────────────────────────────────────────────────────────────


class StrategyError(AACError):
    """Base class for strategy-related errors."""


class BacktestError(StrategyError):
    """Raised when a backtest run encounters an unrecoverable issue."""


class SignalError(StrategyError):
    """Raised when signal generation or processing fails."""


# ── Database / Accounting ────────────────────────────────────────────────────


class DatabaseError(AACError):
    """Raised when a database operation fails."""


class AccountingError(AACError):
    """Raised when an accounting calculation or reconciliation fails."""


# ── Security ─────────────────────────────────────────────────────────────────


class SecurityError(AACError):
    """Raised on security violations (bad credentials, tampered data, etc.)."""


class AuditError(SecurityError):
    """Raised when an audit-log write fails or integrity is compromised."""


# ── Agent / Orchestration ───────────────────────────────────────────────────


class AgentError(AACError):
    """Raised when an agent task fails."""


class OrchestrationError(AACError):
    """Raised when the orchestrator encounters a coordination failure."""
