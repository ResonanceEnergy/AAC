from __future__ import annotations

"""shared/account_value_feed.py — Live account equity reader (Sprint 20).

Provides a single ``AccountValueFeed.get()`` call that:

  1. Returns a cached value if the TTL has not expired.
  2. Fetches ``NetLiquidation`` from IBKR via ``IBKRConnector``.
  3. Falls back to the ``ACCOUNT_VALUE_USD`` env var on any IBKR error.
  4. Falls back to ``DEFAULT_ACCOUNT_VALUE`` (50 000 USD) when both above fail.

Always returns a positive float.  Never raises.

Thread-safe — a ``threading.Lock`` guards the cache.
"""

import asyncio
import os
import threading
import time as _time
from dataclasses import dataclass
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

DEFAULT_ACCOUNT_VALUE: float = 50_000.0
_ENV_KEY: str = "ACCOUNT_VALUE_USD"

# ── internal cache entry ──────────────────────────────────────────────────


@dataclass
class _CacheEntry:
    value: float
    fetched_at: float   # monotonic seconds
    source: str         # "ibkr" | "env" | "default"


# ── public helper ─────────────────────────────────────────────────────────


def extract_net_liq(summary: dict[str, Any]) -> float:
    """Extract the best available net liquidation value from an IBKR account-summary dict.

    Preference order: ``NetLiquidation_USD`` → ``NetLiquidation_BASE`` → ``0.0``.
    Returns ``0.0`` when no positive value is found — callers treat this as
    "not available" and fall through to the env-var / default path.
    """
    for key in ("NetLiquidation_USD", "NetLiquidation_BASE"):
        raw = summary.get(key)
        if raw is None:
            continue
        try:
            fval = float(raw)
            if fval > 0:
                return fval
        except (TypeError, ValueError):
            pass
    return 0.0


# ── AccountValueFeed ──────────────────────────────────────────────────────


class AccountValueFeed:
    """Thread-safe live account value reader.

    Parameters
    ----------
    ibkr_connector:
        Pre-built ``IBKRConnector`` instance.  When *None* (default), a new
        connector is lazily created from environment config on each cache-miss.
    cache_ttl_seconds:
        How long a fetched value is considered fresh.  Default 300 s (5 min).
    paper:
        Forwarded to the lazy-created ``IBKRConnector``.
    """

    def __init__(
        self,
        ibkr_connector: Any | None = None,
        *,
        cache_ttl_seconds: int = 300,
        paper: bool = False,
    ) -> None:
        self._connector = ibkr_connector
        self._ttl = cache_ttl_seconds
        self._paper = paper
        self._cache: _CacheEntry | None = None
        self._lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────

    def get(self) -> float:
        """Return the current account equity in USD.

        Resolution order: cached → IBKR → env var → default.
        Always returns a positive float.  Never raises.
        """
        with self._lock:
            if self._cache is not None and not self._is_stale():
                return self._cache.value

            value, source = self._resolve()
            self._cache = _CacheEntry(
                value=value,
                fetched_at=_time.monotonic(),
                source=source,
            )
            _log.info("account_value_resolved", value=round(value, 2), source=source)
            return value

    def get_source(self) -> str:
        """Return the source of the last resolved value: ``'ibkr'``, ``'env'``, or ``'default'``."""
        with self._lock:
            return self._cache.source if self._cache else "default"

    def invalidate(self) -> None:
        """Clear the cache so the next :meth:`get` re-fetches from the primary source."""
        with self._lock:
            self._cache = None

    # ── internal helpers ──────────────────────────────────────────────────

    def _is_stale(self) -> bool:
        """True when the cached entry has aged past the TTL."""
        assert self._cache is not None  # noqa: S101
        return (_time.monotonic() - self._cache.fetched_at) >= self._ttl

    def _resolve(self) -> tuple[float, str]:
        """Try each source in priority order; return (value, source_label)."""
        # 1. IBKR live read
        try:
            ibkr_val = self._fetch_from_ibkr()
            if ibkr_val > 0:
                return ibkr_val, "ibkr"
            _log.debug("account_value_ibkr_returned_zero")
        except Exception as exc:
            _log.warning("account_value_ibkr_failed", error=str(exc))

        # 2. Env var
        try:
            raw = os.environ.get(_ENV_KEY, "") or ""
            env_val = float(raw) if raw else 0.0
            if env_val > 0:
                return env_val, "env"
        except (ValueError, TypeError) as exc:
            _log.warning("account_value_env_parse_failed", error=str(exc))

        # 3. Hard default
        return DEFAULT_ACCOUNT_VALUE, "default"

    def _fetch_from_ibkr(self) -> float:
        """Open an IBKR connection, read NetLiquidation, then disconnect.

        Raises on any connection or data-fetch error — callers are expected
        to catch and fall through to the next source.
        """
        connector = self._connector if self._connector is not None else self._build_connector()

        async def _go() -> float:
            await connector.connect()
            try:
                summary = await connector.get_account_summary()
            finally:
                try:
                    await connector.disconnect()
                except Exception as exc:  # noqa: BLE001
                    _log = __import__('structlog').get_logger() if '_log' not in dir() else _log
                    _log.warning('suppressed_exception', error=str(exc))
            return extract_net_liq(summary)

        return asyncio.run(_go())

    def _build_connector(self) -> Any:
        """Lazily build an ``IBKRConnector`` from env config."""
        from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector  # noqa: PLC0415

        host = os.environ.get("IBKR_HOST", "127.0.0.1")
        port = int(os.environ.get("IBKR_PORT", "7497" if self._paper else "7496"))
        account = os.environ.get("IBKR_ACCOUNT", "")
        return IBKRConnector(host=host, port=port, account=account, paper=self._paper)
