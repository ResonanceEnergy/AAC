"""strategies/daily_loss_guard.py — Sprint 10.

Enforces the ``max_daily_loss_pct`` rule from ``RiskConfig``.

If the account's total P&L for today (unrealised + realised) exceeds the
configured ceiling as a fraction of account value, ``is_limit_reached()``
returns ``True`` and ``AutoTrader.run_once()`` skips all execution.

This is a *read-only* guard — it never modifies positions itself.  It reads
today's P&L from the same ``PnLTracker`` SQLite database that
``MarketScheduler.run_pnl_snapshot()`` writes to.

Design notes
------------
* Fails-**open** (returns False / "trading allowed") on any DB or import
  error — a broken guard should NOT halt all trading.
* Thread-safe: the guard state is immutable per ``is_limit_reached()`` call;
  there is no mutable shared state between calls.
* ``PnLTracker`` is created lazily on first call so tests can mock it easily.

Usage::

    guard = DailyLossGuard(max_loss_pct=0.05, account_value_usd=50_000)
    tripped, reason = guard.is_limit_reached()
    if tripped:
        print(reason)   # e.g. "daily loss -$2,600 exceeds 5.0% ceiling ($2,500)"
"""
from __future__ import annotations

import os
from datetime import date
from typing import Optional

import structlog

_log = structlog.get_logger(__name__)

# Sentinel — means "read from env / use account value passed at call time"
_UNSET = object()


class DailyLossGuard:
    """Checks whether today's P&L has breached the daily loss ceiling.

    Args:
        max_loss_pct:       Maximum allowed loss as a fraction of account value
                            (default 0.05 = 5 %).  Reads ``MAX_DAILY_LOSS_PCT``
                            env var when 0.
        account_value_usd:  Account size.  0 → reads ``ACCOUNT_VALUE_USD`` env
                            var (default 50 000).  Can be overridden per call.
        db_path:            Path to PnLTracker SQLite DB.  ``None`` → uses
                            PnLTracker's own default.
    """

    def __init__(
        self,
        max_loss_pct: float = 0.05,
        account_value_usd: float = 0.0,
        db_path: Optional[str] = None,
    ) -> None:
        if max_loss_pct <= 0:
            max_loss_pct = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05"))
        self.max_loss_pct = max_loss_pct

        if account_value_usd <= 0:
            account_value_usd = float(os.getenv("ACCOUNT_VALUE_USD", "50000"))
        self.account_value_usd = account_value_usd

        self._db_path = db_path
        self._pnl_tracker: object | None = None   # lazy-init

    # ── public API ────────────────────────────────────────────────────────────

    def is_limit_reached(
        self,
        account_value_usd: float = 0.0,
        as_of: Optional[date] = None,
    ) -> tuple[bool, str]:
        """Return ``(True, reason)`` if the daily loss ceiling has been hit.

        Args:
            account_value_usd:  Override the instance-level account value.
                                0 → use ``self.account_value_usd``.
            as_of:              Date to check (defaults to today UTC).
                                Useful for testing.

        Returns:
            ``(False, "")`` when trading is allowed.
            ``(True,  <reason string>)`` when limit is breached.
        """
        acct = account_value_usd if account_value_usd > 0 else self.account_value_usd
        if acct <= 0:
            _log.warning("daily_loss_guard_no_account_value")
            return False, ""   # fail-open: no account value → allow trading

        ceiling_usd = acct * self.max_loss_pct

        today_pnl = self._fetch_today_pnl(as_of=as_of)
        if today_pnl is None:
            # No snapshot yet today — no basis to block trading
            return False, ""

        # today_pnl is the *total* P&L figure (unrealised + realised).
        # We care about *losses*, so negative values are bad.
        if today_pnl >= 0.0:
            return False, ""

        loss_usd = abs(today_pnl)
        if loss_usd >= ceiling_usd:
            pct_str = f"{self.max_loss_pct * 100:.1f}%"
            reason = (
                f"daily loss -${loss_usd:,.0f} exceeds {pct_str} ceiling "
                f"(${ceiling_usd:,.0f}) on ${acct:,.0f} account"
            )
            _log.warning(
                "daily_loss_guard_tripped",
                loss_usd=round(loss_usd, 2),
                ceiling_usd=round(ceiling_usd, 2),
                max_loss_pct=self.max_loss_pct,
            )
            return True, reason

        return False, ""

    def today_loss_pct(
        self,
        account_value_usd: float = 0.0,
        as_of: Optional[date] = None,
    ) -> float:
        """Return today's P&L as a fraction of account value (negative = loss).

        Returns 0.0 if no snapshot exists or on any error.
        """
        acct = account_value_usd if account_value_usd > 0 else self.account_value_usd
        if acct <= 0:
            return 0.0
        pnl = self._fetch_today_pnl(as_of=as_of)
        if pnl is None:
            return 0.0
        return round(pnl / acct, 6)

    # ── internal ─────────────────────────────────────────────────────────────

    def _get_tracker(self):
        """Lazy-instantiate PnLTracker on first use."""
        if self._pnl_tracker is None:
            from CentralAccounting.pnl_tracker import PnLTracker  # noqa: PLC0415
            kwargs: dict = {}
            if self._db_path:
                kwargs["db_path"] = self._db_path
            self._pnl_tracker = PnLTracker(**kwargs)
        return self._pnl_tracker

    def _fetch_today_pnl(self, as_of: Optional[date] = None) -> Optional[float]:
        """Read today's total P&L from the PnLTracker database.

        Returns ``None`` if no snapshot exists for today or on any error.
        Failures always return ``None`` (guard fails-open).
        """
        today_str = (as_of or date.today()).isoformat()
        try:
            tracker = self._get_tracker()
            report = tracker.today_report(snapshot_date=today_str)
            pnl_row = report.get("daily_pnl")
            if not pnl_row:
                return None
            unrealised = float(pnl_row.get("total_unrealized_pnl", 0) or 0)
            realised = float(pnl_row.get("total_realized_pnl", 0) or 0)
            return unrealised + realised
        except Exception as exc:
            _log.warning("daily_loss_guard_fetch_failed", error=str(exc))
            return None
