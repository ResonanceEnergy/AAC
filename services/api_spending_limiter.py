"""
API Spending Limiter — OpenClaw / BARREN WUFFET
================================================

Tracks and enforces daily spend caps for AI API calls to prevent
runaway costs during automated trading research.

Features:
- Per-provider spend tracking (Anthropic, OpenAI, Google, xAI)
- Per-skill budget allocation
- Daily/monthly rolling limits
- Automatic throttling when approaching cap
- Audit log for all spend events
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpendEvent:
    """Single API spend record."""
    provider: str
    model: str
    skill_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProviderBudget:
    """Budget configuration for a single provider."""
    name: str
    daily_limit_usd: float
    monthly_limit_usd: float
    spent_today_usd: float = 0.0
    spent_month_usd: float = 0.0
    requests_today: int = 0
    throttled: bool = False


# ── Cost rates per 1K tokens (approximate, updated Feb 2026) ───────────

DEFAULT_RATES: Dict[str, Dict[str, float]] = {
    "anthropic": {
        "claude-opus": {"input": 0.015, "output": 0.075},
        "claude-sonnet": {"input": 0.003, "output": 0.015},
        "claude-haiku": {"input": 0.00025, "output": 0.00125},
    },
    "openai": {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    },
    "google": {
        "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
        "gemini-2.0-pro": {"input": 0.00125, "output": 0.005},
    },
    "xai": {
        "grok-3": {"input": 0.003, "output": 0.015},
    },
}


class SpendingLimiter:
    """
    Tracks API spend and enforces budget caps.

    Usage:
        limiter = SpendingLimiter(daily_limit=5.0)

        # Before making a call
        if limiter.can_spend("anthropic", estimated_cost=0.05):
            # make the call
            limiter.record_spend(SpendEvent(...))
        else:
            logger.warning("Budget exceeded, throttling")
    """

    def __init__(
        self,
        daily_limit: float = 10.0,
        monthly_limit: float = 200.0,
        log_path: Optional[Path] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._log_path = log_path or Path("logs/api_spending.jsonl")
        self._events: List[SpendEvent] = []

        # Default per-provider budgets (split evenly, can be customized)
        env_daily = float(os.getenv("OPENCLAW_DAILY_SPEND_LIMIT", str(daily_limit)))
        self.global_daily_limit = env_daily
        self.global_monthly_limit = monthly_limit

        self.providers: Dict[str, ProviderBudget] = {
            "anthropic": ProviderBudget("anthropic", env_daily * 0.5, monthly_limit * 0.5),
            "openai": ProviderBudget("openai", env_daily * 0.25, monthly_limit * 0.25),
            "google": ProviderBudget("google", env_daily * 0.15, monthly_limit * 0.15),
            "xai": ProviderBudget("xai", env_daily * 0.10, monthly_limit * 0.10),
        }

        self._day_started = datetime.now().date()
        self._month_started = datetime.now().replace(day=1).date()

        logger.info(
            f"SpendingLimiter initialized — daily cap: ${env_daily:.2f}, "
            f"monthly cap: ${monthly_limit:.2f}"
        )

    # ── Budget Checks ──────────────────────────────────────────────────

    def can_spend(self, provider: str, estimated_cost: float = 0.0) -> bool:
        """Check if a spend is within budget."""
        self._maybe_reset_counters()

        with self._lock:
            total_today = sum(p.spent_today_usd for p in self.providers.values())

            # Global daily check
            if total_today + estimated_cost > self.global_daily_limit:
                logger.warning(
                    f"Global daily limit would be exceeded: "
                    f"${total_today:.4f} + ${estimated_cost:.4f} > ${self.global_daily_limit:.2f}"
                )
                return False

            # Provider-specific check
            prov = self.providers.get(provider)
            if prov:
                if prov.spent_today_usd + estimated_cost > prov.daily_limit_usd:
                    logger.warning(f"{provider} daily limit reached")
                    prov.throttled = True
                    return False
                if prov.spent_month_usd + estimated_cost > prov.monthly_limit_usd:
                    logger.warning(f"{provider} monthly limit reached")
                    prov.throttled = True
                    return False

            return True

    def estimate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost in USD for a given request."""
        rates = DEFAULT_RATES.get(provider, {}).get(model, {})
        if not rates:
            # Conservative fallback
            return (input_tokens + output_tokens) * 0.01 / 1000

        input_cost = (input_tokens / 1000) * rates.get("input", 0.01)
        output_cost = (output_tokens / 1000) * rates.get("output", 0.03)
        return input_cost + output_cost

    # ── Recording ──────────────────────────────────────────────────────

    def record_spend(self, event: SpendEvent) -> None:
        """Record an API spend event."""
        self._maybe_reset_counters()

        with self._lock:
            self._events.append(event)

            prov = self.providers.get(event.provider)
            if prov:
                prov.spent_today_usd += event.cost_usd
                prov.spent_month_usd += event.cost_usd
                prov.requests_today += 1

            # Persist to JSONL log
            self._append_log(event)

    def _append_log(self, event: SpendEvent) -> None:
        """Append spend event to JSONL file."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                record = {
                    "provider": event.provider,
                    "model": event.model,
                    "skill_id": event.skill_id,
                    "input_tokens": event.input_tokens,
                    "output_tokens": event.output_tokens,
                    "cost_usd": event.cost_usd,
                    "timestamp": event.timestamp.isoformat(),
                }
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write spend log: {e}")

    # ── Counter Management ─────────────────────────────────────────────

    def _maybe_reset_counters(self) -> None:
        """Reset daily/monthly counters if date has rolled over."""
        today = datetime.now().date()
        month_start = today.replace(day=1)

        with self._lock:
            if today > self._day_started:
                for prov in self.providers.values():
                    prov.spent_today_usd = 0.0
                    prov.requests_today = 0
                    prov.throttled = False
                self._day_started = today
                logger.info("Daily spend counters reset")

            if month_start > self._month_started:
                for prov in self.providers.values():
                    prov.spent_month_usd = 0.0
                self._month_started = month_start
                logger.info("Monthly spend counters reset")

    # ── Reporting ──────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get current spend status across all providers."""
        self._maybe_reset_counters()
        total_today = sum(p.spent_today_usd for p in self.providers.values())
        total_month = sum(p.spent_month_usd for p in self.providers.values())

        return {
            "global_daily_limit": self.global_daily_limit,
            "global_monthly_limit": self.global_monthly_limit,
            "total_spent_today": round(total_today, 4),
            "total_spent_month": round(total_month, 4),
            "daily_remaining": round(self.global_daily_limit - total_today, 4),
            "providers": {
                name: {
                    "spent_today": round(p.spent_today_usd, 4),
                    "daily_limit": p.daily_limit_usd,
                    "spent_month": round(p.spent_month_usd, 4),
                    "monthly_limit": p.monthly_limit_usd,
                    "requests_today": p.requests_today,
                    "throttled": p.throttled,
                }
                for name, p in self.providers.items()
            },
            "total_events": len(self._events),
        }
