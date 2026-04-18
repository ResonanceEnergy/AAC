"""Tests for services/api_spending_limiter.py — budget enforcement."""

from datetime import datetime

import pytest

from services.api_spending_limiter import (
    DEFAULT_RATES,
    ProviderBudget,
    SpendEvent,
    SpendingLimiter,
)


@pytest.fixture
def limiter(tmp_path):
    """Limiter."""
    return SpendingLimiter(
        daily_limit=10.0,
        monthly_limit=200.0,
        log_path=tmp_path / "spend.jsonl",
    )


def _event(provider="anthropic", cost=1.0, model="claude-sonnet", skill="test"):
    return SpendEvent(
        provider=provider,
        model=model,
        skill_id=skill,
        input_tokens=100,
        output_tokens=50,
        cost_usd=cost,
    )


class TestSpendingLimiterInit:
    """TestSpendingLimiterInit class."""
    def test_default_providers(self, limiter):
        assert set(limiter.providers.keys()) == {"anthropic", "openai", "google", "xai"}

    def test_daily_limit_split(self, limiter):
        assert limiter.providers["anthropic"].daily_limit_usd == pytest.approx(5.0)
        assert limiter.providers["openai"].daily_limit_usd == pytest.approx(2.5)

    def test_global_limits(self, limiter):
        assert limiter.global_daily_limit == 10.0
        assert limiter.global_monthly_limit == 200.0


class TestCanSpend:
    """TestCanSpend class."""
    def test_allows_under_budget(self, limiter):
        assert limiter.can_spend("anthropic", 1.0) is True

    def test_rejects_over_global_daily(self, limiter):
        assert limiter.can_spend("anthropic", 11.0) is False

    def test_rejects_over_provider_daily(self, limiter):
        # anthropic daily = 5.0
        assert limiter.can_spend("anthropic", 5.1) is False

    def test_rejects_over_provider_monthly(self, limiter):
        limiter.providers["anthropic"].spent_month_usd = 99.0
        assert limiter.can_spend("anthropic", 2.0) is False

    def test_unknown_provider_passes_global(self, limiter):
        assert limiter.can_spend("unknown_provider", 1.0) is True


class TestRecordSpend:
    """TestRecordSpend class."""
    def test_records_event(self, limiter):
        limiter.record_spend(_event(cost=2.0))
        assert limiter.providers["anthropic"].spent_today_usd == pytest.approx(2.0)
        assert limiter.providers["anthropic"].requests_today == 1

    def test_accumulates(self, limiter):
        limiter.record_spend(_event(cost=1.0))
        limiter.record_spend(_event(cost=1.5))
        assert limiter.providers["anthropic"].spent_today_usd == pytest.approx(2.5)
        assert limiter.providers["anthropic"].requests_today == 2

    def test_persists_to_log(self, limiter):
        limiter.record_spend(_event(cost=0.5))
        assert limiter._log_path.exists()
        lines = limiter._log_path.read_text().strip().split("\n")
        assert len(lines) == 1


class TestEstimateCost:
    """TestEstimateCost class."""
    def test_known_model(self, limiter):
        cost = limiter.estimate_cost("anthropic", "claude-sonnet", 1000, 500)
        expected = (1000 / 1000) * 0.003 + (500 / 1000) * 0.015
        assert cost == pytest.approx(expected)

    def test_unknown_model_fallback(self, limiter):
        cost = limiter.estimate_cost("anthropic", "nonexistent", 1000, 1000)
        assert cost > 0  # conservative fallback


class TestGetStatus:
    """TestGetStatus class."""
    def test_status_structure(self, limiter):
        status = limiter.get_status()
        assert "global_daily_limit" in status
        assert "providers" in status
        assert "anthropic" in status["providers"]
        assert status["total_spent_today"] == 0.0

    def test_status_after_spend(self, limiter):
        limiter.record_spend(_event(cost=3.0))
        status = limiter.get_status()
        assert status["total_spent_today"] == pytest.approx(3.0)
        assert status["providers"]["anthropic"]["requests_today"] == 1
