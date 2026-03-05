"""Tests for BARREN WUFFET Telegram Bot.

Validates command routing, message parsing, NL intent detection,
memory storage, and response generation without making real API calls.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.barren_wuffet_telegram_bot import (
    BOT_USERNAME,
    COMMAND_ROUTES,
    BarrenWuffetMemory,
    BarrenWuffetTelegramBot,
    BotResponse,
    MessagePriority,
    TelegramMessage,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def bot():
    """Create a BarrenWuffetTelegramBot instance (no network)."""
    with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token-123"}):
        return BarrenWuffetTelegramBot()


@pytest.fixture
def sample_message():
    """Create a sample incoming TelegramMessage."""
    return TelegramMessage(
        chat_id=12345,
        user_id=67890,
        username="testuser",
        text="/bw-intel show me markets",
        timestamp=datetime.now(),
        message_id=1001,
    )


# ── Bot Initialization ────────────────────────────────────────────────────


class TestBotInit:
    """Validate bot construction."""

    def test_bot_creates(self, bot):
        assert bot is not None
        assert bot.username == "barrenwuffet069bot"

    def test_bot_not_running_by_default(self, bot):
        assert bot.running is False

    def test_bot_has_skill_handlers(self, bot):
        """Bot should register skill handlers for all defined skills."""
        from integrations.openclaw_barren_wuffet_skills import get_skill_count
        assert len(bot._skill_handlers) == get_skill_count()


# ── Command Routing ───────────────────────────────────────────────────────


class TestCommandRouting:
    """Validate command → skill mapping."""

    def test_command_routes_populated(self):
        assert len(COMMAND_ROUTES) >= 30

    def test_core_commands_exist(self):
        assert "/bw-intel" in COMMAND_ROUTES
        assert "/bw-signals" in COMMAND_ROUTES
        assert "/bw-dash" in COMMAND_ROUTES
        assert "/bw-risk" in COMMAND_ROUTES
        assert "/az" in COMMAND_ROUTES

    def test_trading_commands_exist(self):
        assert "/bw-arb" in COMMAND_ROUTES
        assert "/bw-options" in COMMAND_ROUTES
        assert "/bw-fx" in COMMAND_ROUTES

    def test_crypto_commands_exist(self):
        assert "/bw-btc" in COMMAND_ROUTES
        assert "/bw-eth" in COMMAND_ROUTES

    def test_all_routes_point_to_valid_skills(self):
        from integrations.openclaw_barren_wuffet_skills import BARREN_WUFFET_SKILLS
        for cmd, skill in COMMAND_ROUTES.items():
            assert skill in BARREN_WUFFET_SKILLS, (
                f"Command '{cmd}' routes to unknown skill '{skill}'"
            )


# ── Message Types ──────────────────────────────────────────────────────────


class TestMessageTypes:
    """Validate data classes."""

    def test_telegram_message_fields(self, sample_message):
        assert sample_message.chat_id == 12345
        assert sample_message.username == "testuser"
        assert sample_message.reply_to is None

    def test_bot_response_defaults(self):
        resp = BotResponse(text="Hello", chat_id=123)
        assert resp.priority == MessagePriority.INFO
        assert resp.parse_mode == "Markdown"

    def test_message_priority_values(self):
        assert MessagePriority.INFO.value == "info"
        assert MessagePriority.CRITICAL.value == "critical"
        assert MessagePriority.FLASH.value == "flash"


# ── Memory ─────────────────────────────────────────────────────────────────


class TestBarrenWuffetMemory:
    """Validate Second Brain memory system."""

    def test_memory_creates(self):
        mem = BarrenWuffetMemory()
        assert mem is not None

    def test_memory_store_and_recall(self):
        mem = BarrenWuffetMemory()
        mem.add("market intel data", category="market", source="test")
        results = mem.search("market intel")
        assert len(results) > 0
        assert "market intel" in results[0]["text"]

    def test_memory_recall_not_found(self):
        mem = BarrenWuffetMemory()
        results = mem.search("nonexistent_key_xyz_12345")
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
