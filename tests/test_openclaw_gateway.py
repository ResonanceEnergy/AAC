"""
OpenClaw Gateway Bridge Tests
==============================
Verifies the AAC ↔ OpenClaw WebSocket bridge initializes correctly,
handles sessions, classifies intents, and routes messages.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestOpenClawGatewayBridge:
    """Tests for integrations/openclaw_gateway_bridge.py."""

    def test_import_and_models(self):
        """OpenClaw models (channels, intents, messages) are importable."""
        from integrations.openclaw_gateway_bridge import (
            MessageIntent,
            OpenClawChannel,
            OpenClawCronJob,
            OpenClawMessage,
            OpenClawSession,
            OpenClawSkill,
        )
        assert OpenClawChannel.WHATSAPP.value == "whatsapp"
        assert MessageIntent.STRATEGIC_COMMAND.value == "strategic_command"

    def test_channel_enum_covers_9_channels(self):
        """All documented gateway channels exist."""
        from integrations.openclaw_gateway_bridge import OpenClawChannel

        expected = {
            "whatsapp", "telegram", "discord", "slack",
            "signal", "imessage", "webchat", "msteams", "matrix",
        }
        actual = {c.value for c in OpenClawChannel}
        assert expected == actual

    def test_intent_enum_covers_10_intents(self):
        """All routing intents are defined."""
        from integrations.openclaw_gateway_bridge import MessageIntent

        expected = {
            "strategic_command", "operational_query", "market_data",
            "trading_signal", "risk_alert", "portfolio_status",
            "crypto_intel", "infrastructure", "general_chat",
            "doctrine_override",
        }
        actual = {i.value for i in MessageIntent}
        assert expected == actual

    def test_message_defaults(self):
        """OpenClawMessage has sane defaults."""
        from integrations.openclaw_gateway_bridge import (
            MessageIntent,
            OpenClawChannel,
            OpenClawMessage,
        )

        msg = OpenClawMessage(
            message_id="test-001",
            channel=OpenClawChannel.WEBCHAT,
            sender_id="user1",
            sender_name="Test User",
            content="What's my PnL?",
        )
        assert msg.intent == MessageIntent.GENERAL_CHAT
        assert msg.session_key == ""
        assert msg.is_group is False
        assert msg.attachments == []
        assert msg.metadata == {}

    def test_session_tracks_metadata(self):
        """Session dataclass stores channel and agent."""
        from integrations.openclaw_gateway_bridge import (
            OpenClawChannel,
            OpenClawSession,
        )

        sess = OpenClawSession(
            session_id="s1",
            session_key="main",
            agent_id="az_supreme",
            channel=OpenClawChannel.TELEGRAM,
        )
        assert sess.message_count == 0
        assert sess.context_tokens == 0

    def test_cronjob_defaults(self):
        """CronJob is enabled by default."""
        from integrations.openclaw_gateway_bridge import OpenClawCronJob

        job = OpenClawCronJob(
            job_id="j1",
            name="Morning Intel",
            schedule="0 8 * * *",
            message="Run morning intelligence scan",
        )
        assert job.enabled is True
        assert job.session_key == "main"
        assert job.last_run is None
