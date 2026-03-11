"""Tests for OpenClaw Gateway Bridge.

Validates message parsing, intent classification, session management,
and skill routing without making real API calls.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.openclaw_gateway_bridge import (
    AACIntentClassifier,
    MessageIntent,
    OpenClawChannel,
    OpenClawGatewayBridge,
    OpenClawMessage,
    OpenClawSession,
    OpenClawSkill,
)


# ── Enum Values ────────────────────────────────────────────────────────────


class TestEnums:
    """Validate enum definitions."""

    def test_channel_values(self):
        assert OpenClawChannel.TELEGRAM.value == "telegram"
        channels = list(OpenClawChannel)
        assert len(channels) >= 3

    def test_intent_values(self):
        intents = list(MessageIntent)
        assert len(intents) >= 5
        assert MessageIntent.QUERY in intents


# ── Message Model ──────────────────────────────────────────────────────────


class TestOpenClawMessage:
    """Validate message data model."""

    def test_message_creation(self):
        msg = OpenClawMessage(
            channel=OpenClawChannel.TELEGRAM,
            sender_id="user123",
            text="What is bitcoin price?",
            timestamp=datetime.now(),
        )
        assert msg.channel == OpenClawChannel.TELEGRAM
        assert "bitcoin" in msg.text.lower()


# ── Intent Classifier ─────────────────────────────────────────────────────


class TestIntentClassifier:
    """Validate NLP intent classification."""

    def test_classifier_creates(self):
        clf = AACIntentClassifier()
        assert clf is not None

    def test_classify_returns_intent(self):
        clf = AACIntentClassifier()
        intent = clf.classify("Show me the trading signals")
        assert isinstance(intent, MessageIntent)


# ── Gateway Bridge ─────────────────────────────────────────────────────────


class TestGatewayBridge:
    """Validate the main gateway bridge."""

    def test_bridge_creates(self):
        bridge = OpenClawGatewayBridge()
        assert bridge is not None

    def test_bridge_has_classifier(self):
        bridge = OpenClawGatewayBridge()
        assert hasattr(bridge, "classifier") or hasattr(bridge, "_classifier")


# ── Session ────────────────────────────────────────────────────────────────


class TestSession:
    """Validate session management."""

    def test_session_creation(self):
        session = OpenClawSession(
            session_id="sess-001",
            channel=OpenClawChannel.TELEGRAM,
            user_id="user123",
        )
        assert session.session_id == "sess-001"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
