"""Sprint 55 — assert silent MOCK fallbacks are gone from comms + openclaw bridge.

- ``shared/communication_framework.py`` no longer self-describes as "mock"
  (it's an in-process pub/sub bus, which is the production implementation).
- ``integrations/openclaw_gateway_bridge.py``:
  * ``connect()`` raises ImportError when the ``websockets`` package is
    missing (was: silently set ``_connected=True`` and return True).
  * ``send_response()`` raises RuntimeError when not connected (was:
    silently logged "[MOCK SEND ...]").
  * ``send_proactive_message()`` raises RuntimeError when not connected
    (was: silently logged "[PROACTIVE ...]").
  * ``register_cron_job()`` raises RuntimeError when not connected (was:
    silently logged "Registered cron job ..." without publishing).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO = Path(__file__).resolve().parent.parent
_COMMS = _REPO / "shared" / "communication_framework.py"
_BRIDGE = _REPO / "integrations" / "openclaw_gateway_bridge.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-level assertions
# ---------------------------------------------------------------------------

def test_comms_no_mock_label_in_module_header():
    src = _read(_COMMS)
    assert "Mock implementation for inter-agent communication" not in src
    assert '"""Mock communication framework for inter-agent messaging"""' not in src
    assert "(mock implementation)" not in src


def test_bridge_no_silent_mock_mode_label():
    src = _read(_BRIDGE)
    # The old silent-fallback warning string must be gone.
    assert (
        "websockets not installed \u2014 OpenClaw bridge running in MOCK mode"
        not in src
    )
    # "Sprint 55" sentinel comments must be present (proves the rewrite).
    assert "Sprint 55" in src
    # The old MOCK SEND log line is gone.
    assert "[MOCK SEND" not in src or "previously logged" in src
    assert "[PROACTIVE \u2192" not in src


# ---------------------------------------------------------------------------
# Behavioral assertions for the comms framework
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_comms_send_and_retrieve_round_trip():
    from shared.communication_framework import CommunicationFramework

    cf = CommunicationFramework()
    await cf.register_agent("alice")
    await cf.register_agent("bob")
    ok = await cf.send_message("alice", "bob", "ping", {"hello": "world"})
    assert ok is True
    msgs = await cf.get_messages_for_agent("bob")
    assert len(msgs) == 1
    assert msgs[0].sender == "alice"
    assert msgs[0].message_type == "ping"
    assert msgs[0].payload == {"hello": "world"}


# ---------------------------------------------------------------------------
# Behavioral assertions for the OpenClaw bridge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bridge_connect_raises_when_websockets_missing():
    from integrations.openclaw_gateway_bridge import OpenClawGatewayBridge

    bridge = OpenClawGatewayBridge(gateway_url="ws://example.invalid")
    # Force the import inside connect() to fail.
    with patch.dict(sys.modules, {"websockets": None}):
        with pytest.raises(ImportError, match="Sprint 55"):
            await bridge.connect()
    assert bridge._connected is False


@pytest.mark.asyncio
async def test_bridge_send_response_raises_when_not_connected():
    from integrations.openclaw_gateway_bridge import (
        OpenClawChannel,
        OpenClawGatewayBridge,
        OpenClawMessage,
    )

    bridge = OpenClawGatewayBridge(gateway_url="ws://example.invalid")
    msg = OpenClawMessage(
        message_id="m1",
        channel=OpenClawChannel.WEBCHAT,
        sender_id="u1",
        sender_name="user",
        content="hi",
        session_key="s1",
        timestamp=datetime.now(),
    )
    with pytest.raises(RuntimeError, match="not connected"):
        await bridge.send_response(msg, "response text")


@pytest.mark.asyncio
async def test_bridge_send_proactive_raises_when_not_connected():
    from integrations.openclaw_gateway_bridge import (
        OpenClawChannel,
        OpenClawGatewayBridge,
    )

    bridge = OpenClawGatewayBridge(gateway_url="ws://example.invalid")
    with pytest.raises(RuntimeError, match="not connected"):
        await bridge.send_proactive_message(
            channel=OpenClawChannel.WEBCHAT,
            session_key="s1",
            content="alert",
        )


@pytest.mark.asyncio
async def test_bridge_register_cron_raises_when_not_connected():
    from integrations.openclaw_gateway_bridge import (
        OpenClawCronJob,
        OpenClawGatewayBridge,
    )

    bridge = OpenClawGatewayBridge(gateway_url="ws://example.invalid")
    job = OpenClawCronJob(
        job_id="j1",
        name="nightly",
        schedule="0 0 * * *",
        message="run",
        session_key="s1",
    )
    with pytest.raises(RuntimeError, match="not connected"):
        await bridge.register_cron_job(job)
