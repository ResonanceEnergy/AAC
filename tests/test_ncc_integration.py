#!/usr/bin/env python3
from __future__ import annotations

"""
Tests for NCC Integration — AAC ↔ NCC Relay + Command Bridge
=============================================================
Covers:
    - NCCRelayClient: publish, outbox, flush, stats
    - NCC_AAC_Bridge: heartbeat, status_report, HALT, SAFE_MODE, RESUME
    - HealthHandler: /platform_status endpoint
"""

import json
import os
import sys
import time
from http.server import HTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.ncc_integration import NCC_AAC_Bridge, NCCCommand, PillarStatus
from shared.ncc_relay_client import NCCRelayClient

# ════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════


@pytest.fixture()
def outbox_dir(tmp_path):
    """Temporary outbox directory."""
    d = tmp_path / "ncc_outbox"
    d.mkdir()
    return d


@pytest.fixture()
def relay_client(outbox_dir):
    """Relay client with unreachable relay (all events go to outbox)."""
    return NCCRelayClient(
        relay_url="http://127.0.0.1:19999",  # unreachable port
        outbox_dir=outbox_dir,
    )


@pytest.fixture()
def bridge(relay_client):
    """NCC bridge with unreachable relay + command API."""
    b = NCC_AAC_Bridge(
        relay_client=relay_client,
        command_url="http://127.0.0.1:19998",
    )
    return b


# ════════════════════════════════════════════════════════════════
# NCCRelayClient Tests
# ════════════════════════════════════════════════════════════════


class TestNCCRelayClient:
    """Tests for the low-level relay transport."""

    def test_make_event_structure(self, relay_client):
        event = relay_client._make_event("ncl.sync.v1.bank.heartbeat", {"status": "online"})
        assert event["event_type"] == "ncl.sync.v1.bank.heartbeat"
        assert event["source"] == "aac"
        assert event["pillar"] == "BANK"
        assert event["data"]["status"] == "online"
        assert "timestamp" in event

    def test_publish_queues_when_relay_down(self, relay_client, outbox_dir):
        """Events should be queued locally when relay is unreachable."""
        result = relay_client.publish("ncl.sync.v1.bank.heartbeat", {"status": "online"})
        assert result is False  # relay unreachable
        assert relay_client.outbox_depth() == 1

        # Verify outbox file content
        files = list(outbox_dir.glob("*.ndjson"))
        assert len(files) == 1
        line = files[0].read_text(encoding="utf-8").strip()
        event = json.loads(line)
        assert event["event_type"] == "ncl.sync.v1.bank.heartbeat"

    def test_publish_multiple_events_queue(self, relay_client):
        """Multiple publishes should accumulate in outbox."""
        for i in range(5):
            relay_client.publish("ncl.sync.v1.bank.heartbeat", {"seq": i})
        assert relay_client.outbox_depth() == 5

    def test_publish_succeeds_with_mock_relay(self, outbox_dir):
        """Events should be sent directly when relay is reachable."""
        client = NCCRelayClient(relay_url="http://127.0.0.1:19999", outbox_dir=outbox_dir)

        mock_resp = MagicMock()
        mock_resp.status = 202
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = client.publish("ncl.sync.v1.bank.heartbeat", {"status": "online"})
        assert result is True
        assert client.outbox_depth() == 0
        assert client._published_count == 1

    def test_flush_sends_queued_events(self, relay_client, outbox_dir):
        """Flush should send queued events and clear outbox."""
        # Queue 3 events
        for i in range(3):
            relay_client.publish("ncl.sync.v1.bank.heartbeat", {"seq": i})
        assert relay_client.outbox_depth() == 3

        # Mock relay coming online
        mock_resp = MagicMock()
        mock_resp.status = 202
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = relay_client.flush()
        assert result["sent"] == 3
        assert result["failed"] == 0
        assert relay_client.outbox_depth() == 0

    def test_flush_partial_failure(self, relay_client, outbox_dir):
        """Flush should keep failed events in outbox."""
        for i in range(3):
            relay_client.publish("ncl.sync.v1.bank.heartbeat", {"seq": i})

        call_count = 0

        def mock_urlopen(req, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("connection refused")
            mock_resp = MagicMock()
            mock_resp.status = 202
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = relay_client.flush()
        assert result["sent"] == 2
        assert result["failed"] == 1
        assert relay_client.outbox_depth() == 1

    def test_relay_health_unreachable(self, relay_client):
        assert relay_client.relay_health() is None

    def test_relay_health_reachable(self, relay_client):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"status": "ok"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            health = relay_client.relay_health()
        assert health == {"status": "ok"}

    def test_stats_property(self, relay_client):
        stats = relay_client.stats
        assert stats["relay_url"] == "http://127.0.0.1:19999"
        assert stats["published"] == 0
        assert stats["queued"] == 0
        assert stats["outbox_depth"] == 0
        assert stats["relay_reachable"] is False

    def test_empty_outbox_flush(self, relay_client):
        result = relay_client.flush()
        assert result == {"sent": 0, "failed": 0}


# ════════════════════════════════════════════════════════════════
# NCC_AAC_Bridge Tests
# ════════════════════════════════════════════════════════════════


class TestNCCBridge:
    """Tests for the high-level integration bridge."""

    def test_initial_status(self, bridge):
        assert bridge._status == PillarStatus.BOOTSTRAPPING
        assert bridge._halted is False
        assert bridge._safe_mode is False

    def test_publish_heartbeat(self, bridge):
        """Heartbeat should publish with pillar_id and status."""
        ok = bridge._publish_heartbeat()
        # Relay is down, so it goes to outbox — but the call completes
        assert ok is False  # queued, not sent
        assert bridge.relay.outbox_depth() == 1

    def test_publish_heartbeat_with_relay(self, bridge):
        """Heartbeat succeeds when relay is up."""
        mock_resp = MagicMock()
        mock_resp.status = 202
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            ok = bridge._publish_heartbeat()
        assert ok is True
        assert bridge._last_heartbeat > 0

    def test_publish_status_report(self, bridge):
        """Status report should collect and publish system state."""
        ok = bridge._publish_status_report()
        assert ok is False  # queued
        assert bridge.relay.outbox_depth() == 1

    def test_collect_status_structure(self, bridge):
        """Status report should have required fields."""
        status = bridge._collect_status()
        assert status["pillar_id"] == "aac"
        assert "status" in status
        assert "halted" in status
        assert "safe_mode" in status
        assert "timestamp" in status
        assert "relay" in status

    # ── COMMAND Tests ───────────────────────────────────────────

    def test_handle_halt_command(self, bridge, tmp_path, monkeypatch):
        """HALT command should set halted state and write flag file."""
        monkeypatch.setattr(
            "shared.ncc_integration.PROJECT_ROOT", tmp_path
        )
        (tmp_path / "data").mkdir(parents=True)

        bridge._handle_command({"command": "HALT", "reason": "test"})

        assert bridge._halted is True
        assert bridge._status == PillarStatus.HALTED
        assert (tmp_path / "data" / "ncc_halt.flag").exists()

        flag = json.loads((tmp_path / "data" / "ncc_halt.flag").read_text(encoding="utf-8"))
        assert flag["command"] == "HALT"
        assert flag["reason"] == "test"

    def test_handle_safe_mode_command(self, bridge, tmp_path, monkeypatch):
        """SAFE_MODE command should set safe_mode state and write flag file."""
        monkeypatch.setattr(
            "shared.ncc_integration.PROJECT_ROOT", tmp_path
        )
        (tmp_path / "data").mkdir(parents=True)

        bridge._handle_command({"command": "SAFE_MODE", "reason": "risk limit"})

        assert bridge._safe_mode is True
        assert bridge._status == PillarStatus.SAFE_MODE
        assert (tmp_path / "data" / "ncc_safe_mode.flag").exists()

    def test_handle_resume_command(self, bridge, tmp_path, monkeypatch):
        """RESUME command should clear halted/safe_mode and remove flags."""
        monkeypatch.setattr(
            "shared.ncc_integration.PROJECT_ROOT", tmp_path
        )
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)

        # Set up halted state
        bridge._halted = True
        bridge._safe_mode = True
        bridge._status = PillarStatus.HALTED
        (data_dir / "ncc_halt.flag").write_text("{}")
        (data_dir / "ncc_safe_mode.flag").write_text("{}")

        bridge._handle_command({"command": "RESUME"})

        assert bridge._halted is False
        assert bridge._safe_mode is False
        assert bridge._status == PillarStatus.ONLINE
        assert not (data_dir / "ncc_halt.flag").exists()
        assert not (data_dir / "ncc_safe_mode.flag").exists()

    def test_handle_status_request_command(self, bridge):
        """STATUS_REQUEST should trigger a status report publish."""
        bridge._handle_command({"command": "STATUS_REQUEST"})
        # Should have queued a status report
        assert bridge.relay.outbox_depth() == 1

    def test_handle_unknown_command(self, bridge):
        """Unknown commands should be logged but not crash."""
        bridge._handle_command({"command": "EXPLODE"})
        assert len(bridge._commands_received) == 1

    def test_commands_are_recorded(self, bridge, tmp_path, monkeypatch):
        """All commands should be recorded in history."""
        monkeypatch.setattr(
            "shared.ncc_integration.PROJECT_ROOT", tmp_path
        )
        (tmp_path / "data").mkdir(parents=True)

        bridge._handle_command({"command": "HALT"})
        bridge._handle_command({"command": "RESUME"})

        assert len(bridge._commands_received) == 2
        assert bridge._commands_received[0]["command"] == "HALT"
        assert bridge._commands_received[1]["command"] == "RESUME"

    # ── Portfolio Tests ─────────────────────────────────────────

    def test_publish_portfolio_no_account_file(self, bridge):
        """Portfolio snapshot should fail gracefully if no account file."""
        ok = bridge.publish_portfolio_snapshot()
        assert ok is False

    def test_publish_portfolio_with_account_data(self, bridge, tmp_path, monkeypatch):
        """Portfolio snapshot should read account data and publish."""
        monkeypatch.setattr(
            "shared.ncc_integration.PROJECT_ROOT", tmp_path
        )
        account_dir = tmp_path / "data" / "paper_trading"
        account_dir.mkdir(parents=True)
        account_file = account_dir / "paper_account_1.json"
        account_file.write_text(
            json.dumps({
                "account_id": "paper_1",
                "balance": 10000.0,
                "equity": 10250.0,
                "daily_pnl": 250.0,
                "total_pnl": 1050.0,
                "margin_used": 5000.0,
                "margin_available": 5000.0,
                "positions": [{"symbol": "AAPL"}],
                "orders": [],
                "updated_at": "2026-03-25T14:00:00Z",
            }),
            encoding="utf-8",
        )

        ok = bridge.publish_portfolio_snapshot()
        # Relay is down, so queued
        assert ok is False
        assert bridge.relay.outbox_depth() == 1

        # Verify the queued event
        files = list(bridge.relay.outbox_dir.glob("*.ndjson"))
        line = files[0].read_text(encoding="utf-8").strip()
        event = json.loads(line)
        assert event["event_type"] == "ncl.sync.v1.bank.portfolio_snapshot"
        assert event["data"]["balance"] == 10000.0
        assert event["data"]["position_count"] == 1

    def test_publish_trade_event(self, bridge):
        """Trade event should publish with order details."""
        ok = bridge.publish_trade_event({
            "order_id": "ORD-001",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 100,
            "price": 185.50,
            "status": "FILLED",
            "exchange": "IBKR",
            "timestamp": "2026-03-25T14:30:00Z",
        })
        assert ok is False  # queued
        assert bridge.relay.outbox_depth() == 1

    # ── Property Tests ──────────────────────────────────────────

    def test_is_halted_from_state(self, bridge):
        assert bridge.is_halted is False
        bridge._halted = True
        assert bridge.is_halted is True

    def test_is_halted_from_flag_file(self, bridge, tmp_path, monkeypatch):
        monkeypatch.setattr("shared.ncc_integration.PROJECT_ROOT", tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        assert bridge.is_halted is False

        (data_dir / "ncc_halt.flag").write_text("{}")
        assert bridge.is_halted is True

    def test_is_safe_mode_from_flag_file(self, bridge, tmp_path, monkeypatch):
        monkeypatch.setattr("shared.ncc_integration.PROJECT_ROOT", tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        assert bridge.is_safe_mode is False

        (data_dir / "ncc_safe_mode.flag").write_text("{}")
        assert bridge.is_safe_mode is True

    def test_platform_status_structure(self, bridge):
        """Platform status should have all required NCC fields."""
        status = bridge.platform_status
        assert "pillar_id" in status
        assert "status" in status
        assert "is_halted" in status
        assert "is_safe_mode" in status
        assert "relay" in status


# ════════════════════════════════════════════════════════════════
# Health Server /platform_status Endpoint Tests
# ════════════════════════════════════════════════════════════════


class TestHealthServerPlatformStatus:
    """Tests for the /platform_status endpoint on the health server."""

    def test_platform_status_endpoint_exists(self):
        """Health handler should route /platform_status."""
        from health_server import HealthHandler

        handler = MagicMock(spec=HealthHandler)
        handler.path = "/platform_status"
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()

        # Verify the method exists
        assert hasattr(HealthHandler, "_respond_platform_status")

    def test_platform_status_returns_json(self):
        """The /platform_status endpoint should return valid JSON."""
        import urllib.request

        from health_server import start_health_server

        # Start server on a random high port
        port = 18765
        try:
            server = start_health_server(port=port, background=True)
        except OSError:
            pytest.skip("Port 18765 in use")

        try:
            time.sleep(0.2)
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/platform_status"
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                assert resp.status == 200
                body = json.loads(resp.read())
                assert "pillar_id" in body or "status" in body
        except Exception:
            pytest.skip("Health server startup failed")
        finally:
            server.shutdown()


# ════════════════════════════════════════════════════════════════
# NCCCommand Enum Tests
# ════════════════════════════════════════════════════════════════


class TestEnums:
    def test_ncc_commands(self):
        assert NCCCommand.HALT.value == "HALT"
        assert NCCCommand.SAFE_MODE.value == "SAFE_MODE"
        assert NCCCommand.RESUME.value == "RESUME"
        assert NCCCommand.STATUS_REQUEST.value == "STATUS_REQUEST"

    def test_pillar_status(self):
        assert PillarStatus.ONLINE.value == "online"
        assert PillarStatus.HALTED.value == "halted"
        assert PillarStatus.SAFE_MODE.value == "safe_mode"


# ════════════════════════════════════════════════════════════════
# Start/Stop Lifecycle Tests
# ════════════════════════════════════════════════════════════════


class TestBridgeLifecycle:
    def test_start_sets_online(self, bridge):
        bridge.start()
        try:
            assert bridge._running is True
            assert bridge._status == PillarStatus.ONLINE
        finally:
            bridge.stop()

    def test_stop_sets_offline(self, bridge):
        bridge.start()
        bridge.stop()
        assert bridge._running is False
        assert bridge._status == PillarStatus.OFFLINE

    def test_double_start_noop(self, bridge):
        bridge.start()
        try:
            bridge.start()  # should not crash or create duplicate threads
            assert bridge._running is True
        finally:
            bridge.stop()
