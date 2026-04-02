"""
Integration tests for the OpenClaw / ClawHub subsystem.

Validates:
  - Config loading of OpenClaw fields
  - ClawHub client (offline mode)
  - Gateway bridge instantiation, routing, and metrics
  - AZ Supreme handler initialization
  - Skill registration and SKILL.md generation
  - Self-improving error recording
  - ClawHub skill search
  - Deprecated openclaw_skills.py emits warning
"""

import asyncio
import warnings
from pathlib import Path

import pytest

# ─── Config ──────────────────────────────────────────────────────────────

def test_config_has_openclaw_fields():
    """Config dataclass includes all OpenClaw/ClawHub fields."""
    from shared.config_loader import Config
    cfg = Config()
    assert hasattr(cfg, 'openclaw_skills_dir')
    assert hasattr(cfg, 'openclaw_gateway_url')
    assert hasattr(cfg, 'openclaw_gateway_token')
    assert hasattr(cfg, 'openclaw_daily_spend_limit')
    assert hasattr(cfg, 'clawhub_api_key')
    assert hasattr(cfg, 'aac_api_key')
    # Defaults
    assert cfg.openclaw_gateway_url == 'ws://127.0.0.1:18789'
    assert cfg.openclaw_daily_spend_limit == 10.0


def test_config_sensitive_fields_include_openclaw():
    """Sensitive fields list includes OpenClaw keys."""
    from shared.config_loader import Config
    cfg = Config()
    assert 'clawhub_api_key' in cfg._SENSITIVE_FIELDS
    assert 'aac_api_key' in cfg._SENSITIVE_FIELDS
    assert 'openclaw_gateway_token' in cfg._SENSITIVE_FIELDS


# ─── ClawHub Client (archived — skip if module missing) ─────────────────

_clawhub_available = True
try:
    import integrations.clawhub_client  # noqa: F401
except (ImportError, ModuleNotFoundError):
    _clawhub_available = False

_skip_clawhub = pytest.mark.skipif(not _clawhub_available, reason="clawhub_client archived")


@_skip_clawhub
def test_clawhub_client_singleton():
    """get_clawhub_client returns a singleton instance."""
    import integrations.clawhub_client as mod
    # Reset singleton
    mod._client_instance = None
    c1 = mod.get_clawhub_client()
    c2 = mod.get_clawhub_client()
    assert c1 is c2
    mod._client_instance = None  # cleanup


@_skip_clawhub
def test_clawhub_client_offline_search():
    """Offline search returns matching curated skills."""
    from integrations.clawhub_client import ClawHubClient
    client = ClawHubClient()
    results = client._offline_search("agent")
    assert len(results) > 0
    names = [r.name for r in results]
    assert "self-improving-agent" in names


@_skip_clawhub
def test_clawhub_client_offline_popular():
    """Offline popular returns skills sorted by downloads."""
    from integrations.clawhub_client import ClawHubClient
    client = ClawHubClient()
    popular = client._offline_popular()
    assert len(popular) > 0
    assert popular[0].downloads >= popular[-1].downloads


@_skip_clawhub
def test_clawhub_client_offline_get_skill():
    """Offline get_skill returns a known skill."""
    from integrations.clawhub_client import ClawHubClient
    client = ClawHubClient()
    skill = client._offline_get_skill("summarize")
    assert skill is not None
    assert skill.name == "summarize"


@_skip_clawhub
def test_clawhub_client_install_command():
    """Install command format is correct."""
    from integrations.clawhub_client import ClawHubClient
    client = ClawHubClient()
    cmd = client.get_install_command("proactive-agent")
    assert cmd == "npx clawhub@latest install proactive-agent"


@_skip_clawhub
def test_clawhub_client_status():
    """Status dict has expected keys."""
    from integrations.clawhub_client import ClawHubClient
    client = ClawHubClient(api_key="test_key")
    status = client.get_status()
    assert status["configured"] is True
    assert "curated_skills" in status


@_skip_clawhub
@pytest.mark.asyncio
async def test_clawhub_client_search_fallback():
    """search_skills falls back to offline when httpx unavailable."""
    from integrations.clawhub_client import ClawHubClient
    client = ClawHubClient()
    results = await client.search_skills("search")
    assert len(results) > 0


# ─── Gateway Bridge ─────────────────────────────────────────────────────

def test_gateway_bridge_instantiation():
    """Bridge can be instantiated with defaults."""
    from integrations.openclaw_gateway_bridge import OpenClawGatewayBridge
    bridge = OpenClawGatewayBridge()
    assert bridge.gateway_url == "ws://127.0.0.1:18789"
    assert bridge._connected is False
    assert bridge.metrics["messages_received"] == 0


def test_gateway_bridge_status():
    """Bridge status dict contains expected keys."""
    from integrations.openclaw_gateway_bridge import OpenClawGatewayBridge
    bridge = OpenClawGatewayBridge()
    status = bridge.get_status()
    assert "connected" in status
    assert "sessions_active" in status
    assert "metrics" in status


def test_intent_classifier():
    """AACIntentClassifier routes messages to correct intents."""
    from integrations.openclaw_gateway_bridge import AACIntentClassifier, MessageIntent
    assert AACIntentClassifier.classify("show me the portfolio balance") == MessageIntent.PORTFOLIO_STATUS
    assert AACIntentClassifier.classify("crypto whale alert") == MessageIntent.CRYPTO_INTEL
    assert AACIntentClassifier.classify("execute a trade") == MessageIntent.TRADING_SIGNAL
    assert AACIntentClassifier.classify("hello there") == MessageIntent.GENERAL_CHAT


def test_intent_to_agent_mapping():
    """Each intent maps to a valid agent ID."""
    from integrations.openclaw_gateway_bridge import AACIntentClassifier, MessageIntent
    for intent in MessageIntent:
        agent = AACIntentClassifier.get_target_agent(intent)
        assert isinstance(agent, str)
        assert len(agent) > 0


@pytest.mark.asyncio
async def test_gateway_bridge_connect_mock():
    """Bridge connects in mock mode when websockets is not available."""
    from integrations.openclaw_gateway_bridge import OpenClawGatewayBridge
    bridge = OpenClawGatewayBridge()
    # Force mock mode by hiding websockets
    import sys
    ws_mod = sys.modules.get('websockets')
    sys.modules['websockets'] = None
    try:
        result = await bridge.connect()
        assert result is True
        assert bridge._connected is True
    finally:
        if ws_mod is not None:
            sys.modules['websockets'] = ws_mod
        else:
            sys.modules.pop('websockets', None)
        await bridge.shutdown()


def test_webhook_auth_rejects_unsigned():
    """_handle_webhook rejects unsigned webhooks when token is set."""
    from integrations.openclaw_gateway_bridge import OpenClawGatewayBridge
    bridge = OpenClawGatewayBridge(gateway_token="secret123")
    data = {"type": "webhook.inbound", "hookId": "hook1", "payload": {}}
    # Should silently reject (no exception, but error counter increments)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(bridge._handle_webhook(data))
    loop.close()
    assert bridge.metrics["errors"] >= 1


def test_error_recording():
    """record_error writes to error-log.jsonl."""
    import tempfile

    from integrations.openclaw_gateway_bridge import OpenClawGatewayBridge
    with tempfile.TemporaryDirectory() as tmpdir:
        bridge = OpenClawGatewayBridge(workspace_dir=tmpdir)
        bridge.record_error("test_context", ValueError("test failure"))
        assert bridge.metrics["errors"] == 1
        summary = bridge.get_error_summary()
        assert len(summary) == 1
        assert summary[0]["context"] == "test_context"
        assert summary[0]["error_type"] == "ValueError"


def test_health_check_returns_list():
    """_check_system_health returns a list (possibly empty)."""
    from integrations.openclaw_gateway_bridge import OpenClawGatewayBridge
    bridge = OpenClawGatewayBridge()
    alerts = bridge._check_system_health()
    assert isinstance(alerts, list)


@pytest.mark.asyncio
async def test_clawhub_search_from_bridge():
    """search_clawhub_skills returns results via offline mode."""
    from integrations.openclaw_gateway_bridge import OpenClawGatewayBridge
    bridge = OpenClawGatewayBridge()
    results = await bridge.search_clawhub_skills("agent")
    assert isinstance(results, list)
    if results:
        assert "name" in results[0]


# ─── AZ Supreme Handler ─────────────────────────────────────────────────

def test_az_supreme_handler_creation():
    """AZ Supreme handler can be instantiated."""
    from integrations.openclaw_az_supreme_handler import AZSupremeOpenClawHandler
    handler = AZSupremeOpenClawHandler()
    assert handler.agent_id == "AZ-SUPREME"
    assert handler._initialized is False


def test_az_supreme_live_status_defaults():
    """_get_live_system_status returns valid defaults."""
    from integrations.openclaw_az_supreme_handler import AZSupremeOpenClawHandler
    handler = AZSupremeOpenClawHandler()
    status = handler._get_live_system_status()
    assert "doctrine_state" in status
    assert "nav" in status
    assert "pnl" in status


def test_az_supreme_briefing_data_defaults():
    """_get_briefing_data returns valid defaults."""
    from integrations.openclaw_az_supreme_handler import AZSupremeOpenClawHandler
    handler = AZSupremeOpenClawHandler()
    data = handler._get_briefing_data()
    assert "date" in data
    assert "btc_price" in data
    assert "doctrine_state" in data


# ─── Skill Definitions ──────────────────────────────────────────────────

def test_skill_md_generation():
    """generate_skill_md produces valid YAML frontmatter."""
    from integrations.openclaw_barren_wuffet_skills import (
        BARREN_WUFFET_SKILLS,
        generate_skill_md,
    )
    assert len(BARREN_WUFFET_SKILLS) > 0
    # Pick first skill
    first_name = next(iter(BARREN_WUFFET_SKILLS))
    md = generate_skill_md(BARREN_WUFFET_SKILLS[first_name])
    assert md.startswith("---")
    assert f"name: {first_name}" in md


def test_write_all_skills_to_tmpdir():
    """write_all_skills writes SKILL.md files to a temp directory."""
    import tempfile

    from integrations.openclaw_barren_wuffet_skills import write_all_skills
    with tempfile.TemporaryDirectory() as tmpdir:
        written = write_all_skills(tmpdir)
        assert len(written) > 0
        # Verify at least one SKILL.md exists
        for d in written[:3]:
            md_path = Path(d) / "SKILL.md"
            assert md_path.exists(), f"SKILL.md not found in {d}"


# ─── Deprecation Warning ────────────────────────────────────────────────

def test_deprecated_openclaw_skills_warns():
    """Importing openclaw_skills emits a DeprecationWarning."""
    import importlib

    import integrations.openclaw_skills as mod
    # Force re-import to trigger warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.reload(mod)
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "deprecated" in str(dep_warnings[0].message).lower()
