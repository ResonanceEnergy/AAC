#!/usr/bin/env python3
"""Tests for API Key Provisioning Agent"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.api_key_agent import APIKeyAgent, APIDefinition, AgentReport, MANAGED_APIS


# ─── Unit Tests ─────────────────────────────────────────────────

class TestAPIKeyAgentDefinitions:
    """Test that all 7 APIs are properly defined."""

    def test_seven_apis_defined(self):
        assert len(MANAGED_APIS) == 7

    def test_all_apis_have_required_fields(self):
        for api in MANAGED_APIS:
            assert api.name, f"Missing name for {api.env_var}"
            assert api.env_var, f"Missing env_var for {api.name}"
            assert api.signup_url.startswith("https://"), f"Bad signup URL for {api.name}"
            assert api.base_url.startswith("https://"), f"Bad base_url for {api.name}"
            assert api.test_endpoint, f"Missing test_endpoint for {api.name}"
            assert api.free_tier, f"Missing free_tier description for {api.name}"

    def test_expected_api_names(self):
        names = {api.name for api in MANAGED_APIS}
        expected = {"FRED", "Polygon.io", "Finnhub", "Alpha Vantage", "NewsAPI", "Etherscan", "Tradier"}
        assert names == expected

    def test_expected_env_vars(self):
        env_vars = {api.env_var for api in MANAGED_APIS}
        expected = {
            "FRED_API_KEY", "POLYGON_API_KEY", "FINNHUB_API_KEY",
            "ALPHAVANTAGE_API_KEY", "NEWS_API_KEY", "ETHERSCAN_API_KEY",
            "TRADIER_API_KEY",
        }
        assert env_vars == expected


class TestAPIKeyAgentMissing:
    """Test missing key detection."""

    def test_get_missing_with_no_keys(self):
        # Clear all managed env vars AND prevent the constructor from reloading .env
        saved = {}
        for api in MANAGED_APIS:
            saved[api.env_var] = os.environ.pop(api.env_var, None)
        try:
            with patch("agents.api_key_agent.load_env_file"):
                agent = APIKeyAgent()
                missing = agent.get_missing()
                assert len(missing) == 7
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_get_missing_with_some_keys(self):
        saved = {}
        for api in MANAGED_APIS:
            saved[api.env_var] = os.environ.pop(api.env_var, None)
        os.environ["FRED_API_KEY"] = "test_key_123"
        try:
            with patch("agents.api_key_agent.load_env_file"):
                agent = APIKeyAgent()
                missing = agent.get_missing()
                assert len(missing) == 6
                missing_names = {m.name for m in missing}
                assert "FRED" not in missing_names
        finally:
            os.environ.pop("FRED_API_KEY", None)
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_get_configured_with_one_key(self):
        saved = {}
        for api in MANAGED_APIS:
            saved[api.env_var] = os.environ.pop(api.env_var, None)
        os.environ["FINNHUB_API_KEY"] = "test_key_abc"
        try:
            with patch("agents.api_key_agent.load_env_file"):
                agent = APIKeyAgent()
                configured = agent.get_configured()
                assert len(configured) == 1
                assert configured[0].name == "Finnhub"
        finally:
            os.environ.pop("FINNHUB_API_KEY", None)
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


class TestAPIKeyAgentAddKey:
    """Test adding keys to .env file."""

    def test_add_key_updates_existing_var(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False, encoding="utf-8") as f:
            f.write("FRED_API_KEY=\nOTHER=value\n")
            tmp_path = Path(f.name)
        try:
            agent = APIKeyAgent(env_path=tmp_path)
            result = agent.add_key("FRED_API_KEY", "abc123def")
            assert result is True
            content = tmp_path.read_text(encoding="utf-8")
            assert "FRED_API_KEY=abc123def" in content
            assert "OTHER=value" in content
        finally:
            tmp_path.unlink()

    def test_add_key_rejects_unmanaged_var(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False, encoding="utf-8") as f:
            f.write("FOO=bar\n")
            tmp_path = Path(f.name)
        try:
            agent = APIKeyAgent(env_path=tmp_path)
            result = agent.add_key("RANDOM_API_KEY", "xyz")
            assert result is False
        finally:
            tmp_path.unlink()

    def test_add_key_rejects_invalid_characters(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False, encoding="utf-8") as f:
            f.write("FRED_API_KEY=\n")
            tmp_path = Path(f.name)
        try:
            agent = APIKeyAgent(env_path=tmp_path)
            result = agent.add_key("FRED_API_KEY", "key with spaces")
            assert result is False
        finally:
            tmp_path.unlink()


class TestAPIKeyAgentValidation:
    """Test live validation with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_run_skips_validation_for_missing_keys(self):
        saved = {}
        for api in MANAGED_APIS:
            saved[api.env_var] = os.environ.pop(api.env_var, None)
        try:
            with patch("agents.api_key_agent.load_env_file"):
                agent = APIKeyAgent()
                report = await agent.run(validate=True)
                assert report.total == 7
                assert report.configured == 0
                assert report.missing == 7
                assert report.validated == 0
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    @pytest.mark.asyncio
    async def test_run_without_validation(self):
        saved = {}
        for api in MANAGED_APIS:
            saved[api.env_var] = os.environ.pop(api.env_var, None)
        os.environ["FRED_API_KEY"] = "test_key"
        try:
            with patch("agents.api_key_agent.load_env_file"):
                agent = APIKeyAgent()
                report = await agent.run(validate=False)
                fred = [r for r in report.results if r.api_name == "FRED"][0]
                assert fred.key_present is True
                assert fred.key_valid is True  # no-validate = trust presence
        finally:
            os.environ.pop("FRED_API_KEY", None)
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    @pytest.mark.asyncio
    async def test_validation_handles_401(self):
        """Mock a 401 response for an invalid key."""
        os.environ["POLYGON_API_KEY"] = "bad_key"
        try:
            mock_resp = AsyncMock()
            mock_resp.status = 401
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                agent = APIKeyAgent()
                polygon_def = [a for a in MANAGED_APIS if a.name == "Polygon.io"][0]
                result = await agent._check_api(polygon_def, validate=True)
                assert result.key_present is True
                assert result.key_valid is False
                assert "401" in result.error
        finally:
            os.environ.pop("POLYGON_API_KEY", None)

    @pytest.mark.asyncio
    async def test_validation_treats_429_as_valid(self):
        """Rate-limited means the key IS valid."""
        os.environ["FINNHUB_API_KEY"] = "real_key"
        try:
            mock_resp = AsyncMock()
            mock_resp.status = 429
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                agent = APIKeyAgent()
                finnhub_def = [a for a in MANAGED_APIS if a.name == "Finnhub"][0]
                result = await agent._check_api(finnhub_def, validate=True)
                assert result.key_valid is True
                assert "Rate limited" in result.sample_data
        finally:
            os.environ.pop("FINNHUB_API_KEY", None)


class TestAgentReport:
    """Test report formatting."""

    def test_summary_includes_all_sections(self):
        report = AgentReport(total=7, configured=2, validated=1, missing=5)
        from agents.api_key_agent import APIValidationResult
        report.results = [
            APIValidationResult("FRED", "FRED_API_KEY", True, True, 150.0, "Fed Funds: 4.33%"),
            APIValidationResult("Polygon.io", "POLYGON_API_KEY", True, False, 0, "", "Invalid key"),
            APIValidationResult("Finnhub", "FINNHUB_API_KEY", False, False, signup_url="https://finnhub.io", free_tier="60/min"),
        ]
        text = report.summary()
        assert "STATUS REPORT" in text
        assert "FRED" in text
        assert "Polygon.io" in text
        assert "Finnhub" in text
        assert "LIVE & VALIDATED" in text
        assert "CONFIGURED BUT FAILED" in text
        assert "MISSING KEYS" in text
