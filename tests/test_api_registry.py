#!/usr/bin/env python3
"""Tests for secret-aware API registry status checks."""

from tools import api_registry


def test_check_env_var_accepts_file_backed_secret(tmp_path, monkeypatch):
    secret_file = tmp_path / 'uw.txt'
    secret_file.write_text('secret-from-file\n', encoding='utf-8')

    monkeypatch.delenv('UNUSUAL_WHALES_API_KEY', raising=False)
    monkeypatch.setenv('UNUSUAL_WHALES_API_KEY_FILE', str(secret_file))

    assert api_registry.check_env_var('UNUSUAL_WHALES_API_KEY') == '✅ SET'


def test_check_env_var_reports_missing_when_unset(monkeypatch):
    monkeypatch.delenv('UNUSUAL_WHALES_API_KEY', raising=False)
    monkeypatch.delenv('UNUSUAL_WHALES_API_KEY_FILE', raising=False)

    assert api_registry.check_env_var('UNUSUAL_WHALES_API_KEY') == '❌ MISSING'