#!/usr/bin/env python3
"""Focused tests for secret-aware config loading."""

from pathlib import Path

from shared import config_loader


def test_get_env_uses_file_value(tmp_path, monkeypatch):
    secret_file = tmp_path / 'unusual_whales.txt'
    secret_file.write_text('file-based-secret\n', encoding='utf-8')

    monkeypatch.delenv('UNUSUAL_WHALES_API_KEY', raising=False)
    monkeypatch.setenv('UNUSUAL_WHALES_API_KEY_FILE', str(secret_file))

    value = config_loader.get_env('UNUSUAL_WHALES_API_KEY')

    assert value == 'file-based-secret'


def test_get_env_prefers_explicit_env_over_file(tmp_path, monkeypatch):
    secret_file = tmp_path / 'unusual_whales.txt'
    secret_file.write_text('file-based-secret\n', encoding='utf-8')

    monkeypatch.setenv('UNUSUAL_WHALES_API_KEY', 'env-secret')
    monkeypatch.setenv('UNUSUAL_WHALES_API_KEY_FILE', str(secret_file))

    value = config_loader.get_env('UNUSUAL_WHALES_API_KEY')

    assert value == 'env-secret'
