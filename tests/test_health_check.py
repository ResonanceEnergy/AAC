#!/usr/bin/env python3
"""
Tests for scripts/health_check.py — fast, no external services required.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import scripts.health_check as hc  # noqa: E402 — path setup above


class TestPythonVersion:
    def test_current_python_passes(self):
        assert hc.check_python_version() is True

    def test_old_python_fails(self):
        """check_python_version returns False for Python < 3.9."""
        from collections import namedtuple
        VI = namedtuple("version_info", "major minor micro releaselevel serial")
        fake = VI(2, 7, 18, "final", 0)
        with patch.object(sys, "version_info", fake):
            assert hc.check_python_version() is False


class TestEnvFile:
    def test_env_present(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text("KEY=val\n")
        monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
        assert hc.check_env_file() is True

    def test_env_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
        assert hc.check_env_file() is False


class TestVenv:
    def test_venv_exists(self, monkeypatch):
        monkeypatch.setattr(hc, "PROJECT_ROOT", PROJECT_ROOT)
        assert hc.check_venv() is True


class TestCoreImports:
    def test_no_crashes(self):
        """check_core_imports should complete without raising."""
        failures = hc.check_core_imports()
        assert isinstance(failures, int)


class TestRequiredPackages:
    def test_no_crashes(self):
        failures = hc.check_required_packages()
        assert isinstance(failures, int)


class TestDirectoryStructure:
    def test_key_dirs_exist(self):
        failures = hc.check_directory_structure()
        # strategies, tests, shared must exist
        assert failures <= 2  # at most 2 optional dirs might be missing


class TestMain:
    def test_main_returns_int(self):
        rc = hc.main()
        assert rc in (0, 1)
