from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from shared import secrets_manager as sm_mod
from shared.secrets_manager import (
    EncryptionError,
    SecretsError,
    SecretsManager,
    ValidationResult,
    get_secret,
    get_secrets_manager,
    validate_order_side,
    validate_order_type,
    validate_price,
    validate_quantity,
    validate_symbol,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    """Patch get_project_path so secrets + salt land in tmp_path."""
    def fake_get_project_path(*parts):
        p = tmp_path.joinpath(*parts)
        return p

    # Patch where secrets_manager imports it (lazy import inside methods)
    monkeypatch.setattr(
        "shared.config_loader.get_project_path",
        fake_get_project_path,
        raising=False,
    )
    return tmp_path


@pytest.fixture(autouse=True)
def reset_singleton():
    # Reset module-level singleton between tests
    sm_mod._secrets_manager = None
    yield
    sm_mod._secrets_manager = None


def _make_sm(tmp_path, password="hunter2", auto_load=False):
    return SecretsManager(
        master_password=password,
        secrets_file=tmp_path / "secrets.enc",
        auto_load=auto_load,
    )


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_encryption_error_is_secrets_error(self):
        assert issubclass(EncryptionError, SecretsError)
        assert issubclass(SecretsError, Exception)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    def test_requires_crypto_available(self, tmp_paths):
        with patch.object(sm_mod, "CRYPTO_AVAILABLE", False):
            with pytest.raises(EncryptionError):
                SecretsManager(master_password="x", secrets_file=tmp_paths / "x.enc")

    def test_init_no_password(self, tmp_paths):
        sm = SecretsManager(secrets_file=tmp_paths / "no.enc", auto_load=False)
        assert sm._fernet is None
        assert sm._encrypted is False
        assert sm._secrets == {}

    def test_init_with_password_sets_fernet(self, tmp_paths):
        sm = _make_sm(tmp_paths)
        assert sm._fernet is not None
        assert sm._encrypted is True

    def test_auto_load_when_file_exists(self, tmp_paths):
        sm = _make_sm(tmp_paths)
        sm.set_secret("k", "v")
        sm.save()
        sm2 = SecretsManager(
            master_password="hunter2",
            secrets_file=tmp_paths / "secrets.enc",
            auto_load=True,
        )
        assert sm2.get_secret("k") == "v"

    def test_default_secrets_path(self, tmp_paths):
        sm = SecretsManager(
            master_password="hunter2",
            secrets_file=None,
            auto_load=False,
        )
        # default path uses get_project_path('data', 'secrets.enc') → tmp_path
        assert sm.secrets_file.name == "secrets.enc"


# ---------------------------------------------------------------------------
# Salt
# ---------------------------------------------------------------------------


class TestSalt:
    def test_salt_created_then_reused(self, tmp_paths):
        sm = _make_sm(tmp_paths)
        salt_file = tmp_paths / "data" / ".salt"
        assert salt_file.exists()
        salt1 = salt_file.read_bytes()
        # second instance reuses
        sm2 = _make_sm(tmp_paths, password="other")
        salt2 = (tmp_paths / "data" / ".salt").read_bytes()
        assert salt1 == salt2
        assert len(salt1) == 32
        # different fernets though (different password)
        assert sm._fernet is not sm2._fernet


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestSecretsCrud:
    def test_set_get(self, tmp_paths):
        sm = _make_sm(tmp_paths)
        sm.set_secret("api", "abc")
        assert sm.get_secret("api") == "abc"

    def test_get_default(self, tmp_paths):
        sm = _make_sm(tmp_paths)
        assert sm.get_secret("missing") == ""
        assert sm.get_secret("missing", "fallback") == "fallback"

    def test_has_secret(self, tmp_paths):
        sm = _make_sm(tmp_paths)
        sm.set_secret("k", "v")
        assert sm.has_secret("k") is True
        assert sm.has_secret("missing") is False

    def test_delete_present(self, tmp_paths):
        sm = _make_sm(tmp_paths)
        sm.set_secret("k", "v")
        assert sm.delete_secret("k") is True
        assert sm.has_secret("k") is False

    def test_delete_absent(self, tmp_paths):
        sm = _make_sm(tmp_paths)
        assert sm.delete_secret("missing") is False

    def test_list_secrets(self, tmp_paths):
        sm = _make_sm(tmp_paths)
        sm.set_secret("a", "1")
        sm.set_secret("b", "2")
        assert set(sm.list_secrets()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_paths):
        sm = _make_sm(tmp_paths)
        sm.set_secret("k1", "v1")
        sm.set_secret("k2", "v2")
        sm.save()
        # New instance, same password
        sm2 = SecretsManager(
            master_password="hunter2",
            secrets_file=tmp_paths / "secrets.enc",
            auto_load=False,
        )
        assert sm2.load() is True
        assert sm2.get_secret("k1") == "v1"
        assert sm2.get_secret("k2") == "v2"

    def test_save_without_encryption_raises(self, tmp_paths):
        sm = SecretsManager(secrets_file=tmp_paths / "x.enc", auto_load=False)
        with pytest.raises(EncryptionError):
            sm.save()

    def test_load_missing_file_returns_false(self, tmp_paths):
        sm = _make_sm(tmp_paths)
        sm.secrets_file = tmp_paths / "absent.enc"
        assert sm.load() is False

    def test_load_plaintext_format(self, tmp_paths):
        sm = SecretsManager(secrets_file=tmp_paths / "plain.enc", auto_load=False)
        data = json.dumps({"k": "v"}).encode()
        body = b"PLAIN:" + base64.b64encode(data)
        sm.secrets_file.write_bytes(body)
        assert sm.load() is True
        assert sm.get_secret("k") == "v"

    def test_load_encrypted_without_password_returns_false(self, tmp_paths):
        # save encrypted
        sm = _make_sm(tmp_paths)
        sm.set_secret("k", "v")
        sm.save()
        # new instance, no password → load() catches EncryptionError and returns False
        sm2 = SecretsManager(secrets_file=tmp_paths / "secrets.enc", auto_load=False)
        assert sm2.load() is False

    def test_load_wrong_password_returns_false(self, tmp_paths):
        sm = _make_sm(tmp_paths, password="right")
        sm.set_secret("k", "v")
        sm.save()
        sm2 = SecretsManager(
            master_password="wrong",
            secrets_file=tmp_paths / "secrets.enc",
            auto_load=False,
        )
        assert sm2.load() is False


# ---------------------------------------------------------------------------
# migrate_from_env
# ---------------------------------------------------------------------------


class TestMigrateFromEnv:
    def test_migrates_present_env_vars(self, tmp_paths, monkeypatch):
        monkeypatch.setenv("MY_KEY", "abc123")
        monkeypatch.setenv("MY_OTHER", "xyz")
        sm = _make_sm(tmp_paths)
        n = sm.migrate_from_env({"my_key": "MY_KEY", "my_other": "MY_OTHER"})
        assert n == 2
        assert sm.get_secret("my_key") == "abc123"
        assert sm.get_secret("my_other") == "xyz"

    def test_skips_missing_env_vars(self, tmp_paths, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        sm = _make_sm(tmp_paths)
        n = sm.migrate_from_env({"missing": "MISSING_VAR"})
        assert n == 0
        assert sm.has_secret("missing") is False

    def test_does_not_save_when_zero(self, tmp_paths, monkeypatch):
        monkeypatch.delenv("X", raising=False)
        sm = _make_sm(tmp_paths)
        sm.migrate_from_env({"x": "X"})
        # File should not exist since save() not called
        assert not sm.secrets_file.exists()

    def test_saves_when_any_migrated(self, tmp_paths, monkeypatch):
        monkeypatch.setenv("YES", "1")
        sm = _make_sm(tmp_paths)
        sm.migrate_from_env({"yes": "YES"})
        assert sm.secrets_file.exists()


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_secrets_manager_creates_once(self, tmp_paths, monkeypatch):
        monkeypatch.setenv("ACC_MASTER_PASSWORD", "envpw")
        a = get_secrets_manager()
        b = get_secrets_manager()
        assert a is b

    def test_get_secrets_manager_uses_explicit_password(self, tmp_paths):
        a = get_secrets_manager(master_password="explicit")
        assert a._fernet is not None

    def test_get_secret_convenience(self, tmp_paths, monkeypatch):
        monkeypatch.setenv("ACC_MASTER_PASSWORD", "envpw")
        sm = get_secrets_manager()
        sm.set_secret("foo", "bar")
        assert get_secret("foo") == "bar"
        assert get_secret("missing", "default") == "default"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_defaults(self):
        r = ValidationResult(True)
        assert r.valid is True
        assert r.error == ""
        assert r.sanitized_value is None


class TestValidateSymbol:
    def test_valid_spot(self):
        r = validate_symbol("btc/usdt")
        assert r.valid is True
        assert r.sanitized_value == "BTC/USDT"

    def test_valid_compact(self):
        assert validate_symbol("BTCUSDT").valid is True

    def test_valid_perp(self):
        assert validate_symbol("BTC-PERP").valid is True

    def test_empty(self):
        r = validate_symbol("")
        assert r.valid is False
        assert "empty" in r.error.lower()

    def test_too_short(self):
        r = validate_symbol("AB")
        assert r.valid is False

    def test_too_long(self):
        r = validate_symbol("A" * 21)
        assert r.valid is False

    def test_invalid_chars(self):
        r = validate_symbol("BTC$USD")
        assert r.valid is False
        assert "invalid characters" in r.error.lower()

    def test_strips_and_uppercases(self):
        r = validate_symbol("  eth/btc  ")
        assert r.valid is True
        assert r.sanitized_value == "ETH/BTC"


class TestValidateQuantity:
    def test_valid_int(self):
        r = validate_quantity(5)
        assert r.valid is True
        assert r.sanitized_value == 5.0

    def test_valid_float(self):
        r = validate_quantity(0.5)
        assert r.valid is True

    def test_not_a_number(self):
        r = validate_quantity("five")  # type: ignore[arg-type]
        assert r.valid is False

    def test_zero_rejected(self):
        assert validate_quantity(0).valid is False

    def test_below_min(self):
        assert validate_quantity(0.5, min_qty=1.0).valid is False

    def test_above_max(self):
        assert validate_quantity(1e10).valid is False

    def test_nan_rejected(self):
        assert validate_quantity(float("nan")).valid is False

    def test_inf_rejected(self):
        assert validate_quantity(float("inf")).valid is False


class TestValidatePrice:
    def test_none_allowed(self):
        r = validate_price(None, allow_none=True)
        assert r.valid is True
        assert r.sanitized_value is None

    def test_none_disallowed(self):
        assert validate_price(None, allow_none=False).valid is False

    def test_valid(self):
        r = validate_price(100.0)
        assert r.valid is True
        assert r.sanitized_value == 100.0

    def test_not_number(self):
        assert validate_price("100").valid is False  # type: ignore[arg-type]

    def test_zero_rejected(self):
        assert validate_price(0).valid is False

    def test_negative_rejected(self):
        assert validate_price(-1).valid is False

    def test_nan_rejected(self):
        assert validate_price(float("nan")).valid is False

    def test_inf_rejected(self):
        assert validate_price(float("inf")).valid is False


class TestValidateOrderSide:
    def test_buy(self):
        r = validate_order_side("BUY")
        assert r.valid is True
        assert r.sanitized_value == "buy"

    def test_sell(self):
        assert validate_order_side(" sell ").sanitized_value == "sell"

    def test_invalid(self):
        assert validate_order_side("hold").valid is False


class TestValidateOrderType:
    @pytest.mark.parametrize("ot", ["market", "limit", "stop_loss", "stop_limit", "take_profit"])
    def test_valid_types(self, ot):
        r = validate_order_type(ot.upper())
        assert r.valid is True
        assert r.sanitized_value == ot

    def test_invalid(self):
        assert validate_order_type("trailing").valid is False
