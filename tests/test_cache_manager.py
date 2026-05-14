from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared import cache_manager as cm
from shared.cache_manager import (
    CacheEntry,
    CacheManager,
    LocalCache,
    RedisCache,
    get_cache,
)


# ─────────────────────────────────────────────────────────────────────────────
# CacheEntry
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheEntry:
    def test_defaults(self):
        e = CacheEntry(key="k", value=1, created_at=datetime.now())
        assert e.expires_at is None
        assert e.hits == 0

    def test_is_expired_no_expiry(self):
        e = CacheEntry(key="k", value=1, created_at=datetime.now())
        assert e.is_expired is False

    def test_is_expired_future(self):
        e = CacheEntry(
            key="k", value=1, created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=10),
        )
        assert e.is_expired is False

    def test_is_expired_past(self):
        e = CacheEntry(
            key="k", value=1, created_at=datetime.now(),
            expires_at=datetime.now() - timedelta(seconds=1),
        )
        assert e.is_expired is True


# ─────────────────────────────────────────────────────────────────────────────
# LocalCache
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalCache:
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        c = LocalCache()
        assert await c.set("k", "v") is True
        assert await c.get("k") == "v"

    @pytest.mark.asyncio
    async def test_get_missing(self):
        c = LocalCache()
        assert await c.get("missing") is None

    @pytest.mark.asyncio
    async def test_hits_incremented(self):
        c = LocalCache()
        await c.set("k", 1)
        await c.get("k")
        await c.get("k")
        assert c._cache["k"].hits == 2

    @pytest.mark.asyncio
    async def test_ttl_expiry(self):
        c = LocalCache()
        await c.set("k", "v", ttl_seconds=1)
        # force expiry by mutating expires_at
        c._cache["k"].expires_at = datetime.now() - timedelta(seconds=1)
        assert await c.get("k") is None
        assert "k" not in c._cache  # auto-deleted

    @pytest.mark.asyncio
    async def test_no_ttl_persists(self):
        c = LocalCache()
        await c.set("k", "v")
        assert c._cache["k"].expires_at is None

    @pytest.mark.asyncio
    async def test_delete_existing(self):
        c = LocalCache()
        await c.set("k", "v")
        assert await c.delete("k") is True
        assert "k" not in c._cache

    @pytest.mark.asyncio
    async def test_delete_missing(self):
        c = LocalCache()
        assert await c.delete("nope") is False

    @pytest.mark.asyncio
    async def test_exists_true(self):
        c = LocalCache()
        await c.set("k", "v")
        assert await c.exists("k") is True

    @pytest.mark.asyncio
    async def test_exists_false_missing(self):
        c = LocalCache()
        assert await c.exists("nope") is False

    @pytest.mark.asyncio
    async def test_exists_false_when_expired(self):
        c = LocalCache()
        await c.set("k", "v", ttl_seconds=1)
        c._cache["k"].expires_at = datetime.now() - timedelta(seconds=1)
        assert await c.exists("k") is False
        assert "k" not in c._cache

    @pytest.mark.asyncio
    async def test_clear_returns_count(self):
        c = LocalCache()
        await c.set("a", 1)
        await c.set("b", 2)
        assert await c.clear() == 2
        assert len(c._cache) == 0

    @pytest.mark.asyncio
    async def test_clear_empty(self):
        c = LocalCache()
        assert await c.clear() == 0

    @pytest.mark.asyncio
    async def test_evict_when_over_capacity(self):
        c = LocalCache(max_size=10)
        for i in range(10):
            await c.set(f"k{i}", i)
        assert len(c._cache) == 10
        # adding one more triggers eviction (10/10 = remove 1)
        await c.set("k10", 10)
        # 10% of 10 = 1 evicted, then 1 added → still 10
        assert len(c._cache) == 10
        # k0 (oldest) should be evicted
        assert "k0" not in c._cache

    @pytest.mark.asyncio
    async def test_evict_oldest_empty_noop(self):
        c = LocalCache()
        c._evict_oldest()  # no error
        assert len(c._cache) == 0


# ─────────────────────────────────────────────────────────────────────────────
# RedisCache (no real connection)
# ─────────────────────────────────────────────────────────────────────────────

class TestRedisCacheUnconnected:
    """All methods must no-op gracefully when not connected."""

    def test_init_defaults(self):
        r = RedisCache()
        assert r.prefix == "acc:"
        assert r.default_ttl == 300
        assert r._connected is False

    def test_key_with_prefix(self):
        r = RedisCache(prefix="x:")
        assert r._key("foo") == "x:foo"

    @pytest.mark.asyncio
    async def test_get_returns_none_when_disconnected(self):
        r = RedisCache()
        assert await r.get("k") is None

    @pytest.mark.asyncio
    async def test_set_returns_false_when_disconnected(self):
        r = RedisCache()
        assert await r.set("k", "v") is False

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_disconnected(self):
        r = RedisCache()
        assert await r.delete("k") is False

    @pytest.mark.asyncio
    async def test_exists_returns_false_when_disconnected(self):
        r = RedisCache()
        assert await r.exists("k") is False

    @pytest.mark.asyncio
    async def test_clear_returns_zero_when_disconnected(self):
        r = RedisCache()
        assert await r.clear() == 0


class TestRedisCacheConnectFailure:
    @pytest.mark.asyncio
    async def test_connect_returns_false_when_redis_missing(self):
        r = RedisCache()
        # Simulate ImportError by hiding the redis.asyncio module
        with patch.dict(sys.modules, {"redis.asyncio": None}):
            with patch("builtins.__import__", side_effect=ImportError("no redis")):
                result = await r.connect()
        assert result is False
        assert r._connected is False

    @pytest.mark.asyncio
    async def test_connect_returns_false_on_ping_error(self):
        r = RedisCache()
        fake_redis_module = types.ModuleType("redis.asyncio")
        fake_client = MagicMock()
        fake_client.ping = AsyncMock(side_effect=RuntimeError("boom"))
        fake_redis_module.from_url = MagicMock(return_value=fake_client)
        with patch.dict(sys.modules, {"redis.asyncio": fake_redis_module}):
            result = await r.connect()
        assert result is False
        assert r._connected is False


class TestRedisCacheConnected:
    """Test behavior when _connected=True with mocked client."""

    def _make_cache(self, client):
        r = RedisCache(prefix="t:")
        r._client = client
        r._connected = True
        return r

    @pytest.mark.asyncio
    async def test_get_returns_deserialized_value(self):
        client = MagicMock()
        client.get = AsyncMock(return_value=b'{"a": 1}')
        r = self._make_cache(client)
        assert await r.get("k") == {"a": 1}
        client.get.assert_awaited_once_with("t:k")

    @pytest.mark.asyncio
    async def test_get_none_when_no_data(self):
        client = MagicMock()
        client.get = AsyncMock(return_value=None)
        r = self._make_cache(client)
        assert await r.get("k") is None

    @pytest.mark.asyncio
    async def test_get_swallows_exception(self):
        client = MagicMock()
        client.get = AsyncMock(side_effect=RuntimeError("fail"))
        r = self._make_cache(client)
        assert await r.get("k") is None

    @pytest.mark.asyncio
    async def test_set_uses_default_ttl(self):
        client = MagicMock()
        client.setex = AsyncMock(return_value=True)
        r = self._make_cache(client)
        assert await r.set("k", {"v": 1}) is True
        args = client.setex.await_args
        assert args.args[0] == "t:k"
        assert args.args[1] == r.default_ttl

    @pytest.mark.asyncio
    async def test_set_uses_custom_ttl(self):
        client = MagicMock()
        client.setex = AsyncMock(return_value=True)
        r = self._make_cache(client)
        await r.set("k", "v", ttl_seconds=42)
        assert client.setex.await_args.args[1] == 42

    @pytest.mark.asyncio
    async def test_set_swallows_exception(self):
        client = MagicMock()
        client.setex = AsyncMock(side_effect=RuntimeError("fail"))
        r = self._make_cache(client)
        assert await r.set("k", "v") is False

    @pytest.mark.asyncio
    async def test_delete_returns_true_when_count_positive(self):
        client = MagicMock()
        client.delete = AsyncMock(return_value=1)
        r = self._make_cache(client)
        assert await r.delete("k") is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_count_zero(self):
        client = MagicMock()
        client.delete = AsyncMock(return_value=0)
        r = self._make_cache(client)
        assert await r.delete("k") is False

    @pytest.mark.asyncio
    async def test_delete_swallows_exception(self):
        client = MagicMock()
        client.delete = AsyncMock(side_effect=RuntimeError("fail"))
        r = self._make_cache(client)
        assert await r.delete("k") is False

    @pytest.mark.asyncio
    async def test_exists_true(self):
        client = MagicMock()
        client.exists = AsyncMock(return_value=1)
        r = self._make_cache(client)
        assert await r.exists("k") is True

    @pytest.mark.asyncio
    async def test_exists_false(self):
        client = MagicMock()
        client.exists = AsyncMock(return_value=0)
        r = self._make_cache(client)
        assert await r.exists("k") is False

    @pytest.mark.asyncio
    async def test_exists_swallows_exception(self):
        client = MagicMock()
        client.exists = AsyncMock(side_effect=RuntimeError("fail"))
        r = self._make_cache(client)
        assert await r.exists("k") is False

    @pytest.mark.asyncio
    async def test_clear_deletes_keys(self):
        client = MagicMock()
        client.keys = AsyncMock(return_value=[b"t:a", b"t:b"])
        client.delete = AsyncMock(return_value=2)
        r = self._make_cache(client)
        assert await r.clear() == 2

    @pytest.mark.asyncio
    async def test_clear_returns_zero_when_no_keys(self):
        client = MagicMock()
        client.keys = AsyncMock(return_value=[])
        r = self._make_cache(client)
        assert await r.clear() == 0

    @pytest.mark.asyncio
    async def test_clear_swallows_exception(self):
        client = MagicMock()
        client.keys = AsyncMock(side_effect=RuntimeError("fail"))
        r = self._make_cache(client)
        assert await r.clear() == 0

    @pytest.mark.asyncio
    async def test_get_or_set_returns_cached(self):
        client = MagicMock()
        client.get = AsyncMock(return_value=b'42')
        r = self._make_cache(client)
        factory = MagicMock(return_value=99)
        result = await r.get_or_set("k", factory)
        assert result == 42
        factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_set_calls_sync_factory(self):
        client = MagicMock()
        client.get = AsyncMock(return_value=None)
        client.setex = AsyncMock(return_value=True)
        r = self._make_cache(client)
        factory = MagicMock(return_value=99)
        result = await r.get_or_set("k", factory, ttl_seconds=60)
        assert result == 99
        factory.assert_called_once()
        client.setex.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_or_set_calls_async_factory(self):
        client = MagicMock()
        client.get = AsyncMock(return_value=None)
        client.setex = AsyncMock(return_value=True)
        r = self._make_cache(client)

        async def afactory():
            return "computed"

        result = await r.get_or_set("k", afactory)
        assert result == "computed"

    @pytest.mark.asyncio
    async def test_disconnect_closes_client(self):
        client = MagicMock()
        client.close = AsyncMock()
        r = self._make_cache(client)
        await r.disconnect()
        client.close.assert_awaited_once()
        assert r._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_noop_when_no_client(self):
        r = RedisCache()
        await r.disconnect()  # no error


# ─────────────────────────────────────────────────────────────────────────────
# CacheManager
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheManagerInit:
    def test_default_uses_redis(self):
        mgr = CacheManager()
        assert mgr._use_redis is True
        assert mgr._redis is not None
        assert isinstance(mgr._local, LocalCache)
        assert mgr._initialized is False

    def test_disable_redis(self):
        mgr = CacheManager(use_redis=False)
        assert mgr._redis is None
        assert mgr._use_redis is False

    def test_custom_redis_url(self):
        mgr = CacheManager(redis_url="redis://custom:1234")
        assert mgr._redis.url == "redis://custom:1234"


class TestCacheManagerInitialize:
    @pytest.mark.asyncio
    async def test_initialize_uses_local_when_redis_fails(self):
        mgr = CacheManager(use_redis=True)
        # Redis connect fails; manager must still flag initialized
        with patch.object(mgr._redis, "connect", AsyncMock(return_value=False)):
            result = await mgr.initialize()
        assert result is True
        assert mgr._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_uses_redis_when_available(self):
        mgr = CacheManager(use_redis=True)
        with patch.object(mgr._redis, "connect", AsyncMock(return_value=True)):
            result = await mgr.initialize()
        assert result is True
        assert mgr._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_local_only_mode(self):
        mgr = CacheManager(use_redis=False)
        result = await mgr.initialize()
        assert result is True
        assert mgr._initialized is True


class TestCacheManagerCacheSelection:
    def test_uses_local_when_redis_disabled(self):
        mgr = CacheManager(use_redis=False)
        assert mgr._cache is mgr._local

    def test_uses_local_when_redis_disconnected(self):
        mgr = CacheManager(use_redis=True)
        assert mgr._redis._connected is False
        assert mgr._cache is mgr._local

    def test_uses_redis_when_connected(self):
        mgr = CacheManager(use_redis=True)
        mgr._redis._connected = True
        assert mgr._cache is mgr._redis


class TestCacheManagerOps:
    @pytest.mark.asyncio
    async def test_get_set_delete_via_local(self):
        mgr = CacheManager(use_redis=False)
        await mgr.initialize()
        assert await mgr.set("k", "v") is True
        assert await mgr.get("k") == "v"
        assert await mgr.delete("k") is True
        assert await mgr.get("k") is None


class TestCacheManagerSpecialized:
    @pytest.mark.asyncio
    async def test_cache_and_get_ticker(self):
        mgr = CacheManager(use_redis=False)
        await mgr.initialize()
        await mgr.cache_ticker("ibkr", "AAPL", {"price": 150.0})
        result = await mgr.get_ticker("ibkr", "AAPL")
        assert result == {"price": 150.0}

    @pytest.mark.asyncio
    async def test_cache_and_get_orderbook(self):
        mgr = CacheManager(use_redis=False)
        await mgr.initialize()
        await mgr.cache_orderbook("ibkr", "SPY", {"bids": [], "asks": []})
        result = await mgr.get_orderbook("ibkr", "SPY")
        assert result == {"bids": [], "asks": []}

    @pytest.mark.asyncio
    async def test_cache_and_get_finding(self):
        mgr = CacheManager(use_redis=False)
        await mgr.initialize()
        await mgr.cache_finding("scout", "TSLA", {"score": 0.8})
        result = await mgr.get_finding("scout", "TSLA")
        assert result == {"score": 0.8}

    @pytest.mark.asyncio
    async def test_cache_and_get_session(self):
        mgr = CacheManager(use_redis=False)
        await mgr.initialize()
        await mgr.cache_session("sess-1", {"user": "alice"})
        result = await mgr.get_session("sess-1")
        assert result == {"user": "alice"}

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        mgr = CacheManager(use_redis=False)
        await mgr.initialize()
        assert await mgr.get_ticker("x", "Y") is None
        assert await mgr.get_orderbook("x", "Y") is None
        assert await mgr.get_finding("x", "Y") is None
        assert await mgr.get_session("missing") is None

    def test_key_constants(self):
        assert CacheManager.TICKER_KEY == "ticker:{exchange}:{symbol}"
        assert CacheManager.ORDERBOOK_KEY == "orderbook:{exchange}:{symbol}"
        assert CacheManager.FINDING_KEY == "finding:{agent}:{symbol}"
        assert CacheManager.SESSION_KEY == "session:{session_id}"


class TestCacheManagerClose:
    @pytest.mark.asyncio
    async def test_close_calls_redis_disconnect(self):
        mgr = CacheManager(use_redis=True)
        with patch.object(mgr._redis, "disconnect", AsyncMock()) as mock_dc:
            await mgr.close()
        mock_dc.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_noop_when_no_redis(self):
        mgr = CacheManager(use_redis=False)
        await mgr.close()  # no error


# ─────────────────────────────────────────────────────────────────────────────
# get_cache global singleton
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCacheSingleton:
    def setup_method(self):
        # reset module-level singleton before each test
        cm._cache_manager = None

    def teardown_method(self):
        cm._cache_manager = None

    @pytest.mark.asyncio
    async def test_returns_cache_manager(self):
        with patch.object(CacheManager, "initialize", AsyncMock(return_value=True)):
            mgr = await get_cache()
        assert isinstance(mgr, CacheManager)
        assert cm._cache_manager is mgr

    @pytest.mark.asyncio
    async def test_returns_same_instance(self):
        with patch.object(CacheManager, "initialize", AsyncMock(return_value=True)):
            a = await get_cache()
            b = await get_cache()
        assert a is b
