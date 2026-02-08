#!/usr/bin/env python3
"""
Redis Cache Manager
===================
Caching layer for market data, agent findings, and session state.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import os


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    hits: int = 0
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class LocalCache:
    """In-memory fallback cache when Redis is unavailable"""
    
    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self.logger = logging.getLogger("LocalCache")
    
    async def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        
        if entry.is_expired:
            del self._cache[key]
            return None
        
        entry.hits += 1
        return entry.value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        # Evict if at capacity
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        
        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            expires_at=expires_at,
        )
        return True
    
    async def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        entry = self._cache.get(key)
        if entry is None:
            return False
        if entry.is_expired:
            del self._cache[key]
            return False
        return True
    
    async def clear(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count
    
    def _evict_oldest(self):
        """Evict oldest entries"""
        if not self._cache:
            return
        
        # Sort by created_at and remove oldest 10%
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )
        
        to_remove = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:to_remove]:
            del self._cache[key]


class RedisCache:
    """Redis-backed cache implementation"""
    
    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "acc:",
        default_ttl: int = 300,  # 5 minutes
    ):
        self.url = url
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.logger = logging.getLogger("RedisCache")
        self._client = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            import redis.asyncio as redis
            self._client = redis.from_url(self.url)
            await self._client.ping()
            self._connected = True
            self.logger.info(f"Connected to Redis at {self.url}")
            return True
        except ImportError:
            self.logger.warning("redis package not installed")
            return False
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self._client:
            await self._client.close()
            self._connected = False
    
    def _key(self, key: str) -> str:
        """Add prefix to key"""
        return f"{self.prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._connected:
            return None
        
        try:
            data = await self._client.get(self._key(key))
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            self.logger.debug(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Set value in cache"""
        if not self._connected:
            return False
        
        try:
            ttl = ttl_seconds or self.default_ttl
            serialized = json.dumps(value, default=str)
            await self._client.setex(self._key(key), ttl, serialized)
            return True
        except Exception as e:
            self.logger.debug(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self._connected:
            return False
        
        try:
            result = await self._client.delete(self._key(key))
            return result > 0
        except Exception as e:
            self.logger.debug(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self._connected:
            return False
        
        try:
            return await self._client.exists(self._key(key)) > 0
        except Exception:
            return False
    
    async def clear(self, pattern: str = "*") -> int:
        """Clear keys matching pattern"""
        if not self._connected:
            return 0
        
        try:
            keys = await self._client.keys(self._key(pattern))
            if keys:
                return await self._client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.debug(f"Cache clear error: {e}")
            return 0
    
    async def get_or_set(
        self,
        key: str,
        factory: callable,
        ttl_seconds: Optional[int] = None,
    ) -> Any:
        """Get from cache or compute and cache result"""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Compute value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()
        
        await self.set(key, value, ttl_seconds)
        return value


class CacheManager:
    """
    Unified cache manager with Redis primary and local fallback.
    
    Provides caching for:
    - Market data (tickers, order books)
    - Agent findings
    - Session state
    - Configuration
    """
    
    # Cache key patterns
    TICKER_KEY = "ticker:{exchange}:{symbol}"
    ORDERBOOK_KEY = "orderbook:{exchange}:{symbol}"
    FINDING_KEY = "finding:{agent}:{symbol}"
    SESSION_KEY = "session:{session_id}"
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        use_redis: bool = True,
    ):
        self.logger = logging.getLogger("CacheManager")
        
        redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        self._redis = RedisCache(url=redis_url) if use_redis else None
        self._local = LocalCache()
        self._use_redis = use_redis
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize cache connections"""
        if self._use_redis and self._redis:
            if await self._redis.connect():
                self._initialized = True
                return True
        
        self.logger.info("Using local cache (Redis unavailable)")
        self._initialized = True
        return True
    
    @property
    def _cache(self):
        """Get active cache implementation"""
        if self._redis and self._redis._connected:
            return self._redis
        return self._local
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return await self._cache.get(key)
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 300) -> bool:
        """Set value in cache"""
        return await self._cache.set(key, value, ttl_seconds)
    
    async def delete(self, key: str) -> bool:
        """Delete from cache"""
        return await self._cache.delete(key)
    
    # Specialized cache methods
    
    async def cache_ticker(
        self,
        exchange: str,
        symbol: str,
        ticker_data: Dict,
        ttl: int = 10,  # 10 seconds for real-time data
    ) -> bool:
        """Cache ticker data"""
        key = self.TICKER_KEY.format(exchange=exchange, symbol=symbol)
        return await self.set(key, ticker_data, ttl)
    
    async def get_ticker(self, exchange: str, symbol: str) -> Optional[Dict]:
        """Get cached ticker"""
        key = self.TICKER_KEY.format(exchange=exchange, symbol=symbol)
        return await self.get(key)
    
    async def cache_orderbook(
        self,
        exchange: str,
        symbol: str,
        orderbook_data: Dict,
        ttl: int = 5,  # 5 seconds
    ) -> bool:
        """Cache order book"""
        key = self.ORDERBOOK_KEY.format(exchange=exchange, symbol=symbol)
        return await self.set(key, orderbook_data, ttl)
    
    async def get_orderbook(self, exchange: str, symbol: str) -> Optional[Dict]:
        """Get cached order book"""
        key = self.ORDERBOOK_KEY.format(exchange=exchange, symbol=symbol)
        return await self.get(key)
    
    async def cache_finding(
        self,
        agent: str,
        symbol: str,
        finding: Dict,
        ttl: int = 300,  # 5 minutes
    ) -> bool:
        """Cache agent finding"""
        key = self.FINDING_KEY.format(agent=agent, symbol=symbol)
        return await self.set(key, finding, ttl)
    
    async def get_finding(self, agent: str, symbol: str) -> Optional[Dict]:
        """Get cached finding"""
        key = self.FINDING_KEY.format(agent=agent, symbol=symbol)
        return await self.get(key)
    
    async def cache_session(
        self,
        session_id: str,
        session_data: Dict,
        ttl: int = 3600,  # 1 hour
    ) -> bool:
        """Cache session data"""
        key = self.SESSION_KEY.format(session_id=session_id)
        return await self.set(key, session_data, ttl)
    
    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        key = self.SESSION_KEY.format(session_id=session_id)
        return await self.get(key)
    
    async def close(self):
        """Close cache connections"""
        if self._redis:
            await self._redis.disconnect()


# Global cache instance
_cache_manager: Optional[CacheManager] = None


async def get_cache() -> CacheManager:
    """Get or create global cache manager"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.initialize()
    return _cache_manager
