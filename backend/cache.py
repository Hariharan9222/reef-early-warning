"""
backend/cache.py
Caching layer supporting Redis with an automatic in-memory TTL fallback.
Ensures zero disruption if Redis is offline during local testing or CI.
"""

import time
import json
import logging
from typing import Optional, Any
from backend.config import settings

logger = logging.getLogger("coralguard.cache")

class InMemoryCache:
    """Thread-safe in-memory cache with Time-To-Live (TTL) expiration."""
    def __init__(self):
        self._store = {}
        self._expiry = {}

    def get(self, key: str) -> Optional[str]:
        now = time.time()
        if key in self._expiry and now > self._expiry[key]:
            del self._store[key]
            del self._expiry[key]
            return None
        return self._store.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None):
        self._store[key] = value
        if ex:
            self._expiry[key] = time.time() + ex
        elif key in self._expiry:
            del self._expiry[key]

    def delete(self, key: str):
        self._store.pop(key, None)
        self._expiry.pop(key, None)

    def clear(self):
        self._store.clear()
        self._expiry.clear()

class CacheManager:
    """Unified cache interface routing to Redis if available, or InMemoryCache fallback."""
    def __init__(self, redis_url: str = settings.REDIS_URL, default_ttl: int = settings.CACHE_DEFAULT_TTL):
        self.default_ttl = default_ttl
        self.redis_client = None
        self.in_memory = InMemoryCache()
        self.is_redis = False

        if not redis_url or str(redis_url).strip().lower() in ("none", "disabled", "", "false", "0"):
            logger.info("Using in-memory TTL cache.")
            return

        try:
            import redis
            client = redis.from_url(redis_url, socket_timeout=0.5, socket_connect_timeout=0.5)
            client.ping()
            self.redis_client = client
            self.is_redis = True
            logger.info(f"Connected to Redis cache at {redis_url}")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}). Falling back to local in-memory TTL cache.")
            self.is_redis = False

    def get_json(self, key: str) -> Optional[Any]:
        try:
            raw = self.redis_client.get(key) if self.is_redis else self.in_memory.get(key)
            if raw:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                return json.loads(raw)
        except Exception as e:
            logger.warning(f"Cache get error for key '{key}': {e}")
        return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None):
        ttl = ttl if ttl is not None else self.default_ttl
        try:
            serialized = json.dumps(value)
            if self.is_redis:
                self.redis_client.set(key, serialized, ex=ttl)
            else:
                self.in_memory.set(key, serialized, ex=ttl)
        except Exception as e:
            logger.warning(f"Cache set error for key '{key}': {e}")

    def delete(self, key: str):
        try:
            if self.is_redis:
                self.redis_client.delete(key)
            else:
                self.in_memory.delete(key)
        except Exception as e:
            logger.warning(f"Cache delete error for key '{key}': {e}")

cache = CacheManager()
