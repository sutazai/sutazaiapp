#!/usr/bin/env python3.11
"""Tests for the caching module."""

import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path

    Cache,
    CacheEntry,
    CachePolicy,
    CacheManager,
    CacheError,
    MemoryCache,
    FileCache,
    RedisCache,
)


@pytest.fixture
def cache_manager():
    """Create a test cache manager."""
    return CacheManager()


@pytest.fixture
def memory_cache():
    """Create a test memory cache."""
    return MemoryCache()


@pytest.fixture
def file_cache(tmp_path):
    """Create a test file cache."""
    return FileCache(cache_dir=tmp_path)


@pytest.fixture
def redis_cache():
    """Create a test Redis cache."""
    return RedisCache()


def test_cache_entry():
    """Test cache entry functionality."""
    # Test entry creation
    entry = CacheEntry(
        key="test_key",
        value="test_value",
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(hours=1),
    )
    assert entry.key == "test_key"
    assert entry.value == "test_value"
    assert entry.is_expired() is False

    # Test expired entry
    expired_entry = CacheEntry(
        key="expired_key",
        value="expired_value",
        created_at=datetime.now() - timedelta(hours=2),
        expires_at=datetime.now() - timedelta(hours=1),
    )
    assert expired_entry.is_expired() is True


def test_cache_policy():
    """Test cache policy functionality."""
    # Test policy creation
    policy = CachePolicy(
        max_size=1000,
        max_age=3600,
        max_entries=100,
    )
    assert policy.max_size == 1000
    assert policy.max_age == 3600
    assert policy.max_entries == 100

    # Test policy validation
    assert policy.validate_entry_size(500) is True
    assert policy.validate_entry_size(2000) is False

    # Test policy update
    policy.update(max_size=2000)
    assert policy.max_size == 2000


def test_memory_cache(memory_cache):
    """Test memory cache functionality."""
    # Test cache operations
    memory_cache.set("key1", "value1")
    assert memory_cache.get("key1") == "value1"

    # Test cache update
    memory_cache.set("key1", "value2")
    assert memory_cache.get("key1") == "value2"

    # Test cache deletion
    memory_cache.delete("key1")
    assert memory_cache.get("key1") is None

    # Test cache expiration
    memory_cache.set("key2", "value2", expires_in=1)
    time.sleep(2)
    assert memory_cache.get("key2") is None

    # Test cache clear
    memory_cache.set("key3", "value3")
    memory_cache.clear()
    assert memory_cache.get("key3") is None


def test_file_cache(file_cache):
    """Test file cache functionality."""
    # Test cache operations
    file_cache.set("key1", "value1")
    assert file_cache.get("key1") == "value1"

    # Test cache update
    file_cache.set("key1", "value2")
    assert file_cache.get("key1") == "value2"

    # Test cache deletion
    file_cache.delete("key1")
    assert file_cache.get("key1") is None

    # Test cache expiration
    file_cache.set("key2", "value2", expires_in=1)
    time.sleep(2)
    assert file_cache.get("key2") is None

    # Test cache clear
    file_cache.set("key3", "value3")
    file_cache.clear()
    assert file_cache.get("key3") is None


def test_redis_cache(redis_cache):
    """Test Redis cache functionality."""
    # Test cache operations
    redis_cache.set("key1", "value1")
    assert redis_cache.get("key1") == "value1"

    # Test cache update
    redis_cache.set("key1", "value2")
    assert redis_cache.get("key1") == "value2"

    # Test cache deletion
    redis_cache.delete("key1")
    assert redis_cache.get("key1") is None

    # Test cache expiration
    redis_cache.set("key2", "value2", expires_in=1)
    time.sleep(2)
    assert redis_cache.get("key2") is None

    # Test cache clear
    redis_cache.set("key3", "value3")
    redis_cache.clear()
    assert redis_cache.get("key3") is None


def test_cache_manager(cache_manager):
    """Test cache manager functionality."""
    # Test cache registration
    memory_cache = MemoryCache()
    cache_manager.register_cache("memory", memory_cache)
    assert "memory" in cache_manager.caches

    # Test cache operations through manager
    cache_manager.set("memory", "key1", "value1")
    assert cache_manager.get("memory", "key1") == "value1"

    # Test cache deletion through manager
    cache_manager.delete("memory", "key1")
    assert cache_manager.get("memory", "key1") is None

    # Test cache unregistration
    cache_manager.unregister_cache("memory")
    assert "memory" not in cache_manager.caches


def test_cache_policy_enforcement(cache_manager):
    """Test cache policy enforcement."""
    # Create cache with policy
    policy = CachePolicy(
        max_size=1000,
        max_age=3600,
        max_entries=2,
    )
    memory_cache = MemoryCache(policy=policy)
    cache_manager.register_cache("memory", memory_cache)

    # Test max entries
    cache_manager.set("memory", "key1", "value1")
    cache_manager.set("memory", "key2", "value2")
    cache_manager.set("memory", "key3", "value3")
    assert cache_manager.get("memory", "key1") is None  # First entry should be evicted
    assert cache_manager.get("memory", "key2") == "value2"
    assert cache_manager.get("memory", "key3") == "value3"

    # Test max size
    large_value = "x" * 2000
    cache_manager.set("memory", "key4", large_value)
    assert cache_manager.get("memory", "key4") is None  # Should be rejected due to size

    # Test max age
    cache_manager.set("memory", "key5", "value5", expires_in=1)
    time.sleep(2)
    assert cache_manager.get("memory", "key5") is None


def test_cache_serialization(file_cache):
    """Test cache serialization."""
    # Test complex object serialization
    data = {
        "string": "test",
        "number": 42,
        "boolean": True,
        "list": [1, 2, 3],
        "dict": {"key": "value"},
        "datetime": datetime.now(),
    }
    file_cache.set("complex", data)
    retrieved = file_cache.get("complex")
    assert retrieved == data

    # Test custom object serialization

    class CustomObject:
        def __init__(self, value):
            self.value = value

    obj = CustomObject("test")
    file_cache.set("custom", obj)
    retrieved = file_cache.get("custom")
    assert retrieved.value == "test"


def test_cache_error_handling(cache_manager):
    """Test cache error handling."""
    # Test non-existent cache
    with pytest.raises(CacheError):
        cache_manager.get("non_existent", "key")

    # Test invalid cache name
    with pytest.raises(CacheError):
        cache_manager.register_cache("", MemoryCache())

    # Test duplicate cache registration
    memory_cache = MemoryCache()
    cache_manager.register_cache("memory", memory_cache)
    with pytest.raises(CacheError):
        cache_manager.register_cache("memory", MemoryCache())

    # Test invalid cache type
    with pytest.raises(CacheError):
        cache_manager.register_cache("invalid", "not a cache")
