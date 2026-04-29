# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Tests for ThreadSafeCache operations.
"""

import pytest
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from byol.common.caching import ThreadSafeCache


class TestThreadSafeCache:
    """Tests for ThreadSafeCache class."""

    def test_basic_get_set(self):
        """Test basic get/set operations."""
        cache: ThreadSafeCache[str, int] = ThreadSafeCache(maxsize=10)
        
        # Set and retrieve a value
        cache.set("key1", 100)
        assert cache.get("key1") == 100
        
        # Missing key returns None
        assert cache.get("missing") is None
        
        # Missing key with default
        assert cache.get("missing", default=42) == 42

    def test_lru_eviction(self):
        """Test that LRU eviction works correctly."""
        cache: ThreadSafeCache[str, int] = ThreadSafeCache(maxsize=3)
        
        # Fill the cache
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        
        # Access "a" to make it most recently used
        cache.get("a")
        
        # Add a new item - should evict "b" (least recently used)
        cache.set("d", 4)
        
        assert cache.get("a") == 1  # Still present (was accessed)
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == 3  # Still present
        assert cache.get("d") == 4  # Newly added

    def test_get_or_create(self):
        """Test get_or_create factory pattern."""
        cache: ThreadSafeCache[str, str] = ThreadSafeCache(maxsize=10)
        call_count = 0
        
        def factory():
            nonlocal call_count
            call_count += 1
            return f"created_{call_count}"
        
        # First call creates
        result1 = cache.get_or_create("key", factory)
        assert result1 == "created_1"
        assert call_count == 1
        
        # Second call returns cached value
        result2 = cache.get_or_create("key", factory)
        assert result2 == "created_1"
        assert call_count == 1  # Factory not called again

    def test_thread_safety(self):
        """Test that cache is thread-safe under concurrent access."""
        cache: ThreadSafeCache[str, int] = ThreadSafeCache(maxsize=100)
        errors = []
        
        def worker(thread_id: int):
            try:
                for i in range(100):
                    key = f"key_{thread_id}_{i}"
                    cache.set(key, i)
                    value = cache.get(key)
                    if value != i:
                        errors.append(f"Thread {thread_id}: expected {i}, got {value}")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")
        
        # Run 10 threads concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions
        
        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_get_or_create_thread_safety(self):
        """Test get_or_create doesn't call factory multiple times for same key."""
        cache: ThreadSafeCache[str, int] = ThreadSafeCache(maxsize=10)
        call_count = 0
        lock = threading.Lock()
        
        def slow_factory():
            nonlocal call_count
            with lock:
                call_count += 1
            # Simulate slow initialization
            import time
            time.sleep(0.01)
            return 42
        
        # Launch multiple threads trying to get_or_create the same key
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(cache.get_or_create, "same_key", slow_factory)
                for _ in range(10)
            ]
            results = [f.result() for f in as_completed(futures)]
        
        # All should get the same value
        assert all(r == 42 for r in results)
        # Factory should only be called once
        assert call_count == 1

    def test_clear(self):
        """Test cache clearing."""
        cache: ThreadSafeCache[str, int] = ThreadSafeCache(maxsize=10)
        
        cache.set("a", 1)
        cache.set("b", 2)
        
        assert cache.get("a") == 1
        
        cache.clear()
        
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_stats(self):
        """Test cache statistics."""
        cache: ThreadSafeCache[str, int] = ThreadSafeCache(maxsize=10, name="test_cache")
        
        # Initial stats
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        
        # Miss
        cache.get("missing")
        stats = cache.stats()
        assert stats["misses"] == 1
        
        # Set and hit
        cache.set("key", 1)
        cache.get("key")
        stats = cache.stats()
        assert stats["hits"] == 1

    def test_contains(self):
        """Test __contains__ method."""
        cache: ThreadSafeCache[str, int] = ThreadSafeCache(maxsize=10)
        
        assert "key" not in cache
        
        cache.set("key", 1)
        
        assert "key" in cache

    def test_len(self):
        """Test size method."""
        cache: ThreadSafeCache[str, int] = ThreadSafeCache(maxsize=10)
        
        assert cache.size() == 0
        
        cache.set("a", 1)
        cache.set("b", 2)
        
        assert cache.size() == 2
