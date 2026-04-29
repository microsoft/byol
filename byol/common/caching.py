# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Thread-safe caching utilities for BYOL.

Provides bounded, thread-safe caches for expensive objects like
translator instances, loaded models, and tokenizers.

Usage:
    from byol.common.caching import ThreadSafeCache, cached_with_key
    
    # Create a bounded cache
    cache = ThreadSafeCache(maxsize=10)
    cache.set("key", expensive_object)
    obj = cache.get("key")
    
    # Or use the decorator
    @cached_with_key(lambda model, lang: f"{model}:{lang}")
    def create_translator(model: str, lang: str):
        return Translator(model, lang)
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    TypeVar,
    Hashable,
    ParamSpec,
)

from byol.common.logging import get_logger

logger = get_logger(__name__)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
P = ParamSpec("P")
R = TypeVar("R")


class ThreadSafeCache(Generic[K, V]):
    """
    A thread-safe, bounded LRU cache.
    
    Features:
    - Thread-safe via RLock (reentrant for nested calls)
    - Bounded size with LRU eviction
    - Type-safe with generics
    - Logging of cache operations
    
    Args:
        maxsize: Maximum number of items to store. None for unbounded (not recommended).
        name: Optional name for logging purposes.
    
    Example:
        cache: ThreadSafeCache[str, Translator] = ThreadSafeCache(maxsize=10)
        cache.set("gpt-5:Spanish", translator)
        t = cache.get("gpt-5:Spanish")
    """
    
    def __init__(self, maxsize: Optional[int] = 100, name: Optional[str] = None):
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.RLock()
        self._maxsize = maxsize
        self._name = name or "cache"
        self._hits = 0
        self._misses = 0
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get an item from the cache.
        
        Moves the item to the end (most recently used) if found.
        
        Args:
            key: Cache key.
            default: Value to return if key not found.
        
        Returns:
            Cached value or default.
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                logger.debug(f"Cache hit: {self._name}[{key}]")
                return self._cache[key]
            self._misses += 1
            logger.debug(f"Cache miss: {self._name}[{key}]")
            return default
    
    def set(self, key: K, value: V) -> None:
        """
        Store an item in the cache.
        
        If the cache is at capacity, evicts the least recently used item.
        
        Args:
            key: Cache key.
            value: Value to store.
        """
        with self._lock:
            if key in self._cache:
                # Update existing and move to end
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                # Add new item
                self._cache[key] = value
                
                # Evict if over capacity
                if self._maxsize is not None and len(self._cache) > self._maxsize:
                    oldest_key, _ = self._cache.popitem(last=False)
                    logger.debug(f"Cache eviction: {self._name}[{oldest_key}]")
            
            logger.debug(f"Cache set: {self._name}[{key}]")
    
    def __contains__(self, key: K) -> bool:
        """Check if key is in cache (does not affect LRU order)."""
        with self._lock:
            return key in self._cache
    
    def delete(self, key: K) -> bool:
        """
        Remove an item from the cache.
        
        Args:
            key: Cache key to remove.
        
        Returns:
            True if item was removed, False if key not found.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache delete: {self._name}[{key}]")
                return True
            return False
    
    def clear(self) -> None:
        """Remove all items from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info(f"Cache cleared: {self._name}")
    
    def size(self) -> int:
        """Get current number of items in cache."""
        with self._lock:
            return len(self._cache)
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with hits, misses, hit_rate, size, and maxsize.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "name": self._name,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "maxsize": self._maxsize,
            }
    
    def keys(self) -> list[K]:
        """Get list of all keys in cache (in LRU order, oldest first)."""
        with self._lock:
            return list(self._cache.keys())
    
    def get_or_create(self, key: K, factory: Callable[[], V]) -> V:
        """
        Get an item from cache, or create it using the factory if not present.
        
        This is atomic - the factory will only be called once even if multiple
        threads request the same key simultaneously.
        
        Args:
            key: Cache key.
            factory: Callable that creates the value if not cached.
        
        Returns:
            Cached or newly created value.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            
            self._misses += 1
            value = factory()
            self.set(key, value)
            return value


def cached_with_key(
    key_func: Callable[P, str],
    cache: Optional[ThreadSafeCache[str, R]] = None,
    maxsize: int = 100,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that caches function results using a custom key function.
    
    Args:
        key_func: Function that takes the same arguments as the decorated function
                  and returns a cache key string.
        cache: Optional ThreadSafeCache instance to use. Creates a new one if None.
        maxsize: Max cache size if creating a new cache.
    
    Returns:
        Decorator function.
    
    Example:
        @cached_with_key(lambda model, lang: f"{model}:{lang}")
        def create_translator(model: str, lang: str) -> Translator:
            return Translator(model, lang)
    """
    _cache = cache or ThreadSafeCache[str, R](maxsize=maxsize)
    
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            key = key_func(*args, **kwargs)
            return _cache.get_or_create(key, lambda: func(*args, **kwargs))
        
        # Expose cache for testing/management
        wrapper.cache = _cache  # type: ignore
        wrapper.cache_clear = _cache.clear  # type: ignore
        wrapper.cache_stats = _cache.stats  # type: ignore
        
        return wrapper
    
    return decorator
