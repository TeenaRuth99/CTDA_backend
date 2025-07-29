# --- app/services/cachemanager.py ---
import time

class CacheManager:
    def __init__(self):
        # Initialize in-memory cache dictionary
        self.cache = {}

    def get(self, key: str):
        # Retrieve value from cache if it exists
        return self.cache.get(key)

    def set(self, key: str, value: str):
        # Store a value with the given key in cache
        self.cache[key] = value

    def clear(self):
        # Clear all entries from the cache
        self.cache.clear()

    def delete(self, key: str):
        # Delete a specific key from the cache if it exists
        if key in self.cache:
            del self.cache[key]

    def keys(self):
        # List all keys in the cache
        return list(self.cache.keys())
